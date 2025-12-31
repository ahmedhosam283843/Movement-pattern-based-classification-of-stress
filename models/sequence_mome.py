"""
MoME dataset/model/training loop for sequence.
References:
- Cătrună et al. (2025): Multi-stage mixture of movement experts, multi-task regularization (ID/speed).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import os

from preprocess import apply_augmentations


JOINTS = ["hip_flexion","knee_flexion","shoulder_angle","elbow_angle","arm_swing"]

SEGMENTS = {
    "legs":   ["hip_flexion","knee_flexion"],
    "hands":  ["shoulder_angle","elbow_angle","arm_swing"],
    "waist":  ["hip_flexion"],  # proxy
}

def load_seq_artifacts(data_dir="."):
    seq = np.load(os.path.join(data_dir, "preprocessed_sequences.npz"))
    X, y = seq["X"].astype(np.float32), seq["y"].astype(np.int64)
    idx = pd.read_csv(os.path.join(data_dir, "cycle_index.csv"))
    feat_names = pd.read_csv(os.path.join(data_dir, "sequence_features.csv"))["feature"].tolist()
    return X, y, idx, feat_names

def kind_suffix(kind):
    return {"angle":"z","vel":"vel","acc":"acc"}[kind]

def indices_for_kind_segment(feat_names, kind, segment_joints):
    suff = kind_suffix(kind)
    idxs = []
    for j in segment_joints:
        name = f"{j}_{suff}"
        if name in feat_names:
            idxs.append(feat_names.index(name))
    return idxs

def build_segment_index(feat_names):
    """
    Returns:
      seg_idx: dict(kind -> dict(segment -> list_of_channel_indices))
      all_kind_indices: dict(kind -> list_of_all_indices_for_that_kind)
      fast_idx: int index for is_fast channel
    """
    seg_idx = {"angle":{}, "vel":{}, "acc":{}}
    all_kind = {"angle":[], "vel":[], "acc":[]}
    for kind in ["angle","vel","acc"]:
        for seg, joints in SEGMENTS.items():
            idxs = indices_for_kind_segment(feat_names, kind, joints)
            seg_idx[kind][seg] = idxs
            all_kind[kind].extend(idxs)

    fast_idx = feat_names.index("is_fast")
    return seg_idx, all_kind, fast_idx

class BagDatasetMoME(Dataset):
    def __init__(self, data_dir, participants, feat_names, augment=False):
        X, y, idx, _ = load_seq_artifacts(data_dir)
        mask = idx["participant"].isin(participants).to_numpy()
        self.X = X[mask]
        self.y_cycle = y[mask]
        self.idx = idx.loc[mask].reset_index(drop=True)
        self.participants = participants
        self.feat_names = feat_names
        self.augment = augment  # <-- store flag

        # per-participant labels
        self.labels_part = self.idx[["participant"]].assign(y=self.y_cycle).groupby("participant")["y"].first().to_dict()
        # speed labels per cycle
        self.speed_targets = (self.X[:,0,feat_names.index("is_fast")] > 0.5).astype(np.float32)

    def __len__(self):
        return len(self.participants)

    def __getitem__(self, k):
        pid = self.participants[k]
        m = (self.idx["participant"] == pid).to_numpy()
        x_bag = self.X[m].copy()  # (Nc, T, F)

        # Apply augmentations only for training bags
        if self.augment:
            for i in range(x_bag.shape[0]):
                x_bag[i] = apply_augmentations(x_bag[i], self.feat_names)

        x_bag = torch.tensor(x_bag, dtype=torch.float32)
        y_stress = torch.tensor(self.labels_part[pid], dtype=torch.float32)
        spd = torch.tensor(self.speed_targets[m], dtype=torch.float32)  # (Nc,)
        return x_bag, y_stress, torch.tensor(k, dtype=torch.long), spd, pid

class TCNBlock(nn.Module):
    def __init__(self, in_ch, hidden=32, k=5, d=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=k, padding=d*(k-1)//2, dilation=d)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=k, padding=d*(k-1)//2, dilation=d)
        self.bn2   = nn.BatchNorm1d(hidden)
    def forward(self, x):  # (B, C, T)
        h = Fnn.relu(self.bn1(self.conv1(x)))
        h = Fnn.relu(self.bn2(self.conv2(h)))
        return h

class SegmentExpertStream(nn.Module):
    def __init__(self, seg_channels, hidden=32, out_dim=64):
        super().__init__()
        self.seg_channels = seg_channels
        self.segments = list(seg_channels.keys())
        self.blocks = nn.ModuleDict({
            seg: TCNBlock(in_ch=len(seg_channels[seg]), hidden=hidden, k=5, d=1)
            for seg in self.segments
        })
        self.proj = nn.Linear(hidden, out_dim)
        self.gate = nn.Linear(out_dim, len(self.segments))

    def forward(self, x):  # x: (B, T, F_total)
        B, T, C = x.shape  # renamed from ... , F = x.shape
        seg_embs = []
        for seg in self.segments:
            idxs = self.seg_channels[seg]
            if len(idxs) == 0:
                seg_embs.append(torch.zeros(B, self.proj.out_features, device=x.device))
                continue
            xs = x[:, :, idxs].transpose(1, 2)    # (B, Cseg, T)
            h = self.blocks[seg](xs)              # (B, hidden, T)
            z = h.mean(dim=2)                     # (B, hidden)
            z = Fnn.relu(self.proj(z))            # use Fnn
            seg_embs.append(z)
        H = torch.stack(seg_embs, dim=1)         # (B, S, D)
        w = torch.softmax(self.gate(H.mean(dim=1)), dim=1)
        z_kind = torch.sum(H * w.unsqueeze(-1), dim=1)  # (B, D)
        return z_kind, w

class MultiStreamEncoderMoME(nn.Module):
    """
    Builds three segment streams (angle/vel/acc), then concatenates their embeddings.
    """
    def __init__(self, seg_idx):
        super().__init__()
        self.angle_stream = SegmentExpertStream(seg_idx["angle"], hidden=32, out_dim=64)
        self.vel_stream   = SegmentExpertStream(seg_idx["vel"],   hidden=32, out_dim=64)
        self.acc_stream   = SegmentExpertStream(seg_idx["acc"],   hidden=32, out_dim=64)
        self.out_dim = 64 * 3

    def forward(self, x):   # (B, T, F)
        za, wa = self.angle_stream(x)
        zv, wv = self.vel_stream(x)
        zc, wc = self.acc_stream(x)
        z = torch.cat([za, zv, zc], dim=1)  # (B, D=192)
        return z

class MILAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, cycle_embs):  # (Nc, D)
        alpha = torch.softmax(self.fc(cycle_embs).squeeze(-1), dim=0)   # (Nc,)
        z = torch.sum(cycle_embs * alpha.unsqueeze(-1), dim=0)          # (D,)
        return z, alpha

class StressSeqMoME(nn.Module):
    def __init__(self, seg_idx):
        super().__init__()
        self.encoder = MultiStreamEncoderMoME(seg_idx)
        self.mil     = MILAttention(self.encoder.out_dim)
        self.head_stress = nn.Linear(self.encoder.out_dim, 1)    # bag-level stress
        self.head_speed  = nn.Linear(self.encoder.out_dim, 1)    # cycle-level speed
        self.head_id     = None

    def set_id_classes(self, n_ids):
        self.head_id = nn.Linear(self.encoder.out_dim, n_ids)
        # ensure head_id is on the same device as the rest of the model
        self.head_id.to(next(self.parameters()).device)

    def forward(self, x_bag):  # x_bag: (Nc, T, F)
        # encode each cycle
        H = self.encoder(x_bag)               # (Nc, D)
        # MIL pooling
        z_bag, _ = self.mil(H)                # (D,)
        # heads
        logit_stress = self.head_stress(z_bag).view(())       # scalar
        logits_id    = self.head_id(z_bag)                    # (n_ids,)
        logits_speed = self.head_speed(H).squeeze(-1)         # (Nc,) cycle-level
        return logit_stress, logits_speed, logits_id

