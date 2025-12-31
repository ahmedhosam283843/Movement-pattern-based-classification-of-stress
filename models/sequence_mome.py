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

def focal_bce(logit, target, alpha_pos=1.0, alpha_neg=1.0, gamma=1.0):
    """
    Single-sample focal BCE. alpha_pos/alpha_neg weight classes (imbalance).
    """
    p = torch.sigmoid(logit)
    # select alpha
    alpha = alpha_pos if target.item() > 0.5 else alpha_neg
    bce = Fnn.binary_cross_entropy_with_logits(logit, target, reduction='none')
    pt  = p if target.item() > 0.5 else (1.0 - p)
    mod = (1.0 - pt) ** gamma
    return alpha * mod * bce

def temperature_scaling(logits, targets):
    T = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def closure():
        opt.zero_grad()
        p = torch.sigmoid(logits / T)
        loss = Fnn.binary_cross_entropy(p, targets)
        loss.backward()
        return loss
    try:
        opt.step(closure)
        return float(T.detach().cpu().numpy())
    except Exception:
        return 1.0

def tune_threshold_balacc(y_val, p_val):
    if len(np.unique(y_val)) < 2:
        return 0.5
    grid = np.linspace(0.1, 0.9, 33)
    best_thr, best = 0.5, -1
    for t in grid:
        yhat = (p_val >= t).astype(int)
        bal = balanced_accuracy_score(y_val, yhat)
        if bal > best:
            best, best_thr = bal, t
    return best_thr

def run_lopo_sequence_mome(data_dir=".", epochs=50, patience=8, gamma_focal=1.0, aux_w_id=0.1, aux_w_speed=0.2, device=None):
    X, y, idx, feat_names = load_seq_artifacts(data_dir)
    participants = sorted(idx["participant"].unique().tolist())
    seg_idx, all_kind, fast_idx = build_segment_index(feat_names)

    # FIX: Initialize a list to store binary predictions
    parts_out, probs_out, labels_out, preds_out = [], [], [], []
    logo = LeaveOneGroupOut()

    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cpu':
        print("WARNING: CUDA not available. Training on CPU will be slow.")

    X_dummy = np.arange(len(participants))
    participants_np = np.array(participants)

    for fold_id, (tr_idx, te_idx) in enumerate(logo.split(X_dummy, groups=participants_np), 1):
        tr_p = participants_np[tr_idx].tolist()
        te_p = [participants_np[te_idx[0]]]

        rng = np.random.default_rng(42 + fold_id)
        val_count = max(1, int(0.2 * len(tr_p)))
        val_p = rng.choice(tr_p, size=val_count, replace=False).tolist()
        train_p = [p for p in tr_p if p not in val_p]

        # CRITICAL FIX: Enable data augmentation for the training set
        train_ds = BagDatasetMoME(data_dir, train_p, feat_names, augment=True)
        val_ds   = BagDatasetMoME(data_dir, val_p,   feat_names, augment=False)
        te_ds    = BagDatasetMoME(data_dir, te_p,    feat_names, augment=False)

        model = StressSeqMoME(seg_idx).to(device)
        model.set_id_classes(len(train_p))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        pos = sum(int(train_ds.labels_part[p]==1) for p in train_p)
        neg = sum(int(train_ds.labels_part[p]==0) for p in train_p)
        denom = max(1, (pos + neg))
        alpha_pos = neg / denom
        alpha_neg = pos / denom

        best_val, best_state, no_improve = -np.inf, None, 0
        pbar = tqdm(range(1, epochs+1), desc=f"Fold {fold_id}/{len(participants)} ({te_p[0]})", leave=False)
        for ep in pbar:
            model.train()
            loss_acc, nb = 0.0, 0
            for i in range(len(train_ds)):
                x_bag, y_stress, id_t, speed_t, pid = train_ds[i]
                if x_bag.shape[0] == 0: continue
                x_bag, y_stress, speed_t, id_t = x_bag.to(device), y_stress.to(device), speed_t.to(device), id_t.to(device)
                opt.zero_grad()
                logit_stress, logits_speed, logits_id = model(x_bag)
                loss_stress = focal_bce(logit_stress, y_stress, alpha_pos=alpha_pos, alpha_neg=alpha_neg, gamma=gamma_focal)
                loss_id     = Fnn.cross_entropy(logits_id.unsqueeze(0), id_t.unsqueeze(0))
                loss_speed  = Fnn.binary_cross_entropy_with_logits(logits_speed, speed_t)
                loss = loss_stress + aux_w_id*loss_id + aux_w_speed*loss_speed
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_acc += float(loss.item()); nb += 1

            model.eval()
            val_probs, val_labels = [], []
            with torch.no_grad():
                for i in range(len(val_ds)):
                    x_bag, y_stress, _, _, _ = val_ds[i]
                    if x_bag.shape[0]==0: continue
                    logit_stress, _, _ = model(x_bag.to(device))
                    val_probs.append(torch.sigmoid(logit_stress).cpu().item())
                    val_labels.append(int(y_stress.item()))
            auprc = average_precision_score(val_labels, val_probs) if len(set(val_labels))>=2 else 0.5
            pbar.set_postfix_str(f"loss={loss_acc/max(1,nb):.3f}, val_AUPRC={auprc:.3f}")

            if auprc > best_val:
                best_val, best_state, no_improve = auprc, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
            else:
                no_improve += 1
                if no_improve >= patience: break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        model.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for i in range(len(val_ds)):
                x_bag, y_stress, _, _, _ = val_ds[i]
                if x_bag.shape[0]==0: continue
                logit_stress, _, _ = model(x_bag.to(device))
                val_logits.append(logit_stress.cpu().item())
                val_targets.append(float(y_stress.item()))
        
        T = 1.0
        if len(val_logits) > 1 and len(np.unique(val_targets)) > 1:
            T = temperature_scaling(torch.tensor(val_logits), torch.tensor(val_targets))
            thr = tune_threshold_balacc(np.array(val_targets, dtype=int), 1.0/(1.0+np.exp(-np.array(val_logits)/T)))
        else:
            thr = 0.5 # Fallback threshold

        # Test on held-out participant
        with torch.no_grad():
            x_bag, y_stress, _, _, pid = te_ds[0]
            if x_bag.shape[0] > 0:
                logit_stress, _, _ = model(x_bag.to(device))
                p = torch.sigmoid(logit_stress / T).cpu().item()
                
                pred = int(p >= thr)

                parts_out.append(pid)
                probs_out.append(p)
                labels_out.append(int(y_stress.item()))
                preds_out.append(pred) # Append the prediction
                
    return np.array(parts_out), np.array(probs_out), np.array(labels_out), np.array(preds_out)

