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

