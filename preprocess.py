# python
"""
LOPO-safe sequence builder + augmentation helpers.
References:
- Subject-wise normalization; Savitzky–Golay smoothing; vel/acc using stride_time.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from scipy.interpolate import interp1d


def _detect_angle_cols(df: pd.DataFrame) -> List[str]:
    # Auto-detect angle-like numeric columns (exclude stride_time & engineered cols)
    candidates = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and c not in ['stride_time','cortisol','is_fast']:
            cl = c.lower()
            if any(s in cl for s in ['angle','flexion','swing','hip','knee','elbow','shoulder']):
                candidates.append(c)
    # Fallback to your defined set if present
    if not candidates:
        fallback = ['hip_flexion','knee_flexion','elbow_angle','shoulder_angle','arm_swing']
        candidates = [c for c in fallback if c in df.columns]
    return sorted(set(candidates))

def _smooth(arr: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    if len(arr) < 7:
        return arr
    win = window if window % 2 == 1 else window + 1
    win = min(win, len(arr) if len(arr) % 2 == 1 else len(arr)-1)
    if win < 5:
        return arr
    return savgol_filter(arr, window_length=win, polyorder=poly)

def _vel_acc(series: np.ndarray, stride_time: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(series)
    dt = stride_time / (T - 1) if T > 1 else 0.01
    vel = np.gradient(series, dt)
    acc = np.gradient(vel, dt)
    return vel, acc

