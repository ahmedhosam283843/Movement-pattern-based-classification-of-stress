# python
"""
Feature functions & constants; cycle/participant aggregation.
References:
- Wang et al. (2024): Welch & wavelet (D2–D5/A5 power), spectral centroid/entropy.
- Richer et al. (2024): freezing/stillness (fraction, longest still, bouts), head/upper extremities/trunk.
- Coordination & phase-lag (Chen & Lee, 2024).
"""

import numpy as np
import pandas as pd


# Constants (paste from exp1.py, keep names)
JOINTS = [
    "hip_flexion", "knee_flexion", "shoulder_angle", "elbow_angle", "arm_swing",
]

COORD_PAIRS = [
    ("hip_flexion", "knee_flexion"),
    ("shoulder_angle", "elbow_angle"),
    ("hip_flexion", "arm_swing"),
    ("knee_flexion", "arm_swing"),
]
WELCH_NPERSEG = 64
WELCH_NOOVERLAP = 32
BAND_MAX_HZ = 5.0



def build_channel_index(feat_names, joints):
    """
    Returns dicts mapping joint->indices for angle_z, vel, acc and index for is_fast.
    """
    ch_idx = {
        "angle": {},
        "vel": {},
        "acc": {}
    }
    for j in joints:
        ch_idx["angle"][j] = feat_names.index(f"{j}_z")
        ch_idx["vel"][j]   = feat_names.index(f"{j}_vel")
        ch_idx["acc"][j]   = feat_names.index(f"{j}_acc")
    is_fast_idx = feat_names.index("is_fast")
    # stride_time optional channel
    stride_time_idx = feat_names.index("stride_time") if "stride_time" in feat_names else None
    return ch_idx, is_fast_idx, stride_time_idx

def join_stride_times(idx_df: pd.DataFrame, stride_times: pd.DataFrame) -> pd.DataFrame:
    """
    Robust per-cycle stride_time join:
      - Accepts stride_times with either stride_idx or cycle_idx; normalizes to cycle_idx.
      - Works with flat DataFrame or MultiIndex stride_times (resets if needed).
      - Validates many-to-one merge on (participant, condition, bout, speed, cycle_idx).
    """
    st = stride_times.copy()
    # normalize index and column names
    if isinstance(st.index, pd.MultiIndex):
        st = st.reset_index()
    if 'stride_idx' in st.columns and 'cycle_idx' not in st.columns:
        st = st.rename(columns={'stride_idx': 'cycle_idx'})

    required = ['participant','condition','bout','speed','cycle_idx','stride_time']
    missing = set(required) - set(st.columns)
    if missing:
        raise ValueError(f"stride_times missing columns: {missing}")

    common = ['participant','condition','bout','speed','cycle_idx']
    merged = idx_df.merge(st[common + ['stride_time']], on=common, how='left', validate='many_to_one')
    return merged

def safe_dt(stride_time):
    T = 101
    if stride_time is None or not np.isfinite(stride_time) or stride_time <= 0:
        print("Warning: Invalid stride_time encountered, using fallback dt=0.01s")
        # fallback dt ~ 0.01 s
        return 0.01
    return stride_time / (T - 1)

def freezing_stats(vel, dt, eps_factor=0.1):
    """
    Richer et al. (2024)-style stillness metrics with adaptive threshold.
    vel: (T,) array (deg/s-equivalent after normalization); dt: seconds per sample
    eps = eps_factor * RMS(vel) (adaptive)
    Returns: frac_still, longest_still_sec, n_bouts
    """
    rms = np.sqrt(np.mean(vel**2)) + 1e-8
    eps = eps_factor * rms
    still = (np.abs(vel) < eps).astype(np.int32)

    # fraction still
    frac_still = still.mean()

    # longest still run
    longest = 0
    current = 0
    for v in still:
        if v == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    longest_still_sec = longest * dt

    # number of still bouts
    n_bouts = 0
    prev = 0
    for v in still:
        if v == 1 and prev == 0:
            n_bouts += 1
        prev = v
    return frac_still, longest_still_sec, n_bouts

