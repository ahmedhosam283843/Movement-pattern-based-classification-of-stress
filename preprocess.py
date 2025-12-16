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

def build_sequences_subjectwise(
    df: pd.DataFrame,
    angle_cols: List[str] = None,
    add_stride_time_channel: bool = False,
    n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, pd.MultiIndex, List[str]]:
    """
    Parallelized, LOPO-safe preprocessing:
      - MultiIndex-safe participant selection (no boolean mask misalignment).
      - Sort within each cycle by percentage_of_stride to ensure correct temporal order.
      - Subject-wise z-score, Savitzky–Golay smoothing.
      - Angular velocity/acceleration, is_fast channel (+ optional stride_time).
    """
    if angle_cols is None:
        angle_cols = _detect_angle_cols(df)
    print(f"Processing with angle columns: {angle_cols}")

    df = df.copy()
    # build participant label map once
    parts = df.reset_index()[['participant','cortisol']].drop_duplicates()
    label_map = dict(zip(parts['participant'], parts['cortisol']))

    # precompute participant-wise mean/std for angles (LOPO-safe normalizer)
    print("Pre-calculating participant-wise statistics...")
    participant_stats = df.groupby('participant')[angle_cols].agg(['mean', 'std'])

    grp = df.groupby(['participant','condition','bout','speed','cycle_idx'])
    num_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Building sequences for {len(grp)} cycles using {num_cores} cores...")

    def _process_group(
        group_data: Tuple,
        angle_cols: List[str],
        participant_stats: pd.DataFrame,
        label_map: dict,
        add_stride_time_channel: bool
    ) -> Optional[Tuple[np.ndarray, int, Tuple]]:
        key, g = group_data

        # sort by stride percentage to ensure chronological order
        if 'percentage_of_stride' in g.index.names:
            g = g.sort_index(level='percentage_of_stride')

        # skip incomplete/invalid cycles early
        if len(g) != 101:
            return None
        stime = float(g['stride_time'].iloc[0]) if 'stride_time' in g.columns else None
        if (stime is None) or (not np.isfinite(stime)) or (stime <= 0):
            return None
        if g[angle_cols].isna().any().any():
            return None

        pid = key[0]
        # MultiIndex-safe: use precomputed stats by participant
        stats = participant_stats.loc[pid]

        angles_arr, vel_arr, acc_arr = [], [], []
        for c in angle_cols:
            mu, sd = stats[(c, 'mean')], stats[(c, 'std')]
            series = g[c].values.astype(float)
            norm = (series - mu) / sd if (sd > 0 and np.isfinite(sd)) else series * 0.0
            norm_sm = _smooth(norm, window=7, poly=2)
            v, a = _vel_acc(norm_sm, stime)
            angles_arr.append(norm_sm)
            vel_arr.append(v)
            acc_arr.append(a)

        angles_arr = np.vstack(angles_arr).T
        vel_arr    = np.vstack(vel_arr).T
        acc_arr    = np.vstack(acc_arr).T

        is_fast = np.full((len(g), 1), int(key[3] == 'fast'), dtype=np.float32)
        channels = [angles_arr, vel_arr, acc_arr, is_fast]
        if add_stride_time_channel:
            st_chan = np.full((len(g), 1), stime, dtype=np.float32)
            channels.append(st_chan)

        X_cycle = np.concatenate(channels, axis=1).astype(np.float32)
        y_cycle = int(label_map.get(pid, 0))
        return X_cycle, y_cycle, key

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_group)(
            group_data,
            angle_cols=angle_cols,
            participant_stats=participant_stats,
            label_map=label_map,
            add_stride_time_channel=add_stride_time_channel
        )
        for group_data in grp
    )

    valid = [r for r in results if r is not None]
    if not valid:
        raise RuntimeError("No valid cycles found after preprocessing.")
    X_list, y_list, idx_list = zip(*valid)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    idx_cycles = pd.MultiIndex.from_tuples(idx_list, names=['participant','condition','bout','speed','cycle_idx'])

    feature_names = (
        [f"{c}_z" for c in angle_cols] +
        [f"{c}_vel" for c in angle_cols] +
        [f"{c}_acc" for c in angle_cols] +
        ['is_fast'] + (['stride_time'] if add_stride_time_channel else [])
    )
    return X, y, idx_cycles, feature_names

