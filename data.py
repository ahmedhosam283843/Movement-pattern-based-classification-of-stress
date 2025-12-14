"""
Data loading, merging, labels, integrity checks.
References:
- Wang et al. (2024, Gait & Posture): time+frequency features, waist/hands/legs emphasis.
- Shin et al. (2021, SAC): attention & movement patterns.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np

def load_kinematics(path_kin):
    df = pd.read_csv(path_kin)
    df = df.set_index(['participant', 'condition', 'bout', 'speed', 'cycle_idx', 'percentage_of_stride'])
    return df

def load_stride_times(path_st):
    df = pd.read_csv(path_st)
    df = df.set_index(['participant', 'condition', 'bout', 'speed', 'stride_idx'])
    return df

def merge_kinematics_stride(kinematics_df, stride_times_df):
    kin_reset = kinematics_df.reset_index()
    st_reset = stride_times_df.reset_index().rename(columns={'stride_idx': 'cycle_idx'})
    common_cols = ['participant', 'condition', 'bout', 'speed', 'cycle_idx']
    merged = pd.merge(kin_reset, st_reset[common_cols + ['stride_time']], on=common_cols, how='left')
    merged = merged.set_index(['participant', 'condition', 'bout', 'speed', 'cycle_idx', 'percentage_of_stride'])
    return merged

def integrity_checks(df: pd.DataFrame, angle_name_hints: Tuple[str,...] = ('angle','flexion','swing','hip','knee','elbow','shoulder')) -> Dict[str, int]:
    """
    df is expected to be indexed by:
    ['participant','condition','bout','speed','cycle_idx','percentage_of_stride']
    and contain 'stride_time' and joint angle columns.
    """
    report = {}

    # stride_time presence/validity
    st = df['stride_time'] if 'stride_time' in df.columns else pd.Series(dtype=float)
    report['missing_stride_time'] = int(st.isna().sum()) if 'stride_time' in df.columns else -1
    report['nonpositive_stride_time'] = int((st <= 0).sum()) if 'stride_time' in df.columns else -1

    # 101 samples per cycle
    grp = df.groupby(['participant','condition','bout','speed','cycle_idx'])
    sizes = grp.size()
    report['cycles_not_101_samples'] = int((sizes != 101).sum())

    # percent coverage check (min=0, max=100, unique=101)
    bad_percent = 0
    for _, g in grp:
        stride_pct = g.index.get_level_values('percentage_of_stride').to_numpy()
        if (len(stride_pct) != 101) or (stride_pct.min() != 0) or (stride_pct.max() != 100):
            bad_percent += 1
    report['cycles_bad_percent_coverage'] = bad_percent

    # numeric columns and angle plausibility (heuristic ±180 deg)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    report['nan_in_numeric'] = int(df[numeric_cols].isna().sum().sum())
    angle_cols = [c for c in numeric_cols if any(h in c.lower() for h in angle_name_hints)]
    report['angles_exceed_180deg'] = int(sum((df[c].abs() > 180).sum() for c in angle_cols))

    # participant-level single label consistency
    lab = df.reset_index()[['participant','cortisol']].drop_duplicates() if 'cortisol' in df.columns else pd.DataFrame()
    if not lab.empty:
        counts_per_part = lab.groupby('participant')['cortisol'].nunique()
        report['participants_multi_label'] = int((counts_per_part > 1).sum())
    else:
        report['participants_multi_label'] = -1

    # high-level counts
    report['n_participants'] = int(df.index.get_level_values('participant').nunique())
    report['n_cycles'] = int(len(grp))

    # Print brief EDA
    print("=== Integrity Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    # Class balance (participant-level)
    if 'cortisol' in df.columns:
        cond_per_part = df.reset_index().groupby('participant')['cortisol'].first()
        print("\nParticipants per cortisol label:")
        print(cond_per_part.value_counts())

    # Cycles per participant × speed (first 10)
    cps = df.reset_index().groupby(['participant','speed'])['cycle_idx'].nunique()
    print("\nCycles per participant × speed (first 10):")
    print(cps.head(10))

    return report

def add_cortisol_labels(merged_df, cortisol_map):
    """
    cortisol_map: dict like {"02":1, ...}
    """
    merged_df = merged_df.copy()
    merged_df['cortisol'] = merged_df.index.get_level_values('participant').str.replace('VP_', '').map(
        lambda x: cortisol_map.get(x, 0)
    )
    return merged_df

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
