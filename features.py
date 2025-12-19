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
from scipy.signal import welch
import pywt

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

def welch_bandpower(x, fs, fmin, fmax, nperseg=64, noverlap=32):
    x = x - np.mean(x)
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)), noverlap=min(noverlap, len(x)//2))
    if len(f) == 0 or np.all(Pxx <= 0):
        return np.nan
    band = (f >= fmin) & (f <= fmax)
    if not band.any():
        return 0.0
    return float(np.trapz(Pxx[band], f[band]))

def freezing_stats_multi(vel, dt, eps_factors=(0.05, 0.10, 0.20)):
    """
    Compute stillness features at multiple thresholds: eps = eps_factor * RMS(vel).
    Returns dict with keys: frac_still_eXXX, longest_still_sec_eXXX, n_still_bouts_eXXX.
    """
    out = {}
    rms = np.sqrt(np.mean(vel**2)) + 1e-8
    for ef in eps_factors:
        eps = ef * rms
        still = (np.abs(vel) < eps).astype(np.int32)
        # fraction still
        frac_still = float(still.mean())
        # longest still
        longest = 0; cur = 0
        for v in still:
            if v:
                cur += 1; longest = max(longest, cur)
            else:
                cur = 0
        longest_sec = longest * dt
        # number of still bouts
        n_bouts = 0; prev = 0
        for v in still:
            if v and not prev:
                n_bouts += 1
            prev = v
        suffix = f"e{int(ef*100):03d}"
        out[f"frac_still_{suffix}"] = frac_still
        out[f"longest_still_sec_{suffix}"] = longest_sec
        out[f"n_still_bouts_{suffix}"] = n_bouts
    return out

def compute_phase_lag(sig_a, sig_b):
    """
    Phase/lag between two signals via normalized cross-correlation.
    Returns lag_samples (signed), lag_phase (0..1, absolute fraction of the stride).
    """
    a = (sig_a - np.mean(sig_a)) / (np.std(sig_a) + 1e-8)
    b = (sig_b - np.mean(sig_b)) / (np.std(sig_b) + 1e-8)
    cc = np.correlate(a, b, mode='full')
    lag = int(np.argmax(cc) - (len(a) - 1))
    lag_phase = abs(lag) / max(1, (len(a) - 1))
    return lag, lag_phase

def welch_features(signal, fs):
    """
    Time-frequency features (Wang et al., 2024): dominant frequency excl. DC, bandpower 0–5 Hz,
    spectral centroid, spectral entropy.
    """
    # detrend implicitly by mean removal
    x = signal - np.mean(signal)
    f, Pxx = welch(x, fs=fs, nperseg=min(WELCH_NPERSEG, len(x)), noverlap=min(WELCH_NOOVERLAP, len(x)//2))
    # avoid nan
    if Pxx.sum() <= 0 or len(f) == 0:
        return np.nan, np.nan, np.nan, np.nan
    # dominant freq excluding DC
    mask_non_dc = f > 1e-6
    if not mask_non_dc.any():
        dom_freq = 0.0
    else:
        dom_freq = f[mask_non_dc][np.argmax(Pxx[mask_non_dc])]
    # bandpower 0–5 Hz
    band = (f > 0) & (f <= BAND_MAX_HZ)
    bandpower = np.trapz(Pxx[band], f[band]) if band.any() else 0.0
    # spectral centroid
    centroid = np.sum(f * Pxx) / np.sum(Pxx)
    # spectral entropy
    p = Pxx / (Pxx.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return dom_freq, bandpower, centroid, entropy

    """
    Participant × speed aggregates: mean, std, IQR + selected deltas (fast−slow) for segment features.
    Produces consistently flattened column names: <feature>_<agg> (e.g., seg_waist_frac_still_mean).
    """
    def iqr(x):
        q75, q25 = np.nanpercentile(x, 75), np.nanpercentile(x, 25)
        return q75 - q25

    cols_to_agg = [c for c in cycle_features.columns if c not in ["is_fast"]]
    grouped = cycle_features.groupby(["participant","speed"])
    agg_df = grouped[cols_to_agg].agg(["mean","std",iqr])

    # flatten MultiIndex columns
    agg_df.columns = [f"{c}_{agg}" for c, agg in agg_df.columns]
    agg_df["n_cycles"] = grouped.size()

    # slow-fast deltas for key segment features
    speed_pv = agg_df.reset_index().pivot(index="participant", columns="speed")
    speed_pv.columns = [f"{a}_{b}" for a,b in speed_pv.columns]
    seg_feats = ["seg_waist_frac_still_mean","seg_hands_bandpower_0_5_mean","seg_legs_rom_mean"]

    delta_part = pd.DataFrame(index=speed_pv.index)
    for f in seg_feats:
        delta_part[f"delta_{f}_fast_minus_slow"] = speed_pv.get(f+"_fast", np.nan) - speed_pv.get(f+"_slow", np.nan)

    part_df = agg_df.reset_index().merge(delta_part.reset_index(), on="participant", how="left")
    return part_df