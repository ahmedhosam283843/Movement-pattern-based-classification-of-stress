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

def timing_of_extrema(series):
    """
    Phase timing (0..1) of max and min angle within stride (Wang 2024-like).
    """
    T = len(series)
    idx_max = int(np.argmax(series))
    idx_min = int(np.argmin(series))
    return idx_max / (T - 1), idx_min / (T - 1)

def rom(series):
    """
    Range of motion (max - min) of a series.
    """
    return np.ptp(series)

def segment_aggregate(values_dict, joints):
    arr = np.array([values_dict[j] for j in joints if j in values_dict and np.isfinite(values_dict[j])])
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))

def compute_coord_corr(vel_dict, pair):
    a = vel_dict.get(pair[0], None)
    b = vel_dict.get(pair[1], None)
    if a is None or b is None:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0,1])

def wavelet_features(signal, wavelet='db4', level=5):
    """
    Wang 2024-style wavelet features:
      Decompose into D1..D5 and A5; compute abs max, mean, std, and absolute power for each.
    Returns dict {D1_absmax, D1_mean, D1_std, D1_power, ..., A5_power}
    """
    # detrend
    x = signal - np.mean(signal)
    try:
        coeffs = pywt.wavedec(x, wavelet, level=level, mode='symmetric')
    except Exception:
        return {}
    names = [f"D{i}" for i in range(1, level+1)][::-1] + [f"A{level}"]  # D1..D5, A5
    out = {}
    for name, c in zip(names, coeffs):
        if c is None or len(c) == 0:
            out[f"{name}_absmax"] = np.nan
            out[f"{name}_mean"] = np.nan
            out[f"{name}_std"] = np.nan
            out[f"{name}_power"] = np.nan
        else:
            out[f"{name}_absmax"] = float(np.max(np.abs(c)))
            out[f"{name}_mean"]   = float(np.mean(c))
            out[f"{name}_std"]    = float(np.std(c))
            out[f"{name}_power"]  = float(np.sum(c**2))
    return out

def compute_cycle_features(X, idx_df_with_st, feat_names):
    """
    Per-cycle feature computation aligned with the literature:
      - Freezing/stillness at multiple thresholds (Richer et al., 2024).
      - Time–frequency features on velocities/angles (Wang et al., 2024: dominant freq, 0–5 Hz bandpower,
        spectral centroid, spectral entropy; plus 0–1 Hz bandpower and ratios).
      - Coordination (corr) and phase-lag (Chen & Lee, 2024).
      - Segment aggregates (waist/hands/legs emphasis).
    """
    ch_idx, is_fast_idx, stride_time_idx = build_channel_index(feat_names, JOINTS)

    rows, keys = [], []
    for i in range(X.shape[0]):
        pid   = idx_df_with_st.loc[i, "participant"]
        cond  = idx_df_with_st.loc[i, "condition"]
        bout  = idx_df_with_st.loc[i, "bout"]
        speed = idx_df_with_st.loc[i, "speed"]
        cidx  = int(idx_df_with_st.loc[i, "cycle_idx"])
        stime = idx_df_with_st.loc[i, "stride_time"]
        dt    = safe_dt(stime)
        fs    = 1.0 / dt

        cycle = X[i]  # (T, F)
        vel_series = {j: cycle[:, ch_idx["vel"][j]]   for j in JOINTS}
        ang_series = {j: cycle[:, ch_idx["angle"][j]] for j in JOINTS}
        acc_series = {j: cycle[:, ch_idx["acc"][j]]   for j in JOINTS}

        feat = {}
        # per-joint features
        for j in JOINTS:
            v = vel_series[j]
            a = ang_series[j]
            acc = acc_series[j]

            # time-domain
            feat[f"{j}_motion_energy"] = float(np.sum(v**2) * dt)
            feat[f"{j}_vel_rms"] = float(np.sqrt(np.mean(v**2)))

            # freezing/stillness (multi-threshold; Richer 2024)
            multi_freeze = freezing_stats_multi(v, dt, eps_factors=(0.05, 0.10, 0.20))
            # keep the 0.10 series as canonical + include all thresholds
            feat[f"{j}_frac_still"] = multi_freeze["frac_still_e010"]
            feat[f"{j}_longest_still_sec"] = multi_freeze["longest_still_sec_e010"]
            feat[f"{j}_n_still_bouts"] = multi_freeze["n_still_bouts_e010"]
            for kf, val in multi_freeze.items():
                feat[f"{j}_{kf}"] = val

            feat[f"{j}_rom"]  = float(np.ptp(a))
            feat[f"{j}_mean"] = float(np.mean(a))
            feat[f"{j}_std"]  = float(np.std(a))

            # phase timing of extrema
            T = len(a) - 1 if len(a) > 1 else 1
            feat[f"{j}_tmax_phase"] = int(np.argmax(a)) / T
            feat[f"{j}_tmin_phase"] = int(np.argmin(a)) / T

            # velocity spectrum (Wang 2024)
            dom, bp05, cen, ent = welch_features(v, fs)
            feat[f"{j}_dom_freq"]      = dom
            feat[f"{j}_bandpower_0_5"] = bp05
            feat[f"{j}_spec_centroid"]  = cen
            feat[f"{j}_spec_entropy"]   = ent
            bp01_v = welch_bandpower(v, fs, 0.0, 1.0)
            feat[f"{j}_bandpower_0_1"] = bp01_v
            feat[f"{j}_bp_ratio_01_05"] = (bp01_v / (bp05 + 1e-12)) if np.isfinite(bp05) else np.nan

            # angle spectrum (same set)
            dom_a, bp05_a, cen_a, ent_a = welch_features(a, fs)
            feat[f"{j}_angle_dom_freq"]       = dom_a
            feat[f"{j}_angle_bandpower_0_5"]  = bp05_a
            bp01_a = welch_bandpower(a, fs, 0.0, 1.0)
            feat[f"{j}_angle_bandpower_0_1"]  = bp01_a
            feat[f"{j}_angle_bp_ratio_01_05"] = (bp01_a / (bp05_a + 1e-12)) if np.isfinite(bp05_a) else np.nan

            # acceleration vigor
            feat[f"{j}_acc_rms"] = float(np.sqrt(np.mean(acc**2)))

        # segment indices
        legs   = ["hip_flexion", "knee_flexion"]
        hands  = ["shoulder_angle", "elbow_angle", "arm_swing"]
        waist  = ["hip_flexion"]  # proxy

        # Build segment velocity signals by averaging joints within a segment
        def segment_vel_avg(joints):
            arr = [vel_series[j] for j in joints if j in vel_series]
            return np.mean(np.vstack(arr), axis=0) if arr else None

        seg_vel = {
            "legs":  segment_vel_avg(legs),
            "hands": segment_vel_avg(hands),
            "waist": segment_vel_avg(waist),
        }

        # Wavelet features per segment
        for seg_name, vseg in seg_vel.items():
            if vseg is None: 
                continue
            wv = wavelet_features(vseg, wavelet='db4', level=5)
            # Keep a compact set (absmax, std, power) from D2..D5 + A5
            for k, val in wv.items():
                if any(s in k for s in ["D2","D3","D4","D5","A5"]) and any(m in k for m in ["absmax","std","power"]):
                    feat[f"seg_{seg_name}_wave_{k}"] = val


        # coordination (corr + phase-lag)
        for a, b in COORD_PAIRS:
            corr = compute_coord_corr(vel_series, (a, b))
            lag, lag_phase = compute_phase_lag(vel_series[a], vel_series[b])
            feat[f"corr_vel_{a}_{b}"] = corr
            feat[f"phase_lag_samples_{a}_{b}"] = lag
            feat[f"phase_lag_phase_{a}_{b}"] = lag_phase

        # segment aggregates (waist/hands/legs)
        legs   = ["hip_flexion", "knee_flexion"]
        hands  = ["shoulder_angle", "elbow_angle", "arm_swing"]
        waist  = ["hip_flexion"]  # waist proxy

        for m in ["rom", "frac_still", "motion_energy", "vel_rms", "bandpower_0_5"]:
            vals = {j: feat.get(f"{j}_{m}", np.nan) for j in JOINTS}
            feat[f"seg_legs_{m}"]  = segment_aggregate(vals, legs)
            feat[f"seg_hands_{m}"] = segment_aggregate(vals, hands)
            feat[f"seg_waist_{m}"] = segment_aggregate(vals, waist)

        for m in ["bp_ratio_01_05", "angle_bp_ratio_01_05"]:
            vals = {j: feat.get(f"{j}_{m}", np.nan) for j in JOINTS}
            feat[f"seg_legs_{m}"]  = segment_aggregate(vals, legs)
            feat[f"seg_hands_{m}"] = segment_aggregate(vals, hands)
            feat[f"seg_waist_{m}"] = segment_aggregate(vals, waist)

        feat["is_fast"] = float(cycle[0, is_fast_idx])
        feat["stride_time"] = float(stime) if (stime is not None and np.isfinite(stime)) else np.nan

        rows.append(feat)
        keys.append((pid, cond, bout, speed, cidx))

    cy_df = pd.DataFrame(rows)
    cy_idx = pd.MultiIndex.from_tuples(keys, names=["participant","condition","bout","speed","cycle_idx"])
    cy_df.index = cy_idx
    return cy_df

def add_stride_time_aggregates(cycle_features: pd.DataFrame, stride_times: pd.DataFrame) -> pd.DataFrame:
    """
    Safely attach stride-time aggregates (mean, std, CV, bout asymmetry) and cross-speed deltas
    to cycle_features without MultiIndex join issues.
    """
    st = stride_times.copy()
    if isinstance(st.index, pd.MultiIndex):
        st = st.reset_index()
    if 'stride_idx' in st.columns and 'cycle_idx' not in st.columns:
        st = st.rename(columns={'stride_idx': 'cycle_idx'})

    req = {'participant','condition','bout','speed','cycle_idx','stride_time'}
    if not req.issubset(st.columns):
        raise ValueError(f"stride_times must contain {req}")

    # per participant × speed aggregates
    agg_ps = (
        st.groupby(['participant','speed'])['stride_time']
          .agg(st_mean='mean', st_std='std', st_min='min', st_max='max')
          .reset_index()
    )
    agg_ps['st_cv'] = agg_ps['st_std'] / (agg_ps['st_mean'] + 1e-12)

    # bout asymmetry (% diff) if two bouts exist
    pb = st.groupby(['participant','speed','bout'])['stride_time'].mean().reset_index()
    pv = pb.pivot(index=['participant','speed'], columns='bout', values='stride_time').reset_index()

    def _asym(row):
        vals = row.drop(labels=['participant','speed']).dropna().values
        if len(vals) >= 2:
            return (np.abs(vals[0] - vals[1]) / (np.mean(vals) + 1e-12)) * 100.0
        return np.nan

    pv['st_bout_asym_pct'] = pv.apply(_asym, axis=1)
    pv = pv[['participant','speed','st_bout_asym_pct']]

    # cross-speed deltas per participant
    base = st.groupby(['participant','speed'])['stride_time'].agg(mean='mean', std='std').reset_index()
    base['cv'] = base['std'] / (base['mean'] + 1e-12)
    b_piv = base.pivot(index='participant', columns='speed', values=['mean','cv'])
    b_piv.columns = ['_'.join(col).strip() for col in b_piv.columns.to_flat_index()]

    delta_df = b_piv.copy()
    delta_df['st_delta_mean_fast_minus_slow'] = (
        delta_df.get('mean_fast', np.nan) - delta_df.get('mean_slow', np.nan)
    )
    delta_df['st_delta_cv_fast_minus_slow'] = (
        delta_df.get('cv_fast', np.nan) - delta_df.get('cv_slow', np.nan)
    )
    delta_df = delta_df[['st_delta_mean_fast_minus_slow','st_delta_cv_fast_minus_slow']].reset_index()

    # merge into cycle_features
    cf = cycle_features.reset_index()
    cf = cf.merge(agg_ps, on=['participant','speed'], how='left')
    cf = cf.merge(pv, on=['participant','speed'], how='left')
    cf = cf.merge(delta_df, on='participant', how='left')
    cf = cf.set_index(['participant','condition','bout','speed','cycle_idx']).sort_index()
    return cf

def aggregate_participant_features(cycle_features: pd.DataFrame) -> pd.DataFrame:
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