"""
Orchestrator: preprocess -> features -> tabular -> sequence -> ensemble
This script calls the modular functions without changing their logic.
"""
from data import load_kinematics, load_stride_times, merge_kinematics_stride, add_cortisol_labels, integrity_checks, join_stride_times
from preprocess import build_sequences_subjectwise
from features import compute_cycle_features, add_stride_time_aggregates, aggregate_participant_features, build_channel_index, JOINTS
from models.tabular import (
    run_XGB_top20_slow, run_lopo_logistic_combined, run_lopo_rf_combined, run_lopo_rf_slow_tuned,
    run_rf_slow_lopo, run_simple_top20_slow, run_RF_top20_slow, run_xgb_combined, run_xgb_slow_lopo
)
from models.sequence_mome import run_lopo_sequence_mome # Uncomment to run sequence models

from plotting import (
    plot_roc_curve, plot_pr_curve, plot_mean_feature_importance, 
    plot_confusion_matrix, plot_correlation_bouts
)
from validation import bootstrap_metric 
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score, 
    f1_score, precision_score, recall_score, confusion_matrix
)


# ======================
# Utility helpers
# ======================
# python
def metrics_dict(name, y_true, y_pred, probs, parts):
    """
    Calculates a comprehensive set of metrics, including bootstrapped CIs,
    Sensitivity (Recall), and Specificity.
    """
    if len(np.unique(y_true)) < 2:
        print(f"Warning: Only one class present for model '{name}'. Skipping metrics.")
        return {"model": name}

    # --- Calculate standard metrics ---
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = np.nan

    metrics = {
        "model": name,
        "Sensitivity (Recall)": recall_score(y_true, y_pred, zero_division=0.0),
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred, zero_division=0.0),
        "MacroF1": f1_score(y_true, y_pred, average='macro', zero_division=0.0)
    }

    # --- Calculate Bootstrapped 95% CIs ---
    
    # AUROC
    auc, (auc_low, auc_high) = bootstrap_metric(
        parts, y_true, probs, roc_auc_score, n_boot=2000
    )
    metrics.update({
        "AUROC": auc,
        "AUROC_CI_low": auc_low,
        "AUROC_CI_high": auc_high
    })
    
    # AUPRC
    auprc, (auprc_low, auprc_high) = bootstrap_metric(
        parts, y_true, probs, average_precision_score, n_boot=2000
    )
    metrics.update({
        "AUPRC": auprc,
        "AUPRC_CI_low": auprc_low,
        "AUPRC_CI_high": auprc_high
    })

    # Balanced Accuracy
    bal_acc, (bal_acc_low, bal_acc_high) = bootstrap_metric(
        parts, y_true, y_pred, balanced_accuracy_score, n_boot=2000
    )
    metrics.update({
        "BalancedAcc": bal_acc,
        "BalAcc_CI_low": bal_acc_low,
        "BalAcc_CI_high": bal_acc_high
    })

    return metrics


def save_summary(log_dir, rows):
    """
    Saves summary metrics to CSV and JSON files with a timestamp.
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"summary_metrics_{ts}.csv")
    json_path = os.path.join(log_dir, f"summary_metrics_{ts}.json")
    
    df = pd.DataFrame(rows)
    
    # Reorder columns for clarity
    cols_order = [
        "model", "AUROC", "AUROC_CI_low", "AUROC_CI_high",
        "AUPRC", "AUPRC_CI_low", "AUPRC_CI_high",
        "BalancedAcc", "BalAcc_CI_low", "BalAcc_CI_high",
        "Sensitivity (Recall)", "Specificity", "Precision", "MacroF1"
    ]
    # Add any extra columns that might exist, just in case
    other_cols = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + other_cols]
    
    df.to_csv(csv_path, index=False, float_format="%.4f")
    
    # Save a rounded version for easy reading in JSON
    df_json = df.round(4)
    rows_json = df_json.to_dict('records')
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows_json, f, indent=2)
        
    print(f"\nSaved summary metrics:\n  {csv_path}\n  {json_path}")


def main():
    # 0) Paths
    raw_kin_path = "kinematics.csv"
    raw_st_path = "stride_times.csv"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    print("Working directory:", os.getcwd())

    # 1) Load & merge
    kin = load_kinematics(raw_kin_path)
    st = load_stride_times(raw_st_path)
    merged = merge_kinematics_stride(kin, st)

    # Labels (unchanged mapping)
    cortisol_map = {"02": 1, "05": 1, "06": 1, "15": 1, "16": 1,
                    "17": 1, "24": 1, "35": 1, "36": 1, "43": 1, "45": 1, "47": 1}
    merged = add_cortisol_labels(merged, cortisol_map)

    print("\nRunning integrity checks and EDA...")
    integrity_checks(merged)

    # 2) Preprocess sequences
    print("\nBuilding LOPO-safe sequences...")
    X, y, idx_cycles, feat_names = build_sequences_subjectwise(
        merged, angle_cols=None, add_stride_time_channel=False, n_jobs=-1
    )
    np.savez_compressed("preprocessed_sequences.npz", X=X, y=y)
    pd.DataFrame({'feature': feat_names}).to_csv(
        "sequence_features.csv", index=False)
    idx_cycles.to_frame(index=False).to_csv("cycle_index.csv", index=False)
    print("Saved: preprocessed_sequences.npz, sequence_features.csv, cycle_index.csv")

    # 3) Cycle/participant features
    print("\nJoining stride_time data...")
    idx_df = pd.read_csv("cycle_index.csv")
    st_raw = pd.read_csv(raw_st_path).rename(
        columns={'stride_idx': 'cycle_idx'})
    idx_df_st = join_stride_times(idx_df, st_raw)

    print("Computing per-cycle features...")
    cy_feats = compute_cycle_features(X, idx_df_st, feat_names)
    print("Adding stride-time aggregates...")
    cy_feats = add_stride_time_aggregates(cy_feats, st_raw)
    cy_feats.to_csv("cycle_features.csv")
    print(
        f"Saved per-cycle features: cycle_features.csv (shape={cy_feats.shape})")

    print("Computing participant-level features...")
    part_feats = aggregate_participant_features(cy_feats)
    part_feats.to_csv("participant_features.csv", index=False)
    print(
        f"Saved participant-level features: participant_features.csv (shape={part_feats.shape})")

    # Build wide participant×speed matrix
    labels = idx_df[['participant']].assign(
        y=y.astype(int)).groupby('participant')['y'].first()
    

    print("\nBuilding wide matrix for modeling...")
    wide = part_feats.pivot(index='participant', columns='speed')
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().merge(labels.rename(
        "label").reset_index(), on='participant', how='inner')
    Xw = wide.drop(columns=['participant', 'label']).values
    yw = wide['label'].values
    gw = wide['participant'].values
    feature_names = [c for c in wide.columns if c not in (
        'participant', 'label')]

    summary_rows = []

    # --- Start Model Runs ---
    
    # 4) Tabular: Logistic ( combined)
    print("\nLOPO Logistic (combined):")
    probs_log, y_true_log, y_pred_log, parts_log = run_lopo_logistic_combined(Xw, yw, gw, k_features=min(
        30, Xw.shape[1]), C=0.5, verbose=True, desc="LOPO Logistic")
    summary_rows.append(metrics_dict(
        "Logistic (combined)", y_true_log, y_pred_log, probs_log, parts_log))

    # 5) Tabular: RF (combined)
    print("\nLOPO RF (combined):")
    probs_rf, y_true_rf, y_pred_rf, parts_rf = run_lopo_rf_combined(
        Xw, yw, gw, k_features=40, n_estimators=1000, verbose=True)
    summary_rows.append(metrics_dict(
        "RF (combined)", y_true_rf, y_pred_rf, probs_rf, parts_rf))

    print("\nXGBoost (combined):")
    probs_xgb_c, y_true_xgb_c, y_pred_xgb_c, parts_xgb_c, _ = run_xgb_combined(
        X=Xw, y=yw, groups=gw, feature_names=feature_names, k_features=40, random_state=42, verbose=True
    )
    summary_rows.append(metrics_dict("XGB (combined, improved)", y_true_xgb_c, y_pred_xgb_c, probs_xgb_c, parts_xgb_c))

    # 6) Per-speed logistic (slow/fast)
    slow_idx = [i for i, c in enumerate(feature_names) if c.endswith("_slow")]
    fast_idx = [i for i, c in enumerate(feature_names) if c.endswith("_fast")]

    if slow_idx:
        X_slow = Xw[:, slow_idx]
        print(
            f"\nLOPO Slow (logistic, combined slice): {X_slow.shape[1]} features")
        probs_slow_log, y_slow_log, y_pred_slow_log, parts_slow_log = run_lopo_logistic_combined(
            X_slow, yw, gw, k_features=min(30, X_slow.shape[1]), C=0.5, verbose=True, desc="LOPO Slow"
        )
        summary_rows.append(metrics_dict(
            "Logistic (slow bout only)", y_slow_log, y_pred_slow_log, probs_slow_log, parts_slow_log))

    if fast_idx:
        X_fast = Xw[:, fast_idx]
        print(
            f"\nLOPO Fast (logistic, combined slice): {X_fast.shape[1]} features")
        probs_fast_log, y_fast_log, y_pred_fast_log, parts_fast_log = run_lopo_logistic_combined(
            X_fast, yw, gw, k_features=min(30, X_fast.shape[1]), C=0.5, verbose=True, desc="LOPO Fast"
        )
        summary_rows.append(metrics_dict(
            "Logistic (fast bout only)", y_fast_log, y_pred_fast_log, probs_fast_log, parts_fast_log))

    # 7) Slow-only: RF tuned
    print("\nSlow-only RF (tuned):")
    parts_slow_rf_tuned, probs_slow_rf_tuned, y_slow_rf_tuned, y_pred_slow_rf_tuned = run_lopo_rf_slow_tuned(
        wide, labels)
    summary_rows.append(metrics_dict("RF (slow bout only)",
                        y_slow_rf_tuned, y_pred_slow_rf_tuned, probs_slow_rf_tuned, parts_slow_rf_tuned))

    # 8) Slow-only: XGBoost tuned
    print("\nSlow-only XGBoost (tuned):")
    probs_xgb_s, y_true_xgb_s, y_pred_xgb_s, parts_xgb_s, xgb_imp_mean = run_xgb_slow_lopo(
        X_slow=Xw[:, slow_idx] if slow_idx else Xw, y=yw, groups=gw,
        feat_names=[feature_names[i]
                    for i in slow_idx] if slow_idx else feature_names,
        k_features=20, random_state=42, verbose=True
    )
    summary_rows.append(metrics_dict("XGB (slow bout only)",
        y_true_xgb_s, y_pred_xgb_s, probs_xgb_s, parts_xgb_s))

    # 9) Slow-only: RF alt
    print("\nSlow-only RF (alt):")
    probs_rf_alt, y_true_rf_alt, y_pred_rf_alt, parts_rf_alt = run_rf_slow_lopo(
        X_slow=Xw[:, slow_idx] if slow_idx else Xw, y=yw, groups=gw, k_features=40)
    summary_rows.append(metrics_dict("RF (slow bout only, alt)",
        y_true_rf_alt, y_pred_rf_alt, probs_rf_alt, parts_rf_alt))

    # 10) Top-20 slow features: Logistic and RF
    print("\nTop-20 slow features ─ Logistic:")
    probs_log_top20, y_true_log_top20, y_pred_log_top20, parts_log_top20 = run_simple_top20_slow(wide, labels)
    summary_rows.append(metrics_dict("Logistic (top-20 feats slow)",
        y_true_log_top20, y_pred_log_top20, probs_log_top20, parts_log_top20))

    print("\nTop-20 slow features ─ RF:")
    rf_top20_imp, probs_rf_top20, y_true_rf_top20, y_pred_rf_top20, parts_rf_top20 = run_RF_top20_slow(wide, labels)
    summary_rows.append(metrics_dict("RF (top-20 feats slow)",
        y_true_rf_top20, y_pred_rf_top20, probs_rf_top20, parts_rf_top20))
    
    # --- Plotting for the BEST model (RF Top-20) ---
    plot_roc_curve(
        y_true_rf_top20, probs_rf_top20,
        title="ROC Curve ─ RF (top-20 slow features)",
        save_path=os.path.join(log_dir, "roc_rf_top20_slow.png")
    )
    plot_pr_curve(
        y_true_rf_top20, probs_rf_top20,
        title="PR Curve ─ RF (top-20 slow features)",
        save_path=os.path.join(log_dir, "pr_rf_top20_slow.png")
    )
    plot_mean_feature_importance(
        rf_top20_imp,
        title="Mean Feature Importance ─ RF (top-20 slow features)",
        save_path=os.path.join(log_dir, "feature_importance_rf_top20_slow.png")
    )
    plot_confusion_matrix(
        y_true_rf_top20, y_pred_rf_top20,
        title="Confusion Matrix ─ RF (top-20 slow features)",
        save_path=os.path.join(log_dir, "confusion_matrix_rf_top20_slow.png")
    )


    print("\nTop-20 slow features ─ XGB:")
    probs_xgb_top20, y_true_xgb_top20, y_pred_xgb_top20, parts_xgb_top20, xgb_top20_imp = run_XGB_top20_slow(wide, labels, k=20)
    summary_rows.append(metrics_dict("XGB (top-20 feats slow)",
        y_true_xgb_top20, y_pred_xgb_top20, probs_xgb_top20, parts_xgb_top20))
    
    print("\nTop 20 features (XGB top-20 slow):")
    print(xgb_top20_imp.head(20).to_string(index=False))
    plot_mean_feature_importance(
        xgb_top20_imp,
        title="Mean Feature Importance ─ XGB (top-20 slow features)",
        save_path=os.path.join(log_dir, "feature_importance_xgb_top20_slow.png")
    )
    plot_confusion_matrix(
        y_true_xgb_top20, y_pred_xgb_top20,
        title="Confusion Matrix ─ XGB (top-20 slow features)",
        save_path=os.path.join(log_dir, "confusion_matrix_xgb_top20_slow.png")
    )
    

    # 11) Sequence MoME LOPO
    print("\nSequence MoME LOPO:")
    parts_seq, probs_seq, labels_seq, y_pred_seq = run_lopo_sequence_mome(
        data_dir=".", epochs=60, patience=8, gamma_focal=1.0, aux_w_id=0.1, aux_w_speed=0.2
    )
    if len(probs_seq):
        summary_rows.append(metrics_dict("Sequence MoME (LOPO)", labels_seq, y_pred_seq, probs_seq, parts_seq))
    else:
        print("No sequence outputs available.")

    # 12) Ensemble (RF slow tuned + sequence)
    try:
        df_rf  = pd.DataFrame({'participant': parts_slow_rf_tuned, 'p_rf_slow': probs_slow_rf_tuned})
        df_seq = pd.DataFrame({'participant': parts_seq,  'p_seq': probs_seq})
        df_lab = pd.DataFrame({'participant': parts_seq,  'label': labels_seq})

        if df_rf.empty or df_seq.empty or df_lab.empty:
            raise ValueError("One of the prediction dataframes is empty, cannot create ensemble.")

        ens = df_lab.merge(df_rf, on='participant').merge(df_seq, on='participant')

        if ens.empty:
             raise ValueError("Participant merge failed, no common participants for ensemble.")

        ens['p_ens'] = 0.5*ens['p_rf_slow'] + 0.5*ens['p_seq']

        ens_y_true = ens['label'].values
        ens_y_pred = (ens['p_ens'] >= 0.5).astype(int)
        ens_probs = ens['p_ens'].values
        ens_parts = ens['participant'].values # Get parts for ensemble

        summary_rows.append(metrics_dict("Ensemble (RF slow tuned + sequence)", ens_y_true, ens_y_pred, ens_probs, ens_parts))

        print("\nEnsemble (RF slow tuned + sequence):",
              "AUROC:", f"{roc_auc_score(ens_y_true, ens_probs):.3f}",
              "AUPRC:", f"{average_precision_score(ens_y_true, ens_probs):.3f}",
              "BalancedAcc:", f"{balanced_accuracy_score(ens_y_true, ens_y_pred):.3f}",
              "MacroF1:", f"{f1_score(ens_y_true, ens_y_pred, average='macro'):.3f}")
    except Exception as e:
        print(f"Ensemble step skipped: {e}")

    # 13) Save summary
    save_summary(log_dir, summary_rows)
    print("\nAll models run sequentially. Done.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()