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



