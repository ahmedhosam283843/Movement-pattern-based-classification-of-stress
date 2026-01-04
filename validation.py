"""
Validation helpers: threshold tuning, bootstrap CIs, participant-level effect tests.
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score
)
from scipy.stats import mannwhitneyu

def tune_threshold_balacc(y_val, probs_val):
    """
    Tune threshold to maximize balanced accuracy on validation set.
    """
    if len(np.unique(y_val)) < 2:
        return 0.5
    thr_grid = np.linspace(0.1, 0.9, 33)
    best_thr, best_balacc = 0.5, -1.0
    for thr in thr_grid:
        y_hat = (probs_val >= thr).astype(int)
        bal = balanced_accuracy_score(y_val, y_hat)
        if bal > best_balacc:
            best_balacc, best_thr = bal, thr
    return best_thr

def bootstrap_metric(parts, y_true, y_scores, metric_func, n_boot=2000, seed=42, **metric_kwargs):
    """
    Generic participant-level bootstrapping for any sklearn metric.

    Args:
        parts (array): Array of participant IDs, shape (n_samples,).
        y_true (array): True labels, shape (n_samples,).
        y_scores (array): Model scores (probs or binary preds), shape (n_samples,).
        metric_func (callable): The sklearn.metrics function (e.g., roc_auc_score).
        n_boot (int): Number of bootstrap iterations.
        seed (int): Random seed.
        **metric_kwargs: Additional kwargs for the metric_func (e.g., average='macro').

    Returns:
        tuple: (metric_point_estimate, (ci_low, ci_high))
    """
    rng = np.random.default_rng(seed)
    uniq_parts = np.unique(parts)
    n_parts = len(uniq_parts)
    
    # Check if we can even calculate the metric
    if len(np.unique(y_true)) < 2:
        return np.nan, (np.nan, np.nan)

    # Calculate the point estimate on the original data
    try:
        point_estimate = metric_func(y_true, y_scores, **metric_kwargs)
    except ValueError:
        point_estimate = np.nan

    metric_boots = []
    for _ in range(n_boot):
        # Participant-level sampling with replacement
        boot_parts = rng.choice(uniq_parts, size=n_parts, replace=True)
        
        # Create mask for this bootstrap sample
        mask = np.isin(parts, boot_parts)
        
        # Check for invalid bootstrap samples (e.g., all one class)
        if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
            continue
            
        try:
            boot_metric = metric_func(y_true[mask], y_scores[mask], **metric_kwargs)
            if np.isfinite(boot_metric):
                metric_boots.append(boot_metric)
        except ValueError:
            # Silently ignore errors from bad bootstraps (e.g., all one class)
            continue
    
    if not metric_boots:
        return point_estimate, (np.nan, np.nan)

    # Calculate 95% CI
    ci_low, ci_high = np.percentile(metric_boots, [2.5, 97.5])
    return point_estimate, (ci_low, ci_high)


