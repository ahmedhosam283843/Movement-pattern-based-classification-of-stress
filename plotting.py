import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
import re
import seaborn as sns
import os
from sklearn.impute import SimpleImputer

# Use a clean, professional style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def _prettify_label(label):
    """Converts a code-like feature name into a human-readable one."""
    # Remove speed suffixes first
    label = label.replace('_slow', '')
    label = label.replace('_fast', '')
    
    # Body parts
    label = label.replace('l_', 'Left ')
    label = label.replace('r_', 'Right ')
    
    # Kinematics
    label = label.replace('_ang_vel', ' Vel.')
    label = label.replace('_ang_acc', ' Accel.')
    label = label.replace('_ang', ' Angle')
    
    # Joints
    label = label.replace('flexion', 'Flex.')
    label = label.replace('extension', 'Ext.')
    label = label.replace('shoulder', 'Shoulder')
    label = label.replace('hip', 'Hip')
    label = label.replace('knee', 'Knee')
    label = label.replace('ankle', 'Ankle')
    label = label.replace('elbow', 'Elbow')
    
    # Features
    label = label.replace('wav_pow_', 'Wavelet Pow. ')
    label = label.replace('freq_', 'Freq. ')
    label = label.replace('freeze', 'Freeze')
    label = label.replace('bouts', 'Bouts')
    label = label.replace('frac', 'Frac.')
    label = label.replace('longest', 'Longest')
    
    # Aggregates
    label = label.replace('_mean', ' (Mean)')
    label = label.replace('_std', ' (Std)')
    label = label.replace('_iqr', ' (IQR)')
    label = label.replace('_rom', ' (ROM)')
    
    # Final cleanup
    label = label.replace('_', ' ')
    # Capitalize D2, A5 etc.
    label = re.sub(r'\b(d[0-9]|a[0-9])\b', lambda m: m.group(1).upper(), label) 
    
    return label.strip().title()


def plot_roc_curve(y_true, y_pred_proba, save_path="roc_curve.pdf", title="ROC Curve"):
    """
    Plots and saves the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'Model (AUROC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
             label='Random Chance (AUROC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    # Use bbox_inches='tight' to prevent labels from being cut off
    plt.savefig(save_path,  dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()

def plot_pr_curve(y_true, y_pred_proba, save_path="pr_curve.pdf", title="Precision-Recall Curve"):
    """
    Plots and saves the Precision-Recall curve.
    """
    avg_precision = average_precision_score(y_true, y_pred_proba)
    random_baseline = np.mean(y_true)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Model (AUPRC = {avg_precision:.3f})')
    plt.plot([0, 1], [random_baseline, random_baseline], color='gray', lw=2,
             linestyle='--', label=f'Random Baseline (AUPRC = {random_baseline:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    # Use bbox_inches='tight'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to {save_path}")
    plt.close()
    
def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.pdf", title="Confusion Matrix", class_names=['Non-Responder', 'Responder']):
    """
    Plots and saves a normalized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages (normalized by true label)
    # Add a small epsilon to avoid division by zero if a class has 0 samples
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm.astype('float') / (cm_sum + 1e-9)
    
    # Create labels with raw counts and normalized percentages
    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=labels, fmt="", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, vmin=0, vmax=1)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_mean_feature_importance(
    importance_df,
    title="Mean Feature Importance (Top 20)",
    save_path="feature_importance.png"
):
    """
    Plots and saves the mean feature importances from a DataFrame.
    Assumes importance_df has columns ['feature', 'importance']
    and is already sorted descending (most important is at index 0).
    """
    if not isinstance(importance_df, pd.DataFrame) or importance_df.empty:
        print("Importance DataFrame is empty or invalid. Skipping plot.")
        return

    # Sort descending just in case it's not already
    df_sorted = importance_df.sort_values(by='importance', ascending=False)

    # Take top 20 and apply pretty labels
    df_top20 = df_sorted.head(20).copy()
    df_top20['feature_pretty'] = df_top20['feature'].apply(_prettify_label)

    if df_top20.empty:
        print("No features to plot.")
        return
        
    # --- CORRECTED PLOTTING ---
    # Make figure wider (10) and taller (8) to fit names
    plt.figure(figsize=(10, 8)) 
    plt.title(title)
    
    # `plt.barh` plots from bottom to top (index 0 is at the bottom).
    # To get the most important feature at the TOP, we must reverse the list.
    plt.barh(
        df_top20['feature_pretty'][::-1], # Reverse list
        df_top20['importance'][::-1],     # Reverse list
        color='C0'
    )
    
    plt.xlabel('Mean Feature Importance (Gini or Gain)')
    plt.ylabel('Feature')
    plt.grid(True, axis='x') # Gridlines on x-axis only
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout() # Adjust layout
    
    # Save with bbox_inches='tight' to ensure labels are not cut off
    plt.savefig(save_path,  dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {save_path}")
    plt.close()



def plot_correlation_bouts(part_feats_df, labels_ser, log_dir=".", top_k=40):
    """
    Calculates avg (Pearson, Spearman, Kendall) correlation of features
    with the cortisol label, split by slow and fast bouts, and writes the
    top_k results to CSV and a human-readable TXT file instead of making plots.

    Args:
        part_feats_df (pd.DataFrame): The long-form participant features
                                      (from participant_features.csv).
        labels_ser (pd.Series): A Series mapping participant ID to cortisol label.
        log_dir (str): Directory to save the output files.
        top_k (int): How many features to include per speed.
    """
    os.makedirs(log_dir, exist_ok=True)
    print("Calculating feature correlations with cortisol label...")

    # 1. Prepare DataFrame
    df = part_feats_df.merge(
        labels_ser.rename("cortisol").reset_index(),
        on="participant",
        how="left"
    )

    # 2. Split by speed
    slow_df = df[df['speed'] == 'slow'].drop(columns=['speed'], errors='ignore')
    fast_df = df[df['speed'] == 'fast'].drop(columns=['speed'], errors='ignore')

    def _calculate_corrs(speed_df, target_col='cortisol'):
        """Return a DataFrame indexed by feature with pearson/spearman/kendall/avg."""
        if target_col not in speed_df.columns:
            return pd.DataFrame(columns=['pearson', 'spearman', 'kendall', 'avg'])

        corr_df = speed_df.copy()
        numeric_cols = corr_df.select_dtypes(include=np.number).columns.drop([target_col], errors='ignore')

        if not numeric_cols.empty:
            medians = corr_df[numeric_cols].median()
            corr_df[numeric_cols] = corr_df[numeric_cols].fillna(medians)

        # If after imputation there are insufficient samples, correlations may be all NaN
        try:
            c_pear = corr_df.corr(method='pearson', numeric_only=True).get(target_col)
            c_spear = corr_df.corr(method='spearman', numeric_only=True).get(target_col)
            c_kend = corr_df.corr(method='kendall', numeric_only=True).get(target_col)
        except Exception:
            # Return empty DataFrame on unexpected failure
            return pd.DataFrame(columns=['pearson', 'spearman', 'kendall', 'avg'])

        # Drop the target row/column
        if c_pear is None:
            return pd.DataFrame(columns=['pearson', 'spearman', 'kendall', 'avg'])

        df_out = pd.DataFrame({
            'pearson': c_pear.drop(labels=[target_col], errors='ignore'),
            'spearman': c_spear.drop(labels=[target_col], errors='ignore'),
            'kendall': c_kend.drop(labels=[target_col], errors='ignore'),
        })
        df_out['avg'] = df_out[['pearson', 'spearman', 'kendall']].mean(axis=1, skipna=True)
        return df_out

    corr_slow_df = _calculate_corrs(slow_df)
    corr_fast_df = _calculate_corrs(fast_df)

    def _top_k_with_labels(corr_df, speed_name):
        if corr_df.empty:
            return pd.DataFrame(columns=['speed', 'feature', 'feature_pretty', 'pearson', 'spearman', 'kendall', 'avg'])
        corr_df = corr_df.dropna(how='all')  # drop rows where all correlations are NaN
        corr_df = corr_df.assign(feature=corr_df.index)
        corr_df['abs_avg'] = corr_df['avg'].abs()
        top = corr_df.sort_values(by='abs_avg', ascending=False).head(top_k).drop(columns=['abs_avg'])
        top['feature_pretty'] = top['feature'].apply(_prettify_label)
        top = top.reset_index(drop=True)
        top.insert(0, 'speed', speed_name)
        # Reorder columns
        cols = ['speed', 'feature', 'feature_pretty', 'pearson', 'spearman', 'kendall', 'avg']
        return top[cols]

    top_slow = _top_k_with_labels(corr_slow_df, 'slow')
    top_fast = _top_k_with_labels(corr_fast_df, 'fast')

    # Combine and write CSV
    result_df = pd.concat([top_slow, top_fast], ignore_index=True)
    csv_path = os.path.join(log_dir, "feature_correlations_by_speed.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Feature correlations (CSV) saved to {csv_path}")

    # Also write a human-readable text report
    txt_path = os.path.join(log_dir, "feature_correlations_by_speed.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for speed_name, group in result_df.groupby('speed'):
            f.write(f"=== Top {top_k} features for speed = {speed_name} ===\n\n")
            if group.empty:
                f.write("No valid correlation data.\n\n")
                continue
            for _, row in group.iterrows():
                f.write(f"{row['feature_pretty']} ({row['feature']}):\n")
                f.write(f"  Pearson:  {row['pearson']:.4f}\n" if pd.notna(row['pearson']) else "  Pearson:  NaN\n")
                f.write(f"  Spearman: {row['spearman']:.4f}\n" if pd.notna(row['spearman']) else "  Spearman: NaN\n")
                f.write(f"  Kendall:  {row['kendall']:.4f}\n" if pd.notna(row['kendall']) else "  Kendall: NaN\n")
                f.write(f"  Avg:      {row['avg']:.4f}\n\n" if pd.notna(row['avg']) else "  Avg:      NaN\n\n")
        f.write("End of report.\n")
    print(f"Feature correlations (text) saved to {txt_path}")
