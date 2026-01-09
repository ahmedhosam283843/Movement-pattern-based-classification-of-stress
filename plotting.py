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


