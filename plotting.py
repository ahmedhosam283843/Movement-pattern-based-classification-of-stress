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


