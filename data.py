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

