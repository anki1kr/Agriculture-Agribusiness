# Ankit Kumar | Junior ML Data Analyst
# Week 2: Satellite NDVI Cloud Masking & Temporal Smoothing

import os
import numpy as np
import pandas as pd

def savgol_filter_numpy(y, window=7):
    # window=7, order=2 convolution weights
    kernel = np.array([-2, 3, 6, 7, 6, 3, -2], dtype=float) / 21.0
    padded = np.pad(y, window // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')

def clean_field_ndvi(df_series, ndvi_col="ndvi", scl_col="scl"):
    df = df_series.copy()

    # mask cloud pixels (scl: 3=shadow, 8/9/10=clouds)
    df.loc[df[scl_col].isin([3, 8, 9, 10]), ndvi_col] = np.nan

    # linear fill
    df[ndvi_col] = df[ndvi_col].interpolate(method='linear', limit=3).bfill().ffill()

    # smooth curve
    try:
        from scipy.signal import savgol_filter
        df['ndvi_smoothed'] = savgol_filter(df[ndvi_col].values, 7, 2)
    except ImportError:
        df['ndvi_smoothed'] = savgol_filter_numpy(df[ndvi_col].values, 7)

    df['ndvi_smoothed'] = np.clip(df['ndvi_smoothed'], 0.0, 1.0)
    return df

def run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "satellite_ndvi_timeseries.csv")
    df_raw = pd.read_csv(raw_path)
    print(f"[*] Processing {len(df_raw)} NDVI observations across {df_raw['district_id'].nunique()} districts...")

    cleaned_dfs = []
    for _, group in df_raw.groupby("district_id"):
        cleaned_dfs.append(clean_field_ndvi(group))

    df_out = pd.concat(cleaned_dfs, ignore_index=True)
    out_path = os.path.join(base_dir, "data", "satellite_ndvi_cleaned.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[+] Reconstructed clean phenological curves -> {out_path}")
    return df_out

if __name__ == "__main__":
    run_pipeline()
