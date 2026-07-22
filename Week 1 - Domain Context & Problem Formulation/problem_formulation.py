# Ankit Kumar | Junior ML Data Analyst
# Week 1: Agricultural Problem Scoping & Baseline Benchmarking

import os
import numpy as np
import pandas as pd

def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def calc_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def evaluate_baseline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "crop_yield_district_panel.csv")
    
    print(f"[*] Loading panel: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[+] Records: {len(df):,} | Districts: {df['district_id'].nunique()}")

    # 5-year olympic baseline
    test_years = [2019, 2020, 2021, 2022, 2023]
    baseline_metrics = []

    for yr in test_years:
        hist_df = df[(df["crop_year"] >= yr - 5) & (df["crop_year"] < yr)]
        test_df = df[df["crop_year"] == yr]
        
        # historical mean per district
        benchmarks = hist_df.groupby("district_id")["yield_tha"].mean()
        preds = test_df["district_id"].map(benchmarks)
        valid = ~preds.isna()

        y_true = test_df.loc[valid, "yield_tha"]
        y_pred = preds[valid]

        rmse = calc_rmse(y_true, y_pred)
        mae = calc_mae(y_true, y_pred)
        mape = float(np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true)))) * 100.0

        baseline_metrics.append({
            "year": yr,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape_pct": round(mape, 2)
        })

    res_df = pd.DataFrame(baseline_metrics)
    print("\n[+] Historical Moving Average Baseline:")
    print(res_df.to_string(index=False))

    # technology trend fit
    annual_mean = df.groupby("crop_year")["yield_tha"].mean()
    years_idx = annual_mean.index - 2009
    slope, intercept = np.polyfit(years_idx, annual_mean.values, 1)

    print("\n[+] De-trended Genetic Technology Signal:")
    print(f"    Growth rate : +{slope:.4f} t/ha/yr ({slope/annual_mean.iloc[0]*100:.2f}%)")
    print(f"    Intercept   : {intercept:.4f} t/ha")
    print(f"    Mean MAPE   : {res_df['mape_pct'].mean():.2f}% (demonstrates need for weather ML)")
    return res_df

if __name__ == "__main__":
    evaluate_baseline()
