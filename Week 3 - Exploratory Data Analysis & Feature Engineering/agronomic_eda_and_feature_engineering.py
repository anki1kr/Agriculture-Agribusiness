# Ankit Kumar | Junior ML Data Analyst
# Week 3: Meteorological Feature Engineering & Agronomic Indices

import os
import numpy as np
import pandas as pd

def compute_agronomic_features(df_weather, t_base=10.0, t_cutoff=30.0, heat_thresh=32.0):
    df = df_weather.copy()

    # growing degree days (gdd)
    t_mean = (df['t_max'] + df['t_min']) / 2.0
    t_clipped = np.clip(t_mean, t_base, t_cutoff)
    df['daily_gdd'] = np.maximum(0.0, t_clipped - t_base)

    # extreme heat days & degree hours (edd > 30C)
    df['is_heat_shock'] = (df['t_max'] >= heat_thresh).astype(int)
    df['daily_edd'] = np.maximum(0.0, df['t_max'] - 30.0)

    # vapor pressure deficit (tetens formulation)
    svp = 0.61078 * np.exp((17.27 * df['t_max']) / (df['t_max'] + 237.3))
    avp = svp * (df['rh_min'] / 100.0)
    df['vpd_kpa'] = svp - avp

    return {
        'total_gdd': round(float(df['daily_gdd'].sum()), 2),
        'heat_shock_days': int(df['is_heat_shock'].sum()),
        'total_edd': round(float(df['daily_edd'].sum()), 2),
        'mean_vpd': round(float(df['vpd_kpa'].mean()), 3),
        'precip_sum_mm': round(float(df['precip_mm'].sum()), 1),
        'dry_day_ratio': round(float((df['precip_mm'] == 0).mean()), 3)
    }

def run_eda():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weather_path = os.path.join(base_dir, "data", "daily_weather_era5_sample.csv")
    df_raw = pd.read_csv(weather_path)
    print(f"[*] Parsing weather data: {len(df_raw)} records across {df_raw['district_id'].nunique()} districts")

    records = []
    for d_id, group in df_raw.groupby("district_id"):
        feats = compute_agronomic_features(group)
        feats["district_id"] = d_id
        records.append(feats)

    df_feats = pd.DataFrame(records)
    print("\n[+] Engineered Features (Top 5 Districts):")
    print(df_feats.head().to_string(index=False))

    # correlation matrix
    num_cols = [c for c in df_feats.columns if c != "district_id"]
    print("\n[+] Cross-Feature Correlations:")
    print(df_feats[num_cols].corr().round(3).to_string())

    out_path = os.path.join(base_dir, "data", "district_weather_features.csv")
    df_feats.to_csv(out_path, index=False)
    print(f"\n[+] Exported features -> {out_path}")
    return df_feats

if __name__ == "__main__":
    run_eda()
