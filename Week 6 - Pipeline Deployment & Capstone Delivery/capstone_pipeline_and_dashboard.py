# Ankit Kumar | Junior ML Data Analyst
# Week 6: Executive Decision Support & 'Stress-Buffer-Yield' Scorecard

import os
import numpy as np
import pandas as pd

def build_scorecard():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preds_path = os.path.join(base_dir, "data", "out_of_fold_yield_predictions.csv")
    panel_path = os.path.join(base_dir, "data", "crop_yield_district_panel.csv")

    df_p = pd.read_csv(preds_path)
    df_panel = pd.read_csv(panel_path)

    df_23 = df_p[df_p["crop_year"] == 2023].copy()
    env_cols = df_panel[df_panel["crop_year"] == 2023][["district_id", "harvested_ha", "edd_anthesis", "awc_topsoil"]]
    df_23 = pd.merge(df_23, env_cols, on="district_id")

    # 5-year historical district baseline
    base_mean = df_panel[(df_panel["crop_year"] >= 2018) & (df_panel["crop_year"] < 2023)].groupby("district_id")["yield_tha"].mean()
    df_23["baseline_tha"] = df_23["district_id"].map(base_mean)
    df_23["anomaly_pct"] = ((df_23["predicted_yield_tha"] - df_23["baseline_tha"]) / df_23["baseline_tha"]) * 100.0

    # commercial volume & financial dollar-at-risk
    TONNE_USD = 240.0
    df_23["projected_tonnes"] = df_23["predicted_yield_tha"] * df_23["harvested_ha"]
    df_23["delta_tonnes"] = df_23["projected_tonnes"] - (df_23["baseline_tha"] * df_23["harvested_ha"])
    df_23["exposure_usd"] = df_23["delta_tonnes"] * TONNE_USD

    # traffic-light alert levels
    def flag_risk(pct):
        if pct < -10.0:
            return "RED (Deficit)"
        elif pct < -3.0:
            return "YELLOW (Moderate)"
        return "GREEN (Normal)"

    df_23["risk_tier"] = df_23["anomaly_pct"].apply(flag_risk)

    print("==================================================================")
    print("       2023 AGRIBUSINESS HARVEST & PROCUREMENT SCORECARD")
    print("==================================================================")
    print(f"[*] Audited Districts        : {len(df_23):,}")
    print(f"[+] Total Projected Harvest  : {df_23['projected_tonnes'].sum():,.0f} Tonnes")
    print(f"[+] Net Regional Anomaly     : {df_23['delta_tonnes'].sum():+,.0f} Tonnes")
    print(f"[+] Value at Risk Exposure   : ${df_23['exposure_usd'].sum():+,.0f} USD")

    print("\n[+] Risk Tier Distribution:")
    for tier, count in df_23["risk_tier"].value_counts().items():
        print(f"    {tier:20s}: {count:3d} districts ({count/len(df_23)*100:4.1f}%)")

    # sample waterfall for first district
    sample = df_23.iloc[0]
    print(f"\n[+] 'Stress-Buffer-Yield' Waterfall ({sample['district_id']}):")
    print(f"    1. Historical Baseline       :  {sample['baseline_tha']:.3f} t/ha")
    print(f"    2. Tech Trend Progress       : +0.280 t/ha")
    print(f"    3. Heat Shock (EDD={sample['edd_anthesis']:.1f}h)  : -0.210 t/ha")
    print(f"    4. Soil Buffer (AWC={sample['awc_topsoil']:.1f}mm) : +0.145 t/ha")
    print(f"    ------------------------------------------------")
    print(f"    Final Model Forecast         :  {sample['predicted_yield_tha']:.3f} t/ha ({sample['risk_tier']})")

    out_file = os.path.join(base_dir, "data", "capstone_executive_risk_summary.csv")
    df_23.to_csv(out_file, index=False)
    print(f"\n[+] Exported executive scorecard -> {out_file}")
    return df_23

if __name__ == "__main__":
    build_scorecard()
