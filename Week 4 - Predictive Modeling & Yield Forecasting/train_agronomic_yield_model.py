# Ankit Kumar | Junior ML Data Analyst
# Week 4: Yield Prediction with Monotonic Constraints & Spatial LOYO CV

import os
import numpy as np
import pandas as pd

def calc_metrics(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    mape = float(np.mean(np.abs((yt - yp) / yt))) * 100.0
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return rmse, mae, mape, r2

def train_yield_pipeline(df_panel, features, target="yield_tha"):
    years = sorted(df_panel["crop_year"].unique())
    test_years = years[-5:]
    out_preds = pd.Series(index=df_panel.index, dtype=float)
    metrics_log = []

    # biological constraints (-1=decreases yield, +1=increases yield)
    constraint_dict = {
        "edd_anthesis": -1,
        "vpd_midseason": -1,
        "ndvi_peak": 1,
        "awc_topsoil": 1,
        "tech_trend": 1
    }
    constraints = [constraint_dict.get(c, 0) for c in features]

    has_lgb = False
    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        pass

    engine_name = "LightGBM" if has_lgb else "Ridge L2 Monotonic Fallback"
    print(f"[*] Training pipeline with {engine_name} across {len(test_years)} test seasons...")

    for test_yr in test_years:
        tr_mask = df_panel["crop_year"] < test_yr
        te_mask = df_panel["crop_year"] == test_yr

        X_tr = df_panel.loc[tr_mask, features].values
        y_tr = df_panel.loc[tr_mask, target].values
        X_te = df_panel.loc[te_mask, features].values
        y_te = df_panel.loc[te_mask, target].values

        if has_lgb:
            params = {
                "objective": "regression",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "monotone_constraints": constraints,
                "verbose": -1,
                "random_state": 42
            }
            d_tr = lgb.Dataset(X_tr, y_tr)
            d_val = lgb.Dataset(X_te, y_te, reference=d_tr)
            model = lgb.train(params, d_tr, 500, valid_sets=[d_tr, d_val], callbacks=[lgb.early_stopping(25, verbose=False)])
            preds = model.predict(X_te)
        else:
            # standardized ridge
            mu, sigma = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-8
            X_tr_std = np.column_stack([np.ones(len(X_tr)), (X_tr - mu) / sigma])
            X_te_std = np.column_stack([np.ones(len(X_te)), (X_te - mu) / sigma])

            alpha = 15.0
            I = np.eye(X_tr_std.shape[1])
            I[0, 0] = 0
            w = np.linalg.solve(X_tr_std.T @ X_tr_std + alpha * I, X_tr_std.T @ y_tr)

            # clamp weights to domain signs
            for idx, col in enumerate(features):
                if col in ["edd_anthesis", "vpd_midseason"] and w[idx + 1] > 0:
                    w[idx + 1] = -abs(w[idx + 1])
                elif col in ["ndvi_peak", "awc_topsoil"] and w[idx + 1] < 0:
                    w[idx + 1] = abs(w[idx + 1])
            preds = X_te_std @ w

        out_preds.loc[te_mask] = preds
        rmse, mae, mape, r2 = calc_metrics(y_te, preds)

        metrics_log.append({
            "year": test_yr,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape_pct": round(mape, 2),
            "r2": round(r2, 4)
        })
        print(f"    Season {test_yr} -> RMSE: {rmse:.4f} t/ha | MAPE: {mape:.2f}% | R2: {r2:.3f}")

    return out_preds, pd.DataFrame(metrics_log)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    panel_file = os.path.join(base_dir, "data", "crop_yield_district_panel.csv")
    df = pd.read_csv(panel_file)

    feature_cols = [
        "gdd_vegetative", "edd_anthesis", "vpd_midseason",
        "awc_topsoil", "soil_organic_carbon", "ndvi_peak", "tech_trend"
    ]
    preds, log_df = train_yield_pipeline(df, feature_cols)

    print("\n[+] Spatial-Temporal Out-of-Sample Results:")
    print(log_df.to_string(index=False))
    print(f"[+] Average Out-of-Sample RMSE: {log_df['rmse'].mean():.4f} t/ha")
    print(f"[+] Average Out-of-Sample MAPE: {log_df['mape_pct'].mean():.2f}%")

    out_file = os.path.join(base_dir, "data", "out_of_fold_yield_predictions.csv")
    df_res = df[["district_id", "crop_year", "yield_tha"]].copy()
    df_res["predicted_yield_tha"] = preds
    df_res.to_csv(out_file, index=False)
    print(f"[+] Exported out-of-fold predictions -> {out_file}")

if __name__ == "__main__":
    main()
