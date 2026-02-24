from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

import os, sys
# make sure repo root is on path
REPO = "/Users/eshateware/Desktop/dc-cherry-blossom-predictor-minimal"
os.chdir(REPO)
sys.path.insert(0, REPO)

from src import pipeline
# Import from your working pipeline
from src.pipeline import build_master_table, doy_to_date


def main(out_dir: str = "outputs") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build dataset + current feature row (same as pipeline)
    master, x_current, cutoff_date = build_master_table()
    master = master.sort_values("season_yr").reset_index(drop=True)

    X = master[["chill_mar1", "gdd_mar1", "winter_nino34"]]
    y = master["peak_doy"]

    # Use your tuned params (replace if you want different ones)
    params = dict(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
    )

    # -----------------------------
    # 1) Historical peak bloom DOY
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(master["season_yr"], master["peak_doy"])
    plt.xlabel("Year")
    plt.ylabel("Peak Bloom (DOY)")
    plt.title("DC Yoshino Peak Bloom (Historical)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "01_peak_bloom_history.png", dpi=220)
    plt.close()

    # -----------------------------
    # 2) Engineered features over time
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(master["season_yr"], master["chill_mar1"], label="Chill (as of Mar 1)")
    plt.plot(master["season_yr"], master["gdd_mar1"], label="GDD (as of Mar 1)")
    plt.xlabel("Year")
    plt.title("Engineered Features Over Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "02_features_over_time.png", dpi=220)
    plt.close()

    # -----------------------------
    # 3) Walk-forward MAE per fold + Actual vs Pred
    # -----------------------------
    tscv = TimeSeriesSplit(n_splits=min(5, len(master) - 1))

    fold_mae = []
    preds_all = []
    actuals_all = []
    years_all = []

    for tr, te in tscv.split(X):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])

        fold_mae.append(mean_absolute_error(y.iloc[te], pred))
        preds_all.extend(pred.tolist())
        actuals_all.extend(y.iloc[te].tolist())
        years_all.extend(master.loc[te, "season_yr"].tolist())

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(fold_mae) + 1), fold_mae)
    plt.xlabel("Fold (walk-forward)")
    plt.ylabel("MAE (days)")
    plt.title(f"Walk-forward MAE per Fold (mean={np.mean(fold_mae):.2f}, std={np.std(fold_mae):.2f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "03_walkforward_mae_per_fold.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(actuals_all, preds_all)
    mn = min(min(actuals_all), min(preds_all))
    mx = max(max(actuals_all), max(preds_all))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual Peak DOY")
    plt.ylabel("Predicted Peak DOY")
    plt.title("Walk-forward: Actual vs Predicted")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "04_actual_vs_predicted.png", dpi=220)
    plt.close()

    # -----------------------------
    # 4) Feature importance
    # -----------------------------
    final = xgb.XGBRegressor(**params)
    final.fit(X, y)

    plt.figure(figsize=(7, 4))
    plt.bar(X.columns.tolist(), final.feature_importances_)
    plt.ylabel("Importance (gain-based)")
    plt.title("XGBoost Feature Importance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "05_feature_importance.png", dpi=220)
    plt.close()

    # -----------------------------
    # 4b) SHAP explainability (saves 2 plots)
    # -----------------------------
    import shap

    explainer = shap.Explainer(final, X)
    shap_values = explainer(X)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(out / "07_shap_beeswarm.png", dpi=240, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(out / "08_shap_bar.png", dpi=240, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 5) Shareable forecast card
    # -----------------------------
    pred_doy = float(final.predict(x_current)[0])
    year = dt.date.today().year
    pred_date = doy_to_date(year, pred_doy)

    plt.figure(figsize=(10, 3.2))
    plt.axis("off")
    plt.text(0.02, 0.78, "DC Cherry Blossom Peak Bloom Forecast", fontsize=18, weight="bold")
    plt.text(0.02, 0.48, f"Prediction: {pred_date.strftime('%b %d, %Y')}  (DOY {int(round(pred_doy))})", fontsize=15)
    plt.text(
        0.02,
        0.22,
        f"Model: Tuned XGBoost | Walk-forward MAE ≈ {np.mean(fold_mae):.2f} days | Cutoff: {cutoff_date}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out / "06_forecast_card.png", dpi=240)
    plt.close()

    print(f"Saved plots to: {out.resolve()}")
    print("Files:")
    for p in sorted(out.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()