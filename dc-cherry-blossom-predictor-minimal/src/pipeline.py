# DC Cherry Blossom Predictor (END-to-END, works with CSV where bloom date = DOY number)
# Paste into ONE Jupyter/VS Code notebook cell and Run.

from __future__ import annotations

import datetime as dt
import requests
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from pathlib import Path


# ----------------------------
# 1) CONFIG
# ----------------------------
DATA_DIR = "data"
BLOOMS_FILE = "cherry-blossoms_fig-1.csv"   # Yoshino peak bloom date is DOY (e.g., 79, 97, 104)
NINO_FILE = "La Nino Weekly.txt"

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# ----------------------------
# 2) HELPERS
# ----------------------------
def fahrenheit(c: pd.Series) -> pd.Series:
    return c * 9 / 5 + 32

def build_season_year(d: pd.Series) -> pd.Series:
    """Assign Nov–Dec to next season-year (winter belongs to next spring)."""
    return d.dt.year + (d.dt.month >= 11).astype(int)

def gdd_base(temp_f: pd.Series, base_f: float = 40.0) -> pd.Series:
    return np.maximum(temp_f - base_f, 0.0)

def calc_chill(tmin_f: float, tmax_f: float) -> float:
    """Lightweight heuristic chill scoring function."""
    tavg = (tmax_f + tmin_f) / 2
    if tmax_f < 32:
        return 0.0
    if tmax_f > 60 and tmin_f > 45:
        return -12.0
    if 32 <= tavg <= 45:
        return 18.0
    return 6.0

def doy_to_date(year: int, doy: float) -> dt.date:
    """Convert DOY to a calendar date."""
    return dt.date(year, 1, 1) + dt.timedelta(days=int(round(doy)) - 1)

# ----------------------------
# 3) FETCH WEATHER (Open-Meteo Archive)
# ----------------------------
def fetch_dca_weather(start_date: str = "1981-01-01", end_date: str | None = None) -> pd.DataFrame:
    """Fetch daily DCA-ish weather via Open-Meteo archive (no API key).
    Caps end_date at today's date to avoid 400 errors from future dates.
    """
    if end_date is None:
        end_date = dt.date.today().strftime("%Y-%m-%d")

    today_str = dt.date.today().strftime("%Y-%m-%d")
    end_date = min(end_date, today_str)

    params = {
        "latitude": 38.85,
        "longitude": -77.04,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "America/New_York",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    weather = pd.DataFrame(
        {
            "Date": pd.to_datetime(daily["time"]),
            "tmax_c": daily["temperature_2m_max"],
            "tmin_c": daily["temperature_2m_min"],
            "precip": daily["precipitation_sum"],
        }
    )
    weather["tmax_f"] = fahrenheit(weather["tmax_c"])
    weather["tmin_f"] = fahrenheit(weather["tmin_c"])
    weather["tavg_f"] = (weather["tmax_f"] + weather["tmin_f"]) / 2
    return weather

# ----------------------------
# 4) BUILD MASTER TABLE
# ----------------------------
def build_master_table() -> tuple[pd.DataFrame, pd.DataFrame, dt.date]:
    """Builds master + current features row + cutoff date."""
    # A) Bloom targets (DOY already)
    blooms = pd.read_csv(f"{DATA_DIR}/{BLOOMS_FILE}", skiprows=6)
    blooms.columns = [c.strip() for c in blooms.columns]

    blooms = blooms.dropna(subset=["Year", "Yoshino peak bloom date"]).copy()
    blooms["Year"] = pd.to_numeric(blooms["Year"], errors="coerce")
    blooms["peak_doy"] = pd.to_numeric(blooms["Yoshino peak bloom date"], errors="coerce")
    blooms = blooms.dropna(subset=["Year", "peak_doy"]).copy()
    blooms["Year"] = blooms["Year"].astype(int)
    blooms["peak_doy"] = blooms["peak_doy"].astype(int)

    # B) ENSO weekly Niño3.4 (4 header lines)
    nino = pd.read_fwf(
        f"{DATA_DIR}/{NINO_FILE}",
        skiprows=4,
        names=["Week", "Nino1_2", "Nino3", "Nino34", "Nino4"],
    )
    nino["Week"] = nino["Week"].astype(str).str.strip()
    nino["Week"] = pd.to_datetime(nino["Week"], format="%d%b%Y", errors="coerce")
    nino = nino.dropna(subset=["Week"]).copy()
    nino["Nino34"] = pd.to_numeric(nino["Nino34"], errors="coerce")
    nino["season_yr"] = build_season_year(nino["Week"])

    winter_nino = nino[nino["Week"].dt.month.isin([12, 1, 2])].copy()
    nino_feat = (
        winter_nino.groupby("season_yr")["Nino34"].mean().reset_index()
        .rename(columns={"Nino34": "winter_nino34"})
    )

    # C) Weather -> Chill + GDD
    weather = fetch_dca_weather()
    weather["gdd40"] = gdd_base(weather["tavg_f"], base_f=40.0)
    weather["chill_hrs"] = weather.apply(lambda r: calc_chill(r["tmin_f"], r["tmax_f"]), axis=1)
    weather["season_yr"] = build_season_year(weather["Date"])

    active = weather[weather["Date"].dt.month.isin([11, 12, 1, 2, 3, 4])].copy()
    active = active.sort_values("Date")
    active["cum_chill"] = active.groupby("season_yr")["chill_hrs"].cumsum()

    gdd_subset = active[active["Date"].dt.month.isin([1, 2, 3, 4])].copy()
    gdd_subset["cum_gdd"] = gdd_subset.groupby("season_yr")["gdd40"].cumsum()
    active = active.merge(gdd_subset[["Date", "cum_gdd"]], on="Date", how="left")
    active["cum_gdd"] = active["cum_gdd"].fillna(0.0)

    mar1 = active[(active["Date"].dt.month == 3) & (active["Date"].dt.day == 1)].copy()
    features = mar1[["season_yr", "cum_chill", "cum_gdd"]].rename(
        columns={"cum_chill": "chill_mar1", "cum_gdd": "gdd_mar1"}
    )

    # D) Master merge
    master = blooms[["Year", "peak_doy"]].rename(columns={"Year": "season_yr"})
    master = master.merge(features, on="season_yr", how="inner")
    master = master.merge(nino_feat, on="season_yr", how="inner")
    master = master.dropna().sort_values("season_yr").reset_index(drop=True)

    # E) Current feature row
    current_year = int(active["season_yr"].max())

    curr_mar1 = active[
        (active["season_yr"] == current_year)
        & (active["Date"].dt.month == 3)
        & (active["Date"].dt.day == 1)
    ].copy()

    if len(curr_mar1) > 0:
        curr_pheno = curr_mar1.iloc[0]
        cutoff_date = curr_pheno["Date"].date()
    else:
        curr_window = active[
            (active["season_yr"] == current_year)
            & (active["Date"] < pd.Timestamp(f"{current_year}-03-01"))
        ].copy()
        if len(curr_window) == 0:
            raise ValueError(f"No data available before Mar 1 for season_yr={current_year}.")
        curr_pheno = curr_window.iloc[-1]
        cutoff_date = curr_pheno["Date"].date()

    curr_nino = winter_nino[winter_nino["season_yr"] == current_year]["Nino34"].mean()

    x_current = pd.DataFrame(
        {
            "chill_mar1": [curr_pheno["cum_chill"]],
            "gdd_mar1": [curr_pheno["cum_gdd"]],
            "winter_nino34": [curr_nino],
        }
    )

    return master, x_current, cutoff_date

# ----------------------------
# 5) RUN PIPELINE
# ----------------------------
master, x_current, cutoff_date = build_master_table()

print("master rows:", len(master))
print("year range:", master["season_yr"].min(), "to", master["season_yr"].max())
print("current cutoff date used:", cutoff_date)
print(master.tail())
print(x_current)

# ----------------------------
# 6) BASELINE (Holdout-by-year)
# ----------------------------
master = master.sort_values("season_yr").reset_index(drop=True)
X = master[["chill_mar1","gdd_mar1","winter_nino34"]]
y = master["peak_doy"]

cut = int(len(master) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("Holdout-by-year RF MAE:", round(mean_absolute_error(y_test, pred), 2))
print("Holdout years:", master["season_yr"].iloc[cut:].tolist())

# ----------------------------
# 7) Walk-forward XGBoost
# ----------------------------
n_splits = min(5, len(master) - 1)
tscv = TimeSeriesSplit(n_splits=n_splits)

maes = []
for tr_idx, te_idx in tscv.split(X):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    maes.append(mean_absolute_error(y_te, preds))

print("\nWalk-forward (TimeSeriesSplit) — XGBoost")
print("MAE per fold:", [round(m, 2) for m in maes])
print("Mean MAE (days):", round(float(np.mean(maes)), 2))

# ----------------------------
# 8) FINAL MODEL + FORECAST
# ----------------------------
final = xgb.XGBRegressor(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)
final.fit(X, y)

pred_doy = float(final.predict(x_current)[0])
year = dt.date.today().year
pred_date = doy_to_date(year, pred_doy)

print("\nForecast")
print(f"Predicted peak bloom for {year}: {pred_date.strftime('%b %d, %Y')} (DOY {int(round(pred_doy))})")
print(f"(Used cutoff features from: {cutoff_date})")
print("TimeSeries MAE mean±std:", round(np.mean(maes),2), "±", round(np.std(maes),2))

out = Path("outputs")
out.mkdir(exist_ok=True)

summary = pd.DataFrame([
    {
        "metric": "master_rows",
        "value": len(master),
        "notes": "Training seasons used after merging bloom + weather + ENSO"
    },
    {
        "metric": "year_range",
        "value": f"{int(master['season_yr'].min())}–{int(master['season_yr'].max())}",
        "notes": "Season years available in training set"
    },
    {
        "metric": "cutoff_date",
        "value": str(cutoff_date),
        "notes": "Date used for current-year feature row (latest < Mar 1 if Mar 1 not available)"
    },
    {
        "metric": "x_current_chill_mar1",
        "value": float(x_current["chill_mar1"].iloc[0]),
        "notes": "Current-year chill feature"
    },
    {
        "metric": "x_current_gdd_mar1",
        "value": float(x_current["gdd_mar1"].iloc[0]),
        "notes": "Current-year GDD feature"
    },
    {
        "metric": "x_current_winter_nino34",
        "value": float(x_current["winter_nino34"].iloc[0]),
        "notes": "Current-year winter Niño3.4 feature"
    },
    {
        "metric": "walkforward_mae_mean_days",
        "value": float(np.mean(maes)),
        "notes": "Mean MAE across TimeSeriesSplit folds"
    },
    {
        "metric": "walkforward_mae_std_days",
        "value": float(np.std(maes)),
        "notes": "Std of MAE across folds"
    },
    {
        "metric": "walkforward_mae_folds_days",
        "value": str([round(m, 2) for m in maes]),
        "notes": "Per-fold MAE values"
    },
    {
        "metric": "forecast_predicted_doy",
        "value": int(round(pred_doy)),
        "notes": "Predicted DOY for peak bloom"
    },
    {
        "metric": "forecast_predicted_date",
        "value": pred_date.strftime("%Y-%m-%d"),
        "notes": "Predicted peak bloom date"
    },
])

summary.to_csv(out / "results_summary.csv", index=False)

# markdown table (nice for README/Substack)
summary_md = summary[["metric", "value", "notes"]].to_markdown(index=False)
(out / "results_summary.md").write_text(summary_md, encoding="utf-8")

print("\nSaved results:")
print(" - outputs/results_summary.csv")
print(" - outputs/results_summary.md")


def main():
    master, x_current, cutoff_date = build_master_table()
    # (keep the rest of your printing + training + forecast code here)
    return master, x_current, cutoff_date

if __name__ == "__main__":
    main()