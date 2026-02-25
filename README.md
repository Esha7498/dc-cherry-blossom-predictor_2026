# DC Cherry Blossom Predictor 🌸 (Time-Series ML)

Forecast **Washington, DC (Tidal Basin Yoshino)** peak bloom timing as **day-of-year (DOY)** using an interpretable, phenology-inspired ML pipeline with **time-aware validation**.

---

## Project summary

This project predicts peak bloom based on three drivers that are strongly linked to bloom phenology:

- **Winter chill accumulation** (`chill_mar1`)  
- **Early-spring heat accumulation** via **GDD** (`gdd_mar1`, base **40°F**)  
- **Winter ENSO signal** (mean **Niño3.4** over **Dec–Feb**, `winter_nino34`)

The pipeline is implemented as a reproducible script (`src/pipeline.py`) and generates a 2026 forecast using weather data available up to a clear cutoff date.

---

## Results (reproducible)

**Training data after merges:** 35 season-years (**1982–2016**)  
**Current-year cutoff used for features:** **2026-02-24** (latest available date before Mar 1)

### Baselines and evaluation
- **Holdout-by-year baseline (Random Forest, last 20% years = 2010–2016):**  
  **MAE = 5.16 days**
- **Walk-forward validation (TimeSeriesSplit) — XGBoost:**  
  - Fold MAE: **[6.18, 2.87, 3.97, 3.52, 5.74] days**  
  - Mean ± Std MAE: **4.46 ± 1.28 days**

### 2026 forecast (as-of cutoff)
Using features available as of **Feb 24, 2026**, the model predicts:
- **Peak bloom ≈ Apr 03, 2026 (DOY 93)**

> Interpretation: historical walk-forward error is ~4–5 days, so the forecast should be treated as a **window**, not an exact date.

---

## Method (technical details)

### Data sources
- **Peak bloom target (DOY):** `data/cherry-blossoms_fig-1.csv`  
  - Column: `Yoshino peak bloom date` (already recorded as **days from Jan 1**)  
- **ENSO (Niño indices):** `data/La Nino Weekly.txt`  
  - Parsed via fixed-width format (`read_fwf`) with `skiprows=4`  
  - Feature created: mean Niño3.4 anomaly over **Dec–Feb** (`winter_nino34`)
- **Daily weather (DC area):** pulled programmatically from **Open-Meteo Archive API**  
  - daily max/min temperature (converted to °F)

### Feature engineering
All weather features are aligned to a **season-year** definition (Nov–Dec assigned to the next year) so winter conditions map to the following spring bloom.

- `chill_mar1`: cumulative chill score from Nov → Mar 1  
  - heuristic chill scoring (rule-based) from daily min/max temps  
- `gdd_mar1`: cumulative growing degree days from Jan → Mar 1  
  - GDD = max(Tavg°F − 40, 0)  
- `winter_nino34`: average Niño3.4 over Dec–Feb

**Current-year logic:** if Mar 1 is not yet available, the script uses the **latest available date before Mar 1** and prints it as `cutoff_date`.

### Modeling
- **Primary model:** `XGBRegressor`
- **Evaluation:** **walk-forward** `TimeSeriesSplit` (no leakage)
- **Metric:** MAE in **days** (directly interpretable)

---

## Repository structure


dc-cherry-blossom-predictor/
├─ src/
│ ├─ pipeline.py
│ └─ visuals.py
├─ data/  add files locally
├─ outputs/ # generated plots + summary tables
├─ requirements.txt
└─ README.md


---

## Setup

### 1) Install dependencies
```bash
pip install -r requirements.txt
2) Add data files

Place these in data/:

cherry-blossoms_fig-1.csv

La Nino Weekly.txt

(Recommended: do not commit these files—keep them local and list the sources.)

Run
Run the pipeline (metrics + forecast)
python src/pipeline.py
Generate plots (saved to outputs/)
python src/visuals.py
Notes / limitations

The final training set contains 35 years after merging bloom + weather + ENSO overlap.

A small dataset limits certainty; performance is reported using time-aware validation to avoid inflated metrics.

Forecasts can shift as new late-winter temperatures update chill/GDD features (cutoff date is logged).

