# DC Cherry Blossom Predictor 🌸 (Time-Series ML)

This project predicts Washington, DC Yoshino peak bloom timing (**day-of-year, DOY**) using:
- Winter chill accumulation (`chill_mar1`)
- Early-spring heat accumulation via GDD (`gdd_mar1`, base 40°F)
- Winter ENSO signal (mean Niño3.4 over Dec–Feb)

The notebook and script implement the **same** end-to-end pipeline that produced the **Apr 03, 2026 (DOY 93)** forecast (cutoff: **2026-02-24**).

## Files
- **Notebook (single-cell, end-to-end):** `notebooks/DC_Cherry_Blossom_Predictor.ipynb`
- **Script version of the same code:** `src/pipeline.py`

## Data
Put these in `data/`:
- `cherry-blossoms_fig-1.csv` (target is already DOY in `Yoshino peak bloom date`)
- `La Nino Weekly.txt` (weekly Niño indices; this repo assumes 4 header lines and uses `skiprows=4`)

## Run
### Notebook
Open the notebook and run the single cell.

### Script
```bash
pip install -r requirements.txt
python src/pipeline.py
```

## Notes
- Walk-forward validation uses `TimeSeriesSplit`.
- Current-year features use March 1 if available; otherwise the latest available date before March 1 (printed as cutoff).
