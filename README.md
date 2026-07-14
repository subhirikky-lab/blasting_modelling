# Pit 3 Ground Vibration Prediction

Two Streamlit apps in one repo.

## `app.py` — Paper Version (LOCKED)

- 27 measurements from 20 blast events (data cutoff 14 June 2026).
- Built-in dataset, no upload required.
- Constants match the ISEE 2027 paper: k = 52,916, n = 1.381, R² = 0.566.
- **Do not modify.** This is the reference version for the paper.

## `app_lapangan.py` — Field Version (LIVE)

- No built-in dataset. User uploads an Excel file each time.
- Every upload triggers a full re-calibration (k, n, weighting factor, Sobol, LOOCV, design value).
- Use this for day-to-day blast planning as new measurements come in.
- Numerical logic identical to `app.py`; the only difference is the data source.

## Excel Format

Required columns:

```
Amaks (mm/s^s) Maks, nilai_g, distance_m, charge_kg,
geological_condt, tie_up_type, measuring_elevation,
row_number, controll_ms, wall_echelon_ms,
freeface_echelon_ms, freeface_count, depth_m, hole_diameter_mm
```

Rows with `row_number >= 10` are excluded automatically (out of current operating regime).
Rows with `freeface_count = 0` are corrected to `1`.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py            # paper version
streamlit run app_lapangan.py    # field version
```

## Deploy on Streamlit Cloud

Point one Streamlit Cloud app at `app.py` and a separate one at `app_lapangan.py`. Same repo, two deployed apps.

## Adding This File to Your Existing Repo

If you already have `app.py` on GitHub, add the field app like this:

```bash
git pull                              # get the latest state
git add app_lapangan.py README.md
git commit -m "Add field version app_lapangan.py"
git push
```

Or upload `app_lapangan.py` directly through the GitHub web interface (Add file → Upload files).
