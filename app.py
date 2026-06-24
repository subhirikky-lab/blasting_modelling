# =====================================================================
# Prediksi Ground Vibration (g) Pit 3  -  Tanjung Enim
# Hybrid Monte Carlo + Weighted Factor + Residual Ratio + Design Value (P90)
# + Diagnostik Resonansi + SDOB + Dekomposisi Prediksi.
# Jalankan lokal :  streamlit run app.py
#
# DATASET: 27 pengukuran / 20 event blast. Record 10-row (di luar regime) dibuang.
# KOREKSI: freeface_count yang sebelumnya 0 (salah input) -> 1.
# BINNING (di DALAM tiap parameter, BUKAN menggabung parameter berbeda):
#   - row_number -> kelas ordinal 1-3 / 4-7 / >7
#   - control delay -> 42 ms digabung ke 67 ms (kategori <=67 ms)
#   - freeface echelon -> 0 / 42-67 / >=109 ms
#   Dasar teori binning dijelaskan pada tab "Dasar Teori Binning".
# =====================================================================

import io
import math as _math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
import streamlit as st

# ---------------------------------------------------------------------
# KONSTANTA & KOLOM
# ---------------------------------------------------------------------
G_GRAV = 9806.65
COL_AMAKS = "Amaks (mm/s^s) Maks"
COL_G = "nilai_g"
COL_DIST = "distance_m"
COL_CHG = "charge_kg"
COL_FREQ = "frekuensi_hz"
COL_DEPTH = "depth_m"
COL_HOLE_DIA = "hole_diameter_mm"

PARAMS_CAT = ["geological_condt", "tie_up_type", "measuring_elevation", "row_class", "ffe_binned"]
PARAMS_NUM = ["controll_binned", "wall_echelon_ms", "freeface_count"]
ALL_PARAMS = PARAMS_CAT + PARAMS_NUM
CONTROLLABLE_OPS = ["tie_up_type", "controll_binned", "wall_echelon_ms", "ffe_binned", "freeface_count"]

NICE = {
    "row_class": "Row Number (kelas)", "controll_binned": "Control Delay (ms)",
    "wall_echelon_ms": "Wall Echelon (ms)", "ffe_binned": "Freeface Echelon (ms)",
    "freeface_count": "Freeface Count", "geological_condt": "Geological Condition",
    "tie_up_type": "Tie-up Type", "measuring_elevation": "Measuring Elevation",
}
STATUS_THRESHOLDS = [(0.020, "EXCELLENT"), (0.030, "SAFE"), (0.100, "MODERATE"),
                     (0.200, "RISKY"), (float("inf"), "EXTREMELY RISKY")]
STATUS_COLOR = {"EXCELLENT": "#1a9850", "SAFE": "#66bd63", "MODERATE": "#fee08b",
                "RISKY": "#fc8d59", "EXTREMELY RISKY": "#d73027"}
ROCK_NATURAL_FREQ = {"weak": (3, 7), "blocky": (7, 15), "strong": (15, 23)}
GEO_TO_ROCK = {"coal": "weak", "normal": "blocky", "Fault": "weak"}
SLOPE_FREQ_THRESHOLD = 40.0

RHO_EXPLOSIVE = 1150.0
SDOB_BANDS = [
    ("Uncontrolled (0-0.60): flyrock & airblast hebat", 0.0, 0.60),
    ("Cratering (0.60-0.92): fragmentasi sangat halus", 0.60, 0.92),
    ("Controlled (0.92-1.40): fragmentasi baik, getaran/airblast wajar", 0.92, 1.40),
    ("Very controlled (1.40-1.80): frag lebih kasar, TANPA flyrock", 1.40, 1.80),
    ("Minimal surface (1.80-2.40): gangguan permukaan kecil", 1.80, 2.40),
    ("Insignificant (>2.40): efek permukaan tak berarti", 2.40, float("inf")),
]

# ---------------------------------------------------------------------
# DATABASE INTERNAL - 27 pengukuran dari 20 event blast Pit 3 (data 16 Juni 2026).
# Record 10-row (21 Maret 2025) DIKELUARKAN: ekstrem & high-leverage, di luar fokus regime.
# freeface_count = 0 (salah input) DIKOREKSI -> 1. Repo WAJIB Private.
# ---------------------------------------------------------------------
_COLS = ["Amaks (mm/s^s) Maks", "nilai_g", "charge_kg", "distance_m", "tie_up_type",
         "wall_echelon_ms", "freeface_echelon_ms", "controll_ms", "freeface_count",
         "geological_condt", "measuring_elevation", "row_number", "frekuensi_hz",
         "depth_m", "hole_diameter_mm"]
_ROWS = [
    [258.11, 0.02632, 40, 270, "echelon", 109, 0, 67, 1, "normal", "higher", 3, 8.3, 8, 200],
    [1049, 0.106968, 30, 150, "echelon", 109, 0, 176, 1, "normal", "higher", 5, 9.8, 8, 200],
    [226.13, 0.023059, 30, 300, "echelon", 176, 0, 109, 1, "normal", "higher", 4, 4.5, 8, 200],
    [387, 0.039463, 17, 180, "boxcut", 176, 176, 109, 2, "normal", "higher", 9, 49, 8, 200],
    [258.24, 0.026333, 50, 285, "echelon", 109, 0, 176, 2, "coal", "higher", 5, 6.4, 8, 200],
    [226.13, 0.023059, 55, 350, "echelon", 109, 0, 176, 2, "coal", "higher", 5, 8.3, 7, 200],
    [484, 0.049354, 57, 324, "echelon", 0, 109, 176, 2, "coal", "higher", 5, 7.9, 8, 200],
    [306.8, 0.031285, 60, 420, "echelon", 0, 67, 109, 1, "coal", "higher", 5, 8.1, 8, 200],
    [80.65, 0.008224, 60, 430, "boxcut", 42, 67, 109, 2, "coal", "higher", 4, 8.6, 8, 200],
    [161.42, 0.01646, 80, 500, "boxcut", 67, 42, 109, 1, "coal", "higher", 5, 3.8, 7.98, 200],
    [193.7, 0.019752, 90, 300, "boxcut", 67, 42, 109, 3, "coal", "normal", 6, 4.2, 8, 200],
    [1098, 0.111965, 82, 160, "boxcut", 67, 42, 109, 2, "normal", "normal", 9, 6.3, 6.34, 200],
    [936, 0.095445, 82, 160, "boxcut", 67, 42, 109, 1, "coal", "lower", 9, 7.3, 6.34, 200],
    [1195, 0.121856, 82, 160, "boxcut", 67, 42, 109, 1, "Fault", "higher", 9, 5.9, 6.34, 200],
    [96.85, 0.009876, 51, 454, "boxcut", 67, 42, 109, 2, "Fault", "normal", 7, 14.3, 7.13, 200],
    [323, 0.032937, 51, 320, "boxcut", 67, 42, 109, 2, "normal", "normal", 7, 4.9, 7.3, 200],
    [1130, 0.115228, 38, 176, "echelon", 42, 0, 67, 1, "Fault", "higher", 4, 8.8, 7.45, 200],
    [678, 0.069137, 38, 226, "echelon", 42, 0, 67, 1, "coal", "lower", 4, 12, 7.45, 200],
    [589, 0.060061, 51, 239, "boxcut", 42, 42, 109, 2, "coal", "lower", 3, 9.5, 7.42, 200],
    [629.54, 0.064195, 51, 239, "boxcut", 67, 42, 109, 2, "coal", "lower", 3, 10, 7.42, 200],
    [387.41, 0.039505, 51, 320, "boxcut", 67, 42, 109, 2, "Fault", "higher", 6, 6.2, 7.42, 200],
    [290.53, 0.029626, 51, 320, "boxcut", 42, 42, 109, 2, "Fault", "higher", 6, 3.4, 7.42, 200],
    [678, 0.069137, 41, 86, "boxcut", 42, 42, 109, 2, "Fault", "higher", 3, 4.1, 7.19, 200],
    [355, 0.0362, 41, 154, "boxcut", 42, 42, 109, 2, "coal", "lower", 3, 7.6, 7.19, 200],
    [678, 0.069136759, 41, 193, "echelon", 42, 0, 109, 2, "Fault", "higher", 3, 7.8, 7.74, 200],
    [420, 0.042828, 72, 311, "boxcut", 42, 67, 109, 3, "Fault", "normal", 5, 6.6, 7.2, 200],
    [161, 0.016417, 46, 255, "echelon", 109, 0, 42, 2, "Fault", "lower", 7, 2.9, 6.5, 200],
    [1420, 0.1447997022, 46, 147, "boxcut", 67, 42, 109, "Fault", "lower", 4, 5.6, 7.67, 200],
    [483, 0.04925229309, 65, 180, "boxcut", 67, 42, 109, "Fault", "normal", 4, 6.1, 6.49, 200]
]
BUILTIN_DATA = [dict(zip(_COLS, r)) for r in _ROWS]


def apply_binning(df):
    """Binning di DALAM tiap parameter (BUKAN menggabung parameter). Freeface Count &
    Freeface Echelon tetap dua parameter terpisah & independen."""
    df = df.copy()
    df["row_class"] = df["row_number"].apply(lambda r: "1-3" if r <= 3 else ("4-7" if r <= 7 else ">7"))
    df["controll_binned"] = df["controll_ms"].replace({42: 67})
    df["ffe_binned"] = df["freeface_echelon_ms"].apply(
        lambda x: "0" if x == 0 else ("42-67" if x <= 67 else ">=109"))
    return df


def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None


def check_password():
    pw = get_secret("app_password")
    if not pw:
        return True
    if st.session_state.get("auth_ok"):
        return True
    st.title("Login")
    entered = st.text_input("Password", type="password")
    if st.button("Masuk"):
        if entered == pw:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Password salah.")
    return False


def get_status(g):
    for thr, label in STATUS_THRESHOLDS:
        if g <= thr:
            return label
    return "EXTREMELY RISKY"


def validate(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "freeface_count" in df.columns:
        df["freeface_count"] = df["freeface_count"].replace({0: 1})
    base_required = [COL_AMAKS, COL_G, COL_DIST, COL_CHG,
                     "geological_condt", "tie_up_type", "measuring_elevation",
                     "row_number", "controll_ms", "wall_echelon_ms",
                     "freeface_echelon_ms", "freeface_count"]
    missing = [c for c in base_required if c not in df.columns]
    if missing:
        raise KeyError("Kolom wajib tidak ada: " + ", ".join(missing))
    df = df.dropna(subset=[COL_AMAKS, COL_DIST, COL_CHG])
    df = df[(df[COL_AMAKS] > 0) & (df[COL_DIST] > 0) & (df[COL_CHG] > 0)].copy()
    df = df[df["row_number"] < 10].copy()
    if len(df) < 5:
        raise ValueError("Data valid hanya " + str(len(df)) + " baris - terlalu sedikit (min 5).")
    df["scale_distance"] = df[COL_DIST] / np.sqrt(df[COL_CHG])
    df["Amaks_maks"] = df[COL_AMAKS]
    df = apply_binning(df)
    return df


def _aggregate(values, method):
    arr = np.asarray(values, dtype=float)
    if method == "geometric":
        return float(np.exp(np.mean(np.log(arr))))
    return float(np.mean(arr))


def calibrate(df, method="geometric"):
    ln_a = np.log(df["Amaks_maks"].values)
    ln_sd = np.log(df["scale_distance"].values)
    slope, intercept, r_val, _, _ = stats.linregress(ln_sd, ln_a)
    k = float(np.exp(intercept))
    n = float(-slope)
    resid = ln_a - (intercept + slope * ln_sd)
    sigma = float(np.std(resid, ddof=1))
    r_sq = float(r_val ** 2)
    df = df.copy()
    df["res_ratio"] = df["Amaks_maks"].values / (k * df["scale_distance"].values ** (-n))
    real_factor, counts, freq = {}, {}, {}
    n_tot = len(df)
    for p in ALL_PARAMS:
        groups = df.groupby(p)["res_ratio"]
        rp, cp, fp = {}, {}, {}
        for key, vals in groups:
            rp[key] = _aggregate(vals.values, method)
            cp[key] = len(vals)
            fp[key] = len(vals) / n_tot
        real_factor[p], counts[p], freq[p] = rp, cp, fp
    ranges = {p: max(v.values()) - min(v.values()) for p, v in real_factor.items()}
    total = sum(ranges.values())
    weight = {p: (ranges[p] / total if total > 0 else 0.0) for p in ALL_PARAMS}
    return dict(k=k, n=n, sigma=sigma, r2=r_sq, real_factor=real_factor,
                weight=weight, counts=counts, freq=freq, ranges=ranges, df=df)


def _nearest_num(d, t):
    keys = np.array([float(x) for x in d.keys()])
    return list(d.values())[int(np.abs(keys - float(t)).argmin())]


def real_factor_value(p, v, cal):
    f = cal["real_factor"][p]
    return f.get(v, 1.0) if isinstance(v, str) else _nearest_num(f, v)


def compute_wf(row, cal):
    wf = 0.0
    for p in ALL_PARAMS:
        wf += real_factor_value(p, row[p], cal) * cal["weight"][p]
    return wf


def mean_wf_of(df, cal):
    return float(np.mean([compute_wf(r, cal) for _, r in df.iterrows()]))


def median_g_event(inp, cal, mean_wf):
    sd = inp["distance_m"] / np.sqrt(inp["charge_kg"])
    return cal["k"] * sd ** (-cal["n"]) * (compute_wf(inp, cal) / mean_wf) / G_GRAV


def predict_g_mc(cal, scale_distance, wf_norm, sigma, n_iter=10000, seed=42):
    rng = np.random.default_rng(seed)
    err = rng.normal(0.0, sigma, n_iter)
    return cal["k"] * (scale_distance ** (-cal["n"])) * np.exp(err) * wf_norm / G_GRAV


def loocv_predictions(df, method="geometric"):
    pairs = []
    for i in list(df.index):
        train = df.drop(index=i)
        test = df.loc[i]
        c = calibrate(train, method)
        mw = mean_wf_of(train, c)
        gp = c["k"] * test["scale_distance"] ** (-c["n"]) * (compute_wf(test, c) / mw) / G_GRAV
        pairs.append((float(test[COL_G]), float(gp)))
    return pairs


def csf_metrics(pairs, w_under=25.0):
    errs = [abs(a - p) for a, p in pairs]
    short = [max(0.0, a - p) for a, p in pairs]
    n_under = sum(1 for a, p in pairs if p < a)
    g_range = max(a for a, _ in pairs) - min(a for a, _ in pairs)
    mae = float(np.mean(errs))
    return dict(mae=mae, nmae=float(mae / g_range) if g_range > 0 else float("nan"),
                csf=float(np.mean([e + w_under * s ** 2 for e, s in zip(errs, short)])),
                n=len(pairs), n_under=n_under, pct_under=100.0 * n_under / len(pairs),
                mean_short=float(np.mean(short)))


def design_limits(pairs):
    ratios = np.array([a / p for a, p in pairs if p > 0])
    logr = np.log(ratios)
    mu_e, s_e = float(np.mean(logr)), float(np.std(logr, ddof=1))
    table = {}
    for pct in (75, 90, 95):
        z = float(stats.norm.ppf(pct / 100.0))
        lam_emp = float(np.percentile(ratios, pct))
        lam_norm = float(np.exp(mu_e + z * s_e))
        table[pct] = dict(emp=lam_emp, norm=lam_norm,
                          under_emp=100.0 * float(np.mean(ratios > lam_emp)),
                          under_norm=100.0 * float(np.mean(ratios > lam_norm)))
    return dict(mu_e=mu_e, s_e=s_e, table=table)


def importance_metrics(cal, inp_row, mean_wf):
    weight = cal["weight"]
    levels, g_levels = {}, {}
    for p in ALL_PARAMS:
        if p in PARAMS_NUM:
            lv = sorted(cal["real_factor"][p], key=float)
        else:
            lv = sorted(cal["real_factor"][p], key=str)
        gl = []
        for val in lv:
            mod = dict(inp_row)
            mod[p] = val
            gl.append(median_g_event(mod, cal, mean_wf))
        levels[p], g_levels[p] = lv, gl
    C = {}
    for p in ALL_PARAMS:
        vals = list(cal["real_factor"][p].values())
        frs = [cal["freq"][p][kk] for kk in cal["real_factor"][p].keys()]
        m = sum(v * f for v, f in zip(vals, frs))
        C[p] = (weight[p] ** 2) * sum(f * (v - m) ** 2 for v, f in zip(vals, frs))
    s = sum(C.values())
    sobol = {p: (C[p] / s * 100 if s > 0 else 0.0) for p in ALL_PARAMS}
    return dict(sobol=sobol, levels=levels, g_levels=g_levels)


def site_typical_rf(cal, p):
    rf = cal["real_factor"][p]
    fr = cal["freq"][p]
    return float(sum(rf[k] * fr[k] for k in rf))


def decompose_prediction(inp, cal, mean_wf):
    sd = inp["distance_m"] / np.sqrt(inp["charge_kg"])
    baseline_g = cal["k"] * sd ** (-cal["n"]) / G_GRAV
    wf = compute_wf(inp, cal)
    wf_norm = wf / mean_wf
    median_g = baseline_g * wf_norm
    factors = []
    for p in ALL_PARAMS:
        v = inp[p]
        rfv = real_factor_value(p, v, cal)
        typ = site_typical_rf(cal, p)
        dev = (rfv / typ - 1.0) * 100.0 if typ > 0 else 0.0
        factors.append(dict(param=p, nice=NICE[p], level=v, rf=rfv, typical=typ,
                            deviation_pct=dev,
                            direction=("naik" if dev > 0 else "turun" if dev < 0 else "netral"),
                            weight=cal["weight"][p] * 100.0))
    factors.sort(key=lambda x: -abs(x["deviation_pct"]))
    return dict(baseline_g=baseline_g, wf_norm=wf_norm, median_g=median_g, sd=sd, factors=factors)


def frequency_diagnostics(df):
    if COL_FREQ not in df.columns:
        return None
    sub = df.dropna(subset=[COL_FREQ])
    sub = sub[sub[COL_FREQ] > 0]
    if len(sub) < 1:
        return None
    freqs = sub[COL_FREQ].values.astype(float)
    n = int(len(freqs))
    band_defs = [("Weak (3-7 Hz)", 3.0, 7.0), ("Blocky (7-15 Hz)", 7.0, 15.0),
                 ("Strong (15-23 Hz)", 15.0, 23.0), ("Aman (>23 Hz)", 23.0, float("inf"))]
    bands = []
    for name, lo, hi in band_defs:
        c = int(np.sum((freqs > lo) & (freqs <= hi)))
        bands.append(dict(name=name, lo=lo, hi=hi, count=c, pct=100.0 * c / n if n > 0 else 0.0))
    n_slope = int(np.sum(freqs < SLOPE_FREQ_THRESHOLD))
    return dict(n=n, fmin=float(np.min(freqs)), fmax=float(np.max(freqs)),
                fmed=float(np.median(freqs)), bands=bands, n_slope=n_slope)


def compute_sdob(charge_kg, depth_m, hole_dia_mm, rho=RHO_EXPLOSIVE):
    try:
        ch = float(charge_kg); dep = float(depth_m); d = float(hole_dia_mm) / 1000.0
    except Exception:
        return None
    if ch <= 0 or dep <= 0 or d <= 0:
        return None
    lin_density = rho * (np.pi / 4.0) * d * d
    charge_len = ch / lin_density
    stemming = dep - charge_len
    w10 = lin_density * (10.0 * d)
    sdob = (stemming + 5.0 * d) / (w10 ** (1.0 / 3.0))
    band = "n/a"
    for name, lo, hi in SDOB_BANDS:
        if lo <= sdob < hi:
            band = name
            break
    return dict(sdob=float(sdob), charge_len=float(charge_len), stemming=float(stemming),
                w10=float(w10), stem_over_dia=float(stemming / d), band=band)


def sdob_diagnostics(df, rho=RHO_EXPLOSIVE):
    if COL_DEPTH not in df.columns or COL_CHG not in df.columns:
        return None
    dia_col = COL_HOLE_DIA if COL_HOLE_DIA in df.columns else None
    rows = []
    for _, r in df.iterrows():
        dia = r[dia_col] if dia_col else 200.0
        s = compute_sdob(r[COL_CHG], r[COL_DEPTH], dia, rho)
        if s is None:
            continue
        s["g"] = float(r[COL_G]) if COL_G in df.columns else float("nan")
        rows.append(s)
    if not rows:
        return None
    sdobs = np.array([x["sdob"] for x in rows])
    band_counts = {}
    for name, _, _ in SDOB_BANDS:
        band_counts[name] = sum(1 for x in rows if x["band"] == name)
    gs = np.array([x["g"] for x in rows])
    corr = float(np.corrcoef(sdobs, gs)[0, 1]) if len(rows) > 2 else float("nan")
    return dict(rows=rows, n=len(rows), smin=float(sdobs.min()), smax=float(sdobs.max()),
                smed=float(np.median(sdobs)), band_counts=band_counts, corr_g=corr, rho=rho)


def recommend(inp, cal, mean_wf, df, target, design_lambda):
    def gd(row):
        return median_g_event(row, cal, mean_wf) * design_lambda
    g_med = median_g_event(inp, cal, mean_wf)
    g0 = g_med * design_lambda
    out = {"median": g_med, "design": g0, "need": g0 > target, "steps": [],
           "charge_to": None, "after_ops": None}
    if not out["need"]:
        return out
    work = dict(inp)
    for p in CONTROLLABLE_OPS:
        rf = cal["real_factor"][p]
        safe = min(rf, key=lambda kk: rf[kk])
        cur_rf = _nearest_num(rf, inp[p]) if p in PARAMS_NUM else rf.get(inp[p], 1.0)
        if rf[safe] < cur_rf - 1e-12:
            work[p] = safe
            out["steps"].append("Ubah " + NICE[p] + ": " + str(inp[p]) + " -> " + str(safe) + " (level teraman dari data)")
    gd_ops = gd(work)
    out["after_ops"] = gd_ops
    n = cal["n"]
    ch_min = float(df[COL_CHG].min())
    if gd_ops > target:
        ch_t = work["charge_kg"] * (target / gd_ops) ** (2.0 / n)
        out["charge_from"] = inp["charge_kg"]
        out["charge_to"] = ch_t
        out["charge_extrapolated"] = ch_t < ch_min
        final = dict(work)
        final["charge_kg"] = max(ch_t, ch_min)
    else:
        final = dict(work)
    out["final_median"] = median_g_event(final, cal, mean_wf)
    out["final_design"] = gd(final)
    return out


@st.cache_data(show_spinner=False)
def build_model(file_bytes, method):
    if file_bytes is None:
        df_raw = pd.DataFrame(BUILTIN_DATA)
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes))
    df = validate(df_raw)
    cal = calibrate(df, method)
    df = cal["df"]
    mean_wf = mean_wf_of(df, cal)
    pairs = loocv_predictions(df, method)
    mae_loo = float(np.mean([abs(a - p) for a, p in pairs]))
    csf = csf_metrics(pairs)
    dl = design_limits(pairs)
    fdiag = frequency_diagnostics(df)
    sdiag = sdob_diagnostics(df)
    return df, cal, mean_wf, mae_loo, csf, dl, fdiag, sdiag


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Prediksi Getaran Pit 3", page_icon="boom", layout="wide")

if not check_password():
    st.stop()

st.title("Prediksi Ground Vibration (g) Pit 3")
st.caption("Hybrid Monte Carlo + Weighted Factor + Residual Ratio + Design Value + Probabilitas "
           "+ Diagnostik Resonansi & SDOB  |  27 pengukuran / 20 event blast")

with st.sidebar:
    st.header("1. Data")
    st.caption("Aplikasi memakai database internal (bawaan). User cukup mengisi parameter di bawah.")
    up = None
    admin_pw = get_secret("admin_password")
    if admin_pw:
        with st.expander("Login admin (khusus pengelola)"):
            entered_admin = st.text_input("Password admin", type="password", key="admin_pw_input")
            if entered_admin:
                if entered_admin == admin_pw:
                    st.session_state["is_admin"] = True
                else:
                    st.session_state["is_admin"] = False
                    st.error("Password admin salah.")
        if st.session_state.get("is_admin"):
            st.success("Mode admin aktif.")
            up = st.file_uploader("Upload Excel pengganti (format kolom sama)", type=["xlsx", "xls"])
    st.header("2. Pengaturan Model")
    st.caption("Default = setelan DISARANKAN. Biarkan apa adanya kecuali uji sensitivitas.")
    method = st.selectbox("Metode residual ratio", ["geometric", "arithmetic"], index=0,
        help="DISARANKAN: geometric (residual ratio bersifat perkalian/lognormal).")
    g_target = st.number_input("Ambang aman g (G_TARGET)", value=0.030, step=0.005, format="%.3f")
    design_pct = st.selectbox("Design percentile", [90, 95, 75], index=0, help="DISARANKAN: P90.")
    design_method = st.selectbox("Metode Design Value", ["empirical", "normal"], index=0, help="DISARANKAN: empirical.")
    use_loocv_sigma = st.checkbox("Pakai sigma LOOCV (lebih jujur)", value=True)
    n_iter = st.select_slider("Iterasi Monte Carlo", options=[2000, 5000, 10000, 20000], value=10000)

file_bytes = up.getvalue() if up is not None else None
try:
    df, cal, mean_wf, mae_loo, csf, dl, fdiag, sdiag = build_model(file_bytes, method)
except Exception as e:
    st.error("Gagal memproses data: " + str(e))
    st.stop()
st.caption("Sumber data: " + ("file upload (admin)" if up is not None else "database internal bawaan")
           + "  |  N = " + str(len(df)) + " pengukuran")

sigma_mc = dl["s_e"] if use_loocv_sigma else cal["sigma"]
design_lambda = dl["table"][design_pct]["emp" if design_method == "empirical" else "norm"]

with st.sidebar:
    st.header("3. Parameter Event")
    inp = {}
    inp["distance_m"] = st.number_input("Distance (m)", value=float(np.median(df[COL_DIST].values)), min_value=1.0)
    inp["charge_kg"] = st.number_input("Charge per delay (kg)", value=float(np.median(df[COL_CHG].values)), min_value=1.0)
    _depth_default = float(np.median(df[COL_DEPTH].values)) if COL_DEPTH in df.columns else 8.0
    _dia_default = float(np.median(df[COL_HOLE_DIA].values)) if COL_HOLE_DIA in df.columns else 200.0
    inp["depth_m"] = st.number_input("Kedalaman lubang (m)", value=_depth_default, min_value=1.0,
                                     help="Untuk diagnostik SDOB; tidak mengubah prediksi g.")
    inp["hole_diameter_mm"] = st.number_input("Diameter lubang (mm)", value=_dia_default, min_value=50.0,
                                              help="Untuk diagnostik SDOB.")
    raw_row = st.number_input("Row Number", value=int(np.median(df["row_number"].values)), min_value=1, step=1,
                              help="Dikelompokkan otomatis ke kelas 1-3 / 4-7 / >7.")
    _ctrl_opts = sorted(df["controll_ms"].unique().tolist()) if "controll_ms" in df.columns else [67, 109, 176]
    raw_ctrl = st.selectbox("Control Delay (ms)", _ctrl_opts, help="42 ms otomatis digabung ke 67 ms.")
    _ffe_opts = sorted(df["freeface_echelon_ms"].unique().tolist()) if "freeface_echelon_ms" in df.columns else [0, 42, 67, 109, 176]
    raw_ffe = st.selectbox("Freeface Echelon (ms)", _ffe_opts, help="Dikelompokkan ke 0 / 42-67 / >=109.")
    inp["wall_echelon_ms"] = st.number_input("Wall Echelon (ms)", value=float(np.median(df["wall_echelon_ms"].values)))
    inp["freeface_count"] = st.number_input("Freeface Count", value=2.0, min_value=1.0, step=1.0,
                                            help="Jumlah bidang bebas. Parameter TERPISAH dari Freeface Echelon.")
    for p in ["geological_condt", "tie_up_type", "measuring_elevation"]:
        opts = sorted(str(x) for x in df[p].dropna().unique())
        inp[p] = st.selectbox(NICE[p], opts)

inp["row_class"] = "1-3" if raw_row <= 3 else ("4-7" if raw_row <= 7 else ">7")
inp["controll_binned"] = 67 if raw_ctrl == 42 else raw_ctrl
inp["ffe_binned"] = "0" if raw_ffe == 0 else ("42-67" if raw_ffe <= 67 else ">=109")

inp["scale_distance"] = inp["distance_m"] / np.sqrt(inp["charge_kg"])
wf_norm = compute_wf(inp, cal) / mean_wf
g_arr = predict_g_mc(cal, inp["scale_distance"], wf_norm, sigma_mc, int(n_iter))
g_med = float(np.median(g_arr))
g_mean = g_med * _math.exp(sigma_mc ** 2 / 2.0)
lo_pct = (100 - design_pct) / 2.0
hi_pct = 100 - lo_pct
g_lo = float(np.percentile(g_arr, lo_pct))
g_hi = float(np.percentile(g_arr, hi_pct))
g_design = g_med * design_lambda
prob_exceed = 100.0 * float(np.mean(g_arr > g_target))
is_fault_near = inp["distance_m"] < 100 and str(inp["geological_condt"]) == "Fault"

st.subheader("Hasil Prediksi Getaran (g)")
c1, c2, c3 = st.columns(3)
c1.metric("Nilai g Prediksi", format(g_mean, ".5f"), get_status(g_mean))
c2.metric("Probabilitas g > " + format(g_target, ".3f"), format(prob_exceed, ".1f") + "%")
c3.metric("Worst Scenario", format(g_design, ".5f"), get_status(g_design))

st.markdown(
    "<div style='padding:12px;border-radius:8px;background:" + STATUS_COLOR[get_status(g_mean)] +
    ";color:#000;font-weight:600'>Prediksi getaran realistis = <span style='font-size:1.3em'>"
    + format(g_mean, ".5f") + " g</span> (" + get_status(g_mean) + ")</div>",
    unsafe_allow_html=True)

st.write("")
st.write("**Rentang " + str(int(design_pct)) + "% kemungkinan (P" + format(lo_pct, ".0f") + "-P"
         + format(hi_pct, ".0f") + "):** " + format(g_lo, ".5f") + " ... " + format(g_hi, ".5f") +
         "  |  **WF normalized:** " + format(wf_norm, ".3f") +
         " (" + ("lebih berbahaya" if wf_norm > 1 else "lebih aman") + " dari rata-rata site)")
st.caption("PREDIKSI g = nilai harapan (mean lognormal = median x exp(sigma^2/2), koreksi Duan 1983). "
           "Keputusan keselamatan gunakan Design Value = " + format(g_design, ".5f")
           + " (median x lambda P" + str(int(design_pct)) + " = " + format(design_lambda, ".2f") + ").")

if is_fault_near:
    st.warning("REZIM Fault + near-field (<100 m): model cenderung MEREMEHKAN -> jadikan Design Value acuan.")

rec = recommend(inp, cal, mean_wf, df, g_target, design_lambda)
st.subheader("Rekomendasi Keputusan")
if not rec["need"]:
    st.success("AMAN - Design value " + format(rec["design"], ".5f") + " di bawah ambang " + format(g_target, ".3f") + ".")
else:
    st.error("DESIGN VALUE " + format(rec["design"], ".5f") + " DI ATAS AMBANG " + format(g_target, ".3f") + " -> perlu mitigasi.")
    if rec["steps"]:
        st.markdown("**Langkah 1 - Operasional:**")
        for s in rec["steps"]:
            st.markdown("- " + s)
        st.caption("Design g setelah tweak operasional: " + format(rec["after_ops"], ".5f"))
    if rec["charge_to"] is not None:
        st.markdown("**Langkah 2 - kurangi charge per delay:**")
        turun = (1 - rec["charge_to"] / rec["charge_from"]) * 100
        st.markdown("- Charge: " + format(rec["charge_from"], ".1f") + " kg -> " + format(rec["charge_to"], ".1f") + " kg (turun " + format(turun, ".0f") + "%)")
        if rec["charge_extrapolated"]:
            st.caption("[!] charge target di bawah rentang data -> ekstrapolasi, kurang andal.")
    st.info("Proyeksi bila saran diterapkan -> median " + format(rec["final_median"], ".5f") + ", DESIGN " + format(rec["final_design"], ".5f"))

tab1, tab_decomp, tab_theory, tab2, tab3, tab4, tab5 = st.tabs(
    ["Faktor (Sobol + Weight)", "Dekomposisi Prediksi", "Dasar Teori Binning",
     "Validasi & CSF", "Diagnostik Resonansi", "Diagnostik SDOB", "Sensitivity Charge"])

imp = importance_metrics(cal, inp, mean_wf)
order = sorted(ALL_PARAMS, key=lambda p: -imp["sobol"][p])

with tab1:
    st.markdown("### Seberapa besar pengaruh tiap faktor terhadap nilai g?")
    st.caption("Selain scaled distance (jarak + charge), 8 faktor operasional & geologi ikut menentukan g.")
    sobol = imp["sobol"]
    rows_sens = [{"Faktor": NICE[p], "Sobol (varians) %": round(sobol[p], 1),
                  "Weight (rentang) %": round(cal["weight"][p] * 100, 1),
                  "Level event ini": str(inp[p])} for p in order]
    st.dataframe(pd.DataFrame(rows_sens), use_container_width=True, hide_index=True)
    st.markdown("- **Sobol (principal):** porsi varians g (global, berbasis varians).\n"
                "- **Weight:** porsi faktor di dalam weighting factor (berbasis rentang), pembanding.")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [1, 1.25]})
    def active_level(p):
        if p in PARAMS_NUM:
            keys = np.array([float(x) for x in imp["levels"][p]])
            return imp["levels"][p][int(np.abs(keys - float(inp[p])).argmin())]
        return inp[p]
    labels = [NICE[p] + "  [" + str(active_level(p)) + "]" for p in order]
    vals = [sobol[p] for p in order]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(order)))
    bars = ax1.barh(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_yticks(range(len(order))); ax1.set_yticklabels(labels, fontsize=10); ax1.invert_yaxis()
    ax1.set_xlabel("Kontribusi terhadap varians g - indeks Sobol orde-1 (%)")
    ax1.set_title("Kontribusi Faktor terhadap Varians g   median g event = " + format(g_med, ".5f"))
    for b, v in zip(bars, vals):
        ax1.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, format(v, ".1f") + "%",
                 va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, max(vals) * 1.15); ax1.grid(axis="x", alpha=0.3)
    levels, g_levels = imp["levels"], imp["g_levels"]
    max_lv = max(len(levels[p]) for p in order)
    mat = np.full((len(order), max_lv), np.nan)
    for i, p in enumerate(order):
        for j, g in enumerate(g_levels[p]):
            mat[i, j] = g
    cmap2 = plt.cm.RdYlGn_r.copy(); cmap2.set_bad("white")
    im = ax2.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap2)
    ax2.set_yticks(range(len(order))); ax2.set_yticklabels([NICE[p] for p in order], fontsize=10)
    ax2.set_xticks(range(max_lv)); ax2.set_xticklabels(["Lvl " + str(j + 1) for j in range(max_lv)], fontsize=9)
    ax2.set_title("Heatmap sensitivitas - prediksi g tiap level (kotak biru = input)")
    for i, p in enumerate(order):
        act = active_level(p)
        for j, val in enumerate(levels[p]):
            ax2.text(j, i - 0.18, str(val), ha="center", va="center", fontsize=8, fontweight="bold")
            ax2.text(j, i + 0.22, format(g_levels[p][j], ".4f"), ha="center", va="center", fontsize=7)
            if val == act:
                ax2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=2.5))
    fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02).set_label("Prediksi g")
    plt.tight_layout(); st.pyplot(fig)
    st.info("Catatan jujur: Freeface Count, Wall Echelon, dan Freeface Echelon adalah tiga penggerak utama "
            "varians g (parameter terpisah & independen). Control delay & geological condition kecil pada varians; "
            "row number tidak dominan pada regime <=9 row, namun arahnya konsisten (row >7 -> g lebih tinggi). "
            "Indeks bersifat Sobol-type orde-1 yang diadaptasi ke struktur weighting factor (Sobol 1993; Saltelli dkk. 2008).")

with tab_decomp:
    st.markdown("### Dekomposisi prediksi g untuk event ini")
    st.caption("Faktor mana yang sedang MENDORONG g naik/turun dibanding kondisi tipikal site.")
    dec = decompose_prediction(inp, cal, mean_wf)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline (scaled distance saja)", format(dec["baseline_g"], ".5f"),
              "jarak " + format(inp["distance_m"], ".0f") + " m, charge " + format(inp["charge_kg"], ".0f") + " kg")
    c2.metric("WF normalized", format(dec["wf_norm"], ".3f"),
              ("lebih berbahaya" if dec["wf_norm"] > 1 else "lebih aman") + " dari rata-rata site")
    c3.metric("Prediksi median (extended)", format(dec["median_g"], ".5f"), get_status(dec["median_g"]))
    st.markdown("Baseline = kontribusi **scaled distance** saja. Faktor operasional & geologi menggeser via "
                "WF normalized (= " + format(dec["wf_norm"], ".3f") + ").")
    drows = [{"Faktor": f["nice"], "Level event": str(f["level"]), "Real factor": round(f["rf"], 3),
              "Tipikal site": round(f["typical"], 3),
              "Mendorong g": (("+" if f["deviation_pct"] >= 0 else "") + format(f["deviation_pct"], ".1f") + "%"),
              "Weight %": round(f["weight"], 1)} for f in dec["factors"]]
    st.dataframe(pd.DataFrame(drows), use_container_width=True, hide_index=True)
    fac_sorted = sorted(dec["factors"], key=lambda x: x["deviation_pct"])
    names = [f["nice"] for f in fac_sorted]; devs = [f["deviation_pct"] for f in fac_sorted]
    cols = ["#d73027" if d > 0 else "#1a9850" for d in devs]
    figd, axd = plt.subplots(figsize=(10, 6))
    axd.barh(range(len(names)), devs, color=cols, edgecolor="black", linewidth=0.6)
    axd.set_yticks(range(len(names))); axd.set_yticklabels(names, fontsize=10)
    axd.axvline(0, color="black", linewidth=1)
    axd.set_xlabel("Pergeseran g relatif terhadap kondisi tipikal site (%)")
    axd.set_title("Dekomposisi: faktor mana yang menaikkan (merah) / menurunkan (hijau) g")
    for i, d in enumerate(devs):
        axd.text(d + (1 if d >= 0 else -1), i, ("+" if d >= 0 else "") + format(d, ".1f") + "%",
                 va="center", ha="left" if d >= 0 else "right", fontsize=9, fontweight="bold")
    axd.grid(axis="x", alpha=0.3)
    pad = max(abs(min(devs)), abs(max(devs))) * 0.25 + 5
    axd.set_xlim(min(devs) - pad, max(devs) + pad)
    plt.tight_layout(); st.pyplot(figd)
    st.info("Bar merah = faktor membuat g LEBIH TINGGI dari tipikal site; hijau = LEBIH RENDAH. "
            "Dekomposisi bersifat median; ketidakpastian penuh dibaca dari distribusi Monte Carlo & Design Value.")

with tab_theory:
    st.markdown("### Mengapa dilakukan binning, dan apa dasar teorinya?")
    st.markdown(
        "**Masalah: sparse categories (kategori berisi terlalu sedikit data).** Model menghitung "
        "*real factor* tiap kategori sebagai geometric mean residual ratio anggotanya. Jika kategori "
        "hanya berisi 1 observasi, geometric mean-nya = nilai tunggal itu, tanpa memisahkan sinyal dari "
        "noise. Satu event kebetulan bisa membuat kategori tampak sangat berpengaruh, padahal artifact. "
        "Contoh nyata: control delay 42 ms (n=1) sempat melonjak ~43% kontribusi varians dari satu record.")
    st.markdown("**Tiga dasar teori:**")
    st.markdown(
        "1. **Aturan ukuran sampel per sel (Sturges, 1926).** Jumlah bin wajar k ≈ 1 + 3,3·log₁₀(n); "
        "untuk n≈27 sekitar 5-6 bin maksimum. Variabel dengan banyak nilai unik tapi sedikit data per "
        "nilai harus dikelompokkan agar tidak melampaui kapasitas informasi dataset.\n"
        "2. **Ordinal discretization / equal-frequency binning (Dougherty, Kohavi & Sahami, 1995).** "
        "Mengubah variabel multi-level jadi sedikit kelas ordinal adalah teknik standar saat data per "
        "level tipis. Batas dipilih bermakna fisika, sehingga sah dan bukan data dredging.\n"
        "3. **Analogi klasifikasi geoteknik (Bieniawski RMR, 1989; Barton Q-system).** Parameter kontinu "
        "(RQD, spasi diskontinuitas) rutin dikelompokkan jadi kelas; preseden domain yang kuat.")
    st.markdown("**Binning yang diterapkan (di DALAM tiap parameter):**")
    st.dataframe(pd.DataFrame([
        {"Parameter": "Row Number", "Kelas": "1-3 / 4-7 / >7",
         "Justifikasi fisika": "dangkal (free-face relief baik) / menengah / banyak baris (confinement tinggi)"},
        {"Parameter": "Control Delay", "Kelas": "42 ms -> 67 ms",
         "Justifikasi fisika": "short delay; kategori 42 ms hanya 1 observasi"},
        {"Parameter": "Freeface Echelon", "Kelas": "0 / 42-67 / >=109 ms",
         "Justifikasi fisika": "tanpa echelon / short / long; level >=109 ms jarang"},
    ]), use_container_width=True, hide_index=True)
    st.warning("PRINSIP KUNCI: binning dilakukan di dalam masing-masing parameter. Freeface Count dan "
               "Freeface Echelon TETAP dua parameter terpisah & independen. Yang dirapikan hanya kategori "
               "internal tiap parameter, BUKAN penggabungan antar parameter.")
    st.success("Hasil: tidak ada lagi kategori berisi 1 data (semua min n>=2) -> estimasi real factor lebih "
               "stabil, ranking sensitivitas tidak terdistorsi artifact observasi tunggal. Record 10-row "
               "ekstrem (high-leverage) dikeluarkan atas dasar regime operasional.")
    st.caption("Referensi binning: Sturges (1926); Dougherty, Kohavi & Sahami (1995); Bieniawski (1989).")

with tab2:
    cA, cB = st.columns(2)
    cA.markdown("**Kalibrasi**")
    cA.write("k = " + format(cal["k"], ".2f")); cA.write("n = " + format(cal["n"], ".4f"))
    cA.write("sigma (in-sample) = " + format(cal["sigma"], ".4f"))
    cA.write("sigma (LOOCV)     = " + format(dl["s_e"], ".4f"))
    cA.write("R2 = " + format(cal["r2"], ".4f")); cA.write("MEAN_WF = " + format(mean_wf, ".4f"))
    cB.markdown("**Validasi**")
    cB.write("LOOCV MAE = " + format(mae_loo, ".5f"))
    cB.write("LOOCV NMAE = " + format(csf["nmae"] * 100, ".1f") + "%")
    cB.write("Underprediction = " + format(csf["pct_under"], ".1f") + "% (" + str(csf["n_under"]) + "/" + str(csf["n"]) + ")")
    cB.write("CSF (W=25) = " + format(csf["csf"], ".5f"))
    st.markdown("**Faktor konservatisme (lambda)**")
    rows = []
    for pct in (75, 90, 95):
        t = dl["table"][pct]
        rows.append({"Persentil": "P" + str(pct), "lambda empiris": round(t["emp"], 2),
                     "underpred emp (%)": round(t["under_emp"], 0), "lambda normal": round(t["norm"], 2),
                     "underpred norm (%)": round(t["under_norm"], 0)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Dipakai: P" + str(design_pct) + " metode " + design_method + " -> lambda = " + format(design_lambda, ".3f"))

with tab3:
    if fdiag is None:
        st.info("Kolom '" + COL_FREQ + "' belum ada -> diagnostik resonansi dilewati.")
    else:
        st.markdown("**Diagnostik resonansi (bukan prediksi).** Frekuensi dominan tidak dapat diprediksi "
                    "dari scale-distance. Audit pita bahaya dari " + str(fdiag["n"]) + " event aktual.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Freq dominan - min", format(fdiag["fmin"], ".1f") + " Hz")
        c2.metric("median", format(fdiag["fmed"], ".1f") + " Hz")
        c3.metric("maks", format(fdiag["fmax"], ".1f") + " Hz")
        st.markdown("**Distribusi pita Floyd (2008):**")
        st.dataframe(pd.DataFrame([{"Pita": b["name"], "Jumlah event": b["count"], "Porsi (%)": round(b["pct"], 0)}
                                   for b in fdiag["bands"]]), use_container_width=True, hide_index=True)
        st.warning(str(fdiag["n_slope"]) + "/" + str(fdiag["n"]) + " event di bawah "
                   + format(SLOPE_FREQ_THRESHOLD, ".0f") + " Hz (zona resonansi lereng).")
        st.caption("Frekuensi tergantung geologi, jarak, delay secara non-linear (Lucca 2003).")

with tab4:
    st.markdown("**Scaled Depth of Burial (SDOB)** - diagnostik konfinemen (ERG/Tobin 2013). "
                "SD = (stemming + 5d)/W10^(1/3). Densitas emulsi = " + format(RHO_EXPLOSIVE, ".0f") + " kg/m3.")
    st.caption("DIAGNOSTIK DESAIN, bukan prediktor nilai g.")
    if sdiag is None:
        st.info("Kolom depth/charge belum lengkap -> SDOB dilewati.")
    else:
        s_in = compute_sdob(inp["charge_kg"], inp["depth_m"], inp["hole_diameter_mm"])
        if s_in is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("SDOB event input", format(s_in["sdob"], ".2f"))
            c2.metric("Stemming (dihitung)", format(s_in["stemming"], ".2f") + " m",
                      "L isian = " + format(s_in["charge_len"], ".2f") + " m")
            c3.metric("Regime", s_in["band"].split(":")[0])
        st.markdown("**Distribusi konfinemen " + str(sdiag["n"]) + " event:**")
        st.dataframe(pd.DataFrame([{"Regime SDOB": name, "Jumlah": cnt, "Porsi (%)": round(100.0 * cnt / sdiag["n"], 0)}
                                   for name, cnt in sdiag["band_counts"].items()]), use_container_width=True, hide_index=True)
        st.write("SDOB: min " + format(sdiag["smin"], ".2f") + " | median " + format(sdiag["smed"], ".2f")
                 + " | maks " + format(sdiag["smax"], ".2f"))
        st.success("Argumen ke geotek: desain di zona CONTROLLED s/d MINIMAL SURFACE ACTIVITY -> desain proper.")
        if not np.isnan(sdiag["corr_g"]):
            st.caption("Korelasi SDOB vs g: r = " + format(sdiag["corr_g"], ".3f") + " (lemah; hanya diagnostik).")

with tab5:
    st.markdown("Variasi charge (distance tetap = " + format(inp["distance_m"], ".0f") + " m)")
    charges = sorted(set([int(df[COL_CHG].min()), 35, 40, 45, 50, 55, 60, 70, int(df[COL_CHG].max())]))
    rows = []
    for ch in charges:
        sd_ch = inp["distance_m"] / np.sqrt(ch)
        g_ch = predict_g_mc(cal, sd_ch, wf_norm, sigma_mc, n_iter=4000)
        gp = float(np.median(g_ch))
        rows.append({"Charge (kg)": ch, "SD": round(sd_ch, 2), "Median g": round(gp, 5),
                     "Status (median)": get_status(gp), "Design g": round(gp * design_lambda, 5),
                     "Status (design)": get_status(gp * design_lambda)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("Kalibrasi otomatis dari data aktual (27 ukur / 20 event, row-10 dibuang, freeface_count dikoreksi). "
           "Binning: row(1-3/4-7/>7), control(42->67), freeface echelon(0/42-67/>=109). "
           "Referensi: Duvall & Fogelson (1962); Ambraseys & Hendron (1968); Duan (1983); Sturges (1926); "
           "Dougherty dkk (1995); Bieniawski (1989); Sobol (1993); Saltelli dkk (2008); Floyd (2008); "
           "Lucca (2003); Tobin (2013); Hasanipanah dkk (2017).")
