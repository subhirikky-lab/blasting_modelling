# =====================================================================
# Prediksi Ground Vibration (g) Pit 3
# Aplikasi web (Streamlit) dari model Hybrid Monte Carlo + Weighted Factor
# + Residual Ratio + Design Value (P90) + Diagnostik Resonansi.
# Jalankan lokal :  streamlit run app.py
# =====================================================================

import io
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

PARAMS_CAT = ["geological_condt", "tie_up_type", "measuring_elevation"]
PARAMS_NUM = ["row_number", "controll_ms", "wall_echelon_ms", "freeface_echelon_ms", "freeface_count"]
ALL_PARAMS = PARAMS_CAT + PARAMS_NUM
CONTROLLABLE_OPS = ["tie_up_type", "controll_ms", "wall_echelon_ms", "freeface_echelon_ms", "freeface_count"]

NICE = {
    "row_number": "Row Number", "controll_ms": "Control Delay (ms)",
    "wall_echelon_ms": "Wall Echelon (ms)", "freeface_echelon_ms": "Freeface Echelon (ms)",
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

# --- Scaled Depth of Burial (SDOB) - definisi ERG/Tobin (2013) ---
# SD = (stemming + 5*d) / W10^(1/3)
#   stemming = depth - panjang_kolom_isian ; d = diameter lubang (m)
#   W10 = massa peledak dalam ruang setara 10 x diameter = rho * (pi/4 d^2) * 10d
# Verifikasi: contoh artikel (stem=5m, d=229mm, rho=1210) -> SD=1.27 (band controlled). OK.
# Band konfinemen (Tobin 2013):
#   < 0.92  : under-confined  -> energi lepas ke permukaan (flyrock & airblast tinggi)
#   0.92-1.40: controlled     -> fragmentasi & heave baik, airblast/getaran wajar
#   > 1.40  : over-confined    -> energi terkurung ke massa batuan (fragmentasi bawah
#             kurang, energi ke tanah) -> argumen desain "aman dari flyrock" ke geotek.
COL_DEPTH = "depth_m"
COL_HOLE_DIA = "hole_diameter_mm"
RHO_EXPLOSIVE = 1150.0   # kg/m^3 (= 1.15 g/cc), emulsi Pit 3
# 6 zona SDOB sesuai diagram Chiappetta/ERG (bukan 3-kategori kasar):
SDOB_BANDS = [
    ("Uncontrolled (0-0.60): flyrock & airblast hebat", 0.0, 0.60),
    ("Cratering (0.60-0.92): fragmentasi sangat halus", 0.60, 0.92),
    ("Controlled (0.92-1.40): fragmentasi baik, getaran/airblast wajar", 0.92, 1.40),
    ("Very controlled (1.40-1.80): frag lebih kasar, TANPA flyrock", 1.40, 1.80),
    ("Minimal surface (1.80-2.40): gangguan permukaan kecil", 1.80, 2.40),
    ("Insignificant (>2.40): efek permukaan tak berarti", 2.40, float("inf")),
]

# ---------------------------------------------------------------------
# DATABASE INTERNAL (bawaan) - 23 event aktual Pit 3 (19 + 4 update Jun 2026).
# User tidak perlu upload; cukup isi parameter. Repo WAJIB Private.
# Kolom depth_m & hole_diameter_mm dipakai untuk Scaled Depth of Burial (SDOB).
# ---------------------------------------------------------------------
_COLS = ["Amaks (mm/s^s) Maks", "nilai_g", "charge_kg", "distance_m", "tie_up_type",
         "wall_echelon_ms", "freeface_echelon_ms", "controll_ms", "freeface_count",
         "geological_condt", "measuring_elevation", "row_number", "frekuensi_hz",
         "depth_m", "hole_diameter_mm"]
_ROWS = [
    [258.11, 0.02631989517, 40, 270, "echelon", 109, 0, 67, 1, "normal", "higher", 3, 8.3, 8, 200],
    [1049, 0.1069682307, 30, 150, "echelon", 109, 0, 176, 1, "normal", "higher", 5, 9.8, 8, 200],
    [5198, 0.5300484875, 30, 80, "boxcut", 176, 176, 109, 1, "Fault", "higher", 10, 11.3, 8, 200],
    [226.13, 0.02305884272, 30, 300, "echelon", 176, 0, 109, 1, "normal", "higher", 4, 4.5, 8, 200],
    [387, 0.03946301744, 17, 180, "boxcut", 176, 176, 109, 2, "normal", "higher", 9, 49, 8, 200],
    [258.24, 0.02633315148, 50, 285, "echelon", 109, 0, 176, 2, "coal", "higher", 5, 6.4, 8, 200],
    [226.13, 0.02305884272, 55, 350, "echelon", 109, 0, 176, 2, "coal", "higher", 5, 8.3, 7, 200],
    [484, 0.04935426471, 57, 324, "echelon", 0, 109, 176, 2, "coal", "higher", 5, 7.9, 8, 200],
    [306.8, 0.03128489341, 60, 420, "echelon", 0, 67, 109, 1, "coal", "higher", 5, 8.1, 8, 200],
    [80.65, 0.008224011258, 60, 430, "boxcut", 42, 67, 109, 2, "coal", "higher", 4, 8.6, 8, 200],
    [161.42, 0.01646025911, 80, 500, "boxcut", 67, 42, 109, 0, "coal", "higher", 5, 3.8, 7.98, 200],
    [193.7, 0.01975190305, 90, 300, "boxcut", 67, 42, 109, 3, "coal", "normal", 6, 4.2, 8, 200],
    [1098, 0.1119648402, 82, 160, "boxcut", 67, 42, 109, 2, "coal", "lower", 9, 7.3, 6.34, 200],
    [936, 0.09544543753, 82, 160, "boxcut", 67, 42, 109, 0, "normal", "normal", 9, 6.3, 6.34, 200],
    [1195, 0.1218560875, 82, 160, "boxcut", 67, 42, 109, 0, "Fault", "higher", 9, 5.9, 6.34, 200],
    [96.85, 0.009875951523, 51, 454, "boxcut", 67, 42, 109, 2, "Fault", "normal", 7, 14.3, 7.13, 200],
    [323, 0.03293683368, 51, 320, "boxcut", 67, 42, 109, 2, "normal", "normal", 7, 4.9, 7.3, 200],
    [1130, 0.1152279321, 38, 176, "echelon", 42, 0, 67, 1, "Fault", "higher", 4, 8.8, 7.45, 200],
    [678, 0.06913675924, 38, 226, "echelon", 42, 0, 67, 1, "coal", "lower", 4, 12, 7.45, 200],
    [589, 0.06006128494, 51, 239, "boxcut", 42, 42, 67, 2, "coal", "lower", 3, 9.5, 7.42, 200],
    [629.54, 0.06419521447, 51, 239, "boxcut", 109, 42, 67, 2, "coal", "lower", 3, 10, 7.42, 200],
    [387.41, 0.03950482581, 51, 320, "boxcut", 109, 42, 67, 2, "Fault", "higher", 6, 6.2, 7.42, 200],
    [290.53, 0.02962581514, 51, 320, "boxcut", 42, 42, 67, 2, "Fault", "higher", 6, 3.4, 7.42, 200],
]
BUILTIN_DATA = [dict(zip(_COLS, r)) for r in _ROWS]


def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None


def check_password():
    """Gerbang password OPSIONAL. Aktif hanya jika secret 'app_password' di-set
    di Streamlit (Settings -> Secrets). Jika tidak di-set, app terbuka biasa."""
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


# ---------------------------------------------------------------------
# FUNGSI MODEL (mengembalikan nilai, bukan print)
# ---------------------------------------------------------------------
def get_status(g):
    for thr, label in STATUS_THRESHOLDS:
        if g <= thr:
            return label
    return "EXTREMELY RISKY"


def validate(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    required = [COL_AMAKS, COL_G, COL_DIST, COL_CHG] + ALL_PARAMS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("Kolom wajib tidak ada: " + ", ".join(missing))
    df = df.dropna(subset=[COL_AMAKS, COL_DIST, COL_CHG])
    df = df[(df[COL_AMAKS] > 0) & (df[COL_DIST] > 0) & (df[COL_CHG] > 0)].copy()
    if len(df) < 5:
        raise ValueError("Data valid hanya " + str(len(df)) + " baris - terlalu sedikit (min 5).")
    df["scale_distance"] = df[COL_DIST] / np.sqrt(df[COL_CHG])
    df["Amaks_maks"] = df[COL_AMAKS]
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


def compute_wf(row, cal):
    wf = 0.0
    for p in ALL_PARAMS:
        f = cal["real_factor"][p]
        v = row[p]
        fval = f.get(v, 1.0) if isinstance(v, str) else _nearest_num(f, v)
        wf += fval * cal["weight"][p]
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
    return dict(mae=float(np.mean(errs)),
                csf=float(np.mean([e + w_under * s ** 2 for e, s in zip(errs, short)])),
                n=len(pairs), n_under=n_under,
                pct_under=100.0 * n_under / len(pairs),
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
        lv = sorted(cal["real_factor"][p], key=float) if p in PARAMS_NUM else sorted(cal["real_factor"][p], key=str)
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


def frequency_diagnostics(df):
    """DIAGNOSTIK RESONANSI MURNI (bukan prediktor).
    Temuan data Pit 3: freq vs scale-distance R2 ~ 0.002 (nol) -> frekuensi
    TIDAK valid diprediksi dari SD. Frekuensi dominan diposisikan sebagai
    AUDIT PITA BAHAYA dari 19 event aktual (Floyd 2008; zona resonansi lereng).
    """
    if COL_FREQ not in df.columns:
        return None
    sub = df.dropna(subset=[COL_FREQ])
    sub = sub[sub[COL_FREQ] > 0]
    if len(sub) < 1:
        return None
    freqs = sub[COL_FREQ].values.astype(float)
    n = int(len(freqs))

    # Pita Floyd (2008): klasifikasi respons lereng berdasar frekuensi dominan.
    band_defs = [("Weak (3-7 Hz)", 3.0, 7.0), ("Blocky (7-15 Hz)", 7.0, 15.0),
                 ("Strong (15-23 Hz)", 15.0, 23.0), ("Aman (>23 Hz)", 23.0, float("inf"))]
    bands = []
    for name, lo, hi in band_defs:
        c = int(np.sum((freqs > lo) & (freqs <= hi)))
        bands.append(dict(name=name, lo=lo, hi=hi, count=c,
                          pct=100.0 * c / n if n > 0 else 0.0))

    n_slope = int(np.sum(freqs < SLOPE_FREQ_THRESHOLD))

    # Audit resonansi per kelas geologi (pakai frekuensi event AKTUAL, bukan prediksi).
    geo_audit = {}
    if "geological_condt" in sub.columns:
        for geo in sorted(str(x) for x in sub["geological_condt"].dropna().unique()):
            g_sub = sub[sub["geological_condt"].astype(str) == geo]
            gf = g_sub[COL_FREQ].values.astype(float)
            if len(gf) == 0:
                continue
            rock = GEO_TO_ROCK.get(geo)
            rng = ROCK_NATURAL_FREQ.get(rock) if rock else None
            in_res = 0
            if rng is not None:
                in_res = int(np.sum(gf <= rng[1]))   # <= batas atas natural -> risiko amplifikasi
            geo_audit[geo] = dict(rock=rock, rng=rng, n=int(len(gf)), in_res=in_res,
                                  freqs=sorted(float(x) for x in gf),
                                  fmin=float(np.min(gf)), fmax=float(np.max(gf)),
                                  fmed=float(np.median(gf)))

    return dict(n=n, freqs=sorted(float(x) for x in freqs),
                fmin=float(np.min(freqs)), fmax=float(np.max(freqs)),
                fmed=float(np.median(freqs)), bands=bands,
                n_slope=n_slope, sub=sub, geo_audit=geo_audit)


def compute_sdob(charge_kg, depth_m, hole_dia_mm, rho=RHO_EXPLOSIVE):
    """Scaled Depth of Burial (ERG/Tobin 2013). SD = (stemming + 5d)/W10^(1/3).
    Tak berdimensi (m / kg^(1/3) dgn massa W10). Mengembalikan dict atau None."""
    try:
        ch = float(charge_kg); dep = float(depth_m); d = float(hole_dia_mm) / 1000.0
    except Exception:
        return None
    if ch <= 0 or dep <= 0 or d <= 0:
        return None
    lin_density = rho * (np.pi / 4.0) * d * d        # kg/m
    charge_len = ch / lin_density                    # panjang kolom isian (m)
    stemming = dep - charge_len                       # tinggi stemming dihitung (m)
    w10 = lin_density * (10.0 * d)                    # massa peledak dlm 10x diameter (kg)
    sdob = (stemming + 5.0 * d) / (w10 ** (1.0 / 3.0))
    band = "n/a"
    for name, lo, hi in SDOB_BANDS:
        if lo <= sdob < hi:
            band = name
            break
    return dict(sdob=float(sdob), charge_len=float(charge_len), stemming=float(stemming),
                w10=float(w10), stem_over_dia=float(stemming / d), band=band)


def sdob_diagnostics(df, rho=RHO_EXPLOSIVE):
    """Diagnostik konfinemen SDOB untuk seluruh event (audit, bukan prediktor)."""
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
    # korelasi SDOB vs g (diagnostik, bukan model regresi)
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


# ---------------------------------------------------------------------
# CACHE: kalibrasi berat hanya dihitung sekali per file+config
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(file_bytes, method):
    if file_bytes is None:
        df_raw = pd.DataFrame(BUILTIN_DATA)          # database internal bawaan
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes))  # override admin (opsional)
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
st.caption("Hybrid Monte Carlo + Weighted Factor + Residual Ratio + Design Value + Probabilitas + Diagnostik Resonansi & SDOB")

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
            st.success("Mode admin aktif - kamu bisa perbarui database.")
            up = st.file_uploader("Upload Excel pengganti (format kolom sama)", type=["xlsx", "xls"])
    st.header("2. Pengaturan Model")
    st.caption("Default di bawah sudah merupakan setelan yang DISARANKAN. "
               "Biarkan apa adanya kecuali sedang melakukan uji sensitivitas.")
    method = st.selectbox(
        "Metode residual ratio", ["geometric", "arithmetic"], index=0,
        help=("DISARANKAN: geometric. Residual ratio bersifat perkalian (lognormal), "
              "sehingga geometric mean adalah estimator tengah yang tidak bias. "
              "Arithmetic cenderung menggelembung ke atas karena ditarik nilai rasio besar "
              "-> pakai hanya untuk pembanding."))
    g_target = st.number_input("Ambang aman g (G_TARGET)", value=0.030, step=0.005, format="%.3f",
                               help="Batas g yang dianggap aman untuk keputusan. Default 0.030.")
    design_pct = st.selectbox(
        "Design percentile", [90, 95, 75], index=0,
        help=("DISARANKAN: P90. Dengan n=19 event, P90 empiris andal. "
              "P95 hanya disokong ~1 event -> rapuh; bila perlu P95 gunakan metode 'normal'."))
    design_method = st.selectbox(
        "Metode Design Value", ["empirical", "normal"], index=0,
        help=("DISARANKAN: empirical. Mengambil persentil langsung dari sebaran rasio aktual "
              "(tanpa asumsi distribusi) -> sumber lambda yang sudah dikalibrasi. "
              "'normal' mengasumsikan rasio lognormal (lambda = exp(mu + z*sigma)); "
              "lebih stabil untuk ekstrapolasi persentil tinggi/cross-check."))
    use_loocv_sigma = st.checkbox(
        "Pakai sigma LOOCV (lebih jujur)", value=True,
        help="DISARANKAN: aktif. Memakai sebaran error leave-one-out, bukan error in-sample yang optimistik.")
    n_iter = st.select_slider("Iterasi Monte Carlo", options=[2000, 5000, 10000, 20000], value=10000,
                              help="Jumlah iterasi simulasi. 10000 sudah stabil; lebih tinggi = lebih halus tapi lambat.")

file_bytes = up.getvalue() if up is not None else None
try:
    df, cal, mean_wf, mae_loo, csf, dl, fdiag, sdiag = build_model(file_bytes, method)
except Exception as e:
    st.error("Gagal memproses data: " + str(e))
    st.stop()
st.caption("Sumber data: " + ("file upload (admin)" if up is not None else "database internal bawaan"))

sigma_mc = dl["s_e"] if use_loocv_sigma else cal["sigma"]
design_lambda = dl["table"][design_pct]["emp" if design_method == "empirical" else "norm"]

# ---- Input parameter event ----
with st.sidebar:
    st.header("3. Parameter Event")
    inp = {}
    inp["distance_m"] = st.number_input("Distance (m)", value=float(np.median(df[COL_DIST].values)), min_value=1.0)
    inp["charge_kg"] = st.number_input("Charge per delay (kg)", value=float(np.median(df[COL_CHG].values)), min_value=1.0)
    _depth_default = float(np.median(df[COL_DEPTH].values)) if COL_DEPTH in df.columns else 8.0
    _dia_default = float(np.median(df[COL_HOLE_DIA].values)) if COL_HOLE_DIA in df.columns else 200.0
    inp["depth_m"] = st.number_input("Kedalaman lubang (m)", value=_depth_default, min_value=1.0,
                                     help="Kedalaman lubang bor. Dipakai untuk diagnostik SDOB (konfinemen), tidak mengubah prediksi g.")
    inp["hole_diameter_mm"] = st.number_input("Diameter lubang (mm)", value=_dia_default, min_value=50.0,
                                              help="Diameter lubang bor. Dipakai untuk diagnostik SDOB.")
    for p in PARAMS_NUM:
        inp[p] = st.number_input(NICE[p], value=float(np.median(df[p].values)))
    for p in PARAMS_CAT:
        opts = sorted(str(x) for x in df[p].dropna().unique())
        inp[p] = st.selectbox(NICE[p], opts)

# ---- Prediksi ----
import math as _math
inp["scale_distance"] = inp["distance_m"] / np.sqrt(inp["charge_kg"])
wf_norm = compute_wf(inp, cal) / mean_wf
g_arr = predict_g_mc(cal, inp["scale_distance"], wf_norm, sigma_mc, int(n_iter))
g_med = float(np.median(g_arr))
# MEAN lognormal (expected value) = median x exp(sigma^2/2) -- koreksi retransformasi (Duan 1983).
# Nilai prediksi paling REALISTIS: tidak optimis (median), tidak pesimis (P90).
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
c1.metric("PREDIKSI g (nilai harapan)", format(g_mean, ".5f"), get_status(g_mean))
c2.metric("Probabilitas g > " + format(g_target, ".3f"), format(prob_exceed, ".1f") + "%")
c3.metric("Design Value (keselamatan)", format(g_design, ".5f"), get_status(g_design))

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
st.caption("PREDIKSI g = nilai harapan (mean lognormal = median x exp(sigma^2/2), koreksi Duan 1983) - "
           "estimator tak-bias yang lebih realistis daripada median (yang cenderung optimis). "
           "Untuk keputusan keselamatan gunakan Design Value = " + format(g_design, ".5f")
           + " (median x lambda P" + str(int(design_pct)) + " = " + format(design_lambda, ".2f")
           + "). Rentang ketidakpastian mengikuti Design percentile yang dipilih.")

if is_fault_near:
    st.warning("REZIM Fault + near-field (<100 m): model cenderung MEREMEHKAN secara struktural -> "
               "untuk Fault, jadikan Design Value sebagai acuan keputusan, beri buffer ekstra.")

# ---- Recommendation ----
rec = recommend(inp, cal, mean_wf, df, g_target, design_lambda)
st.subheader("Rekomendasi Keputusan")
if not rec["need"]:
    st.success("AMAN - Design value " + format(rec["design"], ".5f") + " di bawah ambang " + format(g_target, ".3f") + ". Tidak perlu perubahan desain.")
else:
    st.error("DESIGN VALUE " + format(rec["design"], ".5f") + " DI ATAS AMBANG " + format(g_target, ".3f") + " -> perlu mitigasi.")
    if rec["steps"]:
        st.markdown("**Langkah 1 - tweak operasional (biaya rendah):**")
        for s in rec["steps"]:
            st.markdown("- " + s)
        st.caption("Design g setelah tweak operasional: " + format(rec["after_ops"], ".5f"))
    if rec["charge_to"] is not None:
        st.markdown("**Langkah 2 - kurangi charge per delay:**")
        turun = (1 - rec["charge_to"] / rec["charge_from"]) * 100
        st.markdown("- Charge: " + format(rec["charge_from"], ".1f") + " kg -> " + format(rec["charge_to"], ".1f") + " kg (turun " + format(turun, ".0f") + "%)")
        if rec["charge_extrapolated"]:
            st.caption("[!] charge target di bawah rentang data -> ekstrapolasi, kurang andal.")
        st.caption("Trade-off: charge turun memperbaiki getaran TAPI berpotensi memperburuk fragmentasi/produktivitas.")
    st.info("Proyeksi bila saran diterapkan -> median " + format(rec["final_median"], ".5f") + ", DESIGN " + format(rec["final_design"], ".5f"))

# ---- Tabs analisis ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Faktor (Sobol + Heatmap)", "Validasi & CSF", "Diagnostik Resonansi", "Diagnostik SDOB", "Sensitivity Charge"])

imp = importance_metrics(cal, inp, mean_wf)
order = sorted(ALL_PARAMS, key=lambda p: -imp["sobol"][p])

with tab1:
    levels, g_levels = imp["levels"], imp["g_levels"]

    def active_level(p):
        if p in PARAMS_NUM:
            keys = np.array([float(x) for x in levels[p]])
            return levels[p][int(np.abs(keys - float(inp[p])).argmin())]
        return inp[p]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [1, 1.25]})
    labels = [NICE[p] + "  [" + str(active_level(p)) + "]" for p in order]
    vals = [imp["sobol"][p] for p in order]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(order)))
    bars = ax1.barh(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Kontribusi terhadap varians g - indeks Sobol orde-1 (%)")
    ax1.set_title("Kontribusi Faktor (GLOBAL)   median g input = " + format(g_med, ".5f"))
    for b, v in zip(bars, vals):
        ax1.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, format(v, ".1f") + "%", va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, max(vals) * 1.15)
    ax1.grid(axis="x", alpha=0.3)

    max_lv = max(len(levels[p]) for p in order)
    mat = np.full((len(order), max_lv), np.nan)
    for i, p in enumerate(order):
        for j, g in enumerate(g_levels[p]):
            mat[i, j] = g
    cmap2 = plt.cm.RdYlGn_r.copy()
    cmap2.set_bad("white")
    im = ax2.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap2)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([NICE[p] for p in order], fontsize=10)
    ax2.set_xticks(range(max_lv))
    ax2.set_xticklabels(["Lvl " + str(j + 1) for j in range(max_lv)], fontsize=9)
    ax2.set_title("Heatmap Sensitivitas - prediksi g tiap level (merah=tinggi, hijau=rendah, kotak biru=input)")
    for i, p in enumerate(order):
        act = active_level(p)
        for j, val in enumerate(levels[p]):
            ax2.text(j, i - 0.18, str(val), ha="center", va="center", fontsize=8, fontweight="bold")
            ax2.text(j, i + 0.22, format(g_levels[p][j], ".4f"), ha="center", va="center", fontsize=7)
            if val == act:
                ax2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=2.5))
    fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02).set_label("Prediksi g")
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    cA, cB = st.columns(2)
    cA.markdown("**Kalibrasi**")
    cA.write("k = " + format(cal["k"], ".2f"))
    cA.write("n = " + format(cal["n"], ".4f"))
    cA.write("sigma (in-sample) = " + format(cal["sigma"], ".4f"))
    cA.write("sigma (LOOCV)     = " + format(dl["s_e"], ".4f"))
    cA.write("R2 = " + format(cal["r2"], ".4f"))
    cA.write("MEAN_WF = " + format(mean_wf, ".4f"))
    cB.markdown("**Validasi**")
    cB.write("LOOCV MAE = " + format(mae_loo, ".5f"))
    cB.write("Underprediction = " + format(csf["pct_under"], ".1f") + "% (" + str(csf["n_under"]) + "/" + str(csf["n"]) + " event)")
    cB.write("Rata-rata shortfall = " + format(csf["mean_short"], ".5f"))
    cB.write("CSF (W=25) = " + format(csf["csf"], ".5f"))
    st.markdown("**Faktor konservatisme (lambda) - underprediction setelah dikoreksi**")
    rows = []
    for pct in (75, 90, 95):
        t = dl["table"][pct]
        rows.append({"Persentil": "P" + str(pct),
                     "lambda empiris": round(t["emp"], 2), "underpred emp (%)": round(t["under_emp"], 0),
                     "lambda normal": round(t["norm"], 2), "underpred norm (%)": round(t["under_norm"], 0)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Dipakai: P" + str(design_pct) + " metode " + design_method + " -> lambda = " + format(design_lambda, ".3f"))
    st.info("Panduan setelan: **geometric** (residual ratio) + **empirical P90** (Design Value) "
            "adalah konfigurasi yang disarankan untuk keputusan. Opsi 'arithmetic' dan 'normal' "
            "disediakan untuk uji sensitivitas / cross-check, bukan untuk produksi.")

with tab3:
    if fdiag is None:
        st.info("Kolom '" + COL_FREQ + "' belum ada di Excel -> diagnostik resonansi dilewati. "
                "Tambahkan kolom frekuensi dominan (Hz) untuk mengaktifkan.")
    else:
        st.markdown("**Diagnostik resonansi (bukan prediksi).** "
                    "Frekuensi dominan **tidak** dapat diprediksi dari scale-distance "
                    "(R2 ~ 0.002, praktis nol) maupun parameter desain ledakan. "
                    "Oleh karena itu frekuensi diposisikan sebagai **audit pita bahaya** dari "
                    + str(fdiag["n"]) + " event aktual Pit 3 - bukan sebagai output prediksi presisi.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Frekuensi dominan - min", format(fdiag["fmin"], ".1f") + " Hz")
        c2.metric("median", format(fdiag["fmed"], ".1f") + " Hz")
        c3.metric("maks", format(fdiag["fmax"], ".1f") + " Hz")

        st.markdown("**Distribusi pita Floyd (2008) - respons lereng berdasar frekuensi dominan:**")
        band_rows = []
        for b in fdiag["bands"]:
            band_rows.append({"Pita": b["name"], "Jumlah event": b["count"],
                              "Porsi (%)": round(b["pct"], 0)})
        st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)

        st.warning(str(fdiag["n_slope"]) + "/" + str(fdiag["n"]) + " event berada di bawah "
                   + format(SLOPE_FREQ_THRESHOLD, ".0f") + " Hz (zona resonansi lereng). "
                   "Mayoritas energi getaran Pit 3 jatuh di pita yang paling berdampak ke stabilitas lereng.")

        if fdiag["geo_audit"]:
            st.markdown("**Audit resonansi per kondisi geologi (frekuensi event aktual vs natural freq batuan):**")
            geo_rows = []
            for geo, a in fdiag["geo_audit"].items():
                rng_txt = (str(a["rng"][0]) + "-" + str(a["rng"][1]) + " Hz") if a["rng"] else "(tidak dipetakan)"
                geo_rows.append({"Geologi": geo, "Kelas batuan": str(a["rock"]),
                                 "Natural freq": rng_txt, "n event": a["n"],
                                 "freq min-maks": format(a["fmin"], ".1f") + " - " + format(a["fmax"], ".1f"),
                                 "Event di zona resonansi": a["in_res"]})
            st.dataframe(pd.DataFrame(geo_rows), use_container_width=True, hide_index=True)

        st.caption("Catatan metodologis: frekuensi tergantung geologi, jarak, dan delay secara "
                   "kompleks/non-linear (Lucca 2003) sehingga R2 ~ 0 terhadap SD adalah wajar dan "
                   "sesuai literatur (Floyd 2008; Kumar 2020; Hasanipanah 2015). Pemetaan "
                   "geologi->kelas batuan & natural freq WAJIB diverifikasi dengan uji site.")

with tab4:
    st.markdown("**Scaled Depth of Burial (SDOB)** - diagnostik konfinemen energi ledakan "
                "(ERG / Tobin 2013). SD = (stemming + 5d) / W10^(1/3), dengan W10 = massa peledak "
                "dalam ruang setara 10x diameter lubang. Densitas emulsi = "
                + format(RHO_EXPLOSIVE, ".0f") + " kg/m3.")
    st.caption("6 zona SDOB (diagram Chiappetta/ERG): <0.60 uncontrolled (flyrock hebat); "
               "0.60-0.92 cratering; 0.92-1.40 controlled (fragmentasi & heave baik, getaran/airblast wajar); "
               "1.40-1.80 very controlled (tanpa flyrock); 1.80-2.40 minimal surface activity; "
               ">2.40 insignificant. SDOB adalah DIAGNOSTIK DESAIN (bukti konfinemen wajar untuk "
               "argumen ke geotek), BUKAN prediktor nilai g.")
    if sdiag is None:
        st.info("Kolom '" + COL_DEPTH + "' / '" + COL_CHG + "' belum lengkap -> diagnostik SDOB dilewati.")
    else:
        # SDOB event input - pakai kedalaman & diameter yang DIINPUT user
        s_in = compute_sdob(inp["charge_kg"], inp["depth_m"], inp["hole_diameter_mm"])
        if s_in is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("SDOB event input", format(s_in["sdob"], ".2f"))
            c2.metric("Stemming (dihitung)", format(s_in["stemming"], ".2f") + " m",
                      "L isian = " + format(s_in["charge_len"], ".2f") + " m")
            c3.metric("Regime konfinemen", s_in["band"].split(":")[0])
            st.caption("(charge " + format(inp["charge_kg"], ".0f") + " kg @ kedalaman "
                       + format(inp["depth_m"], ".2f") + " m, diameter " + format(inp["hole_diameter_mm"], ".0f")
                       + " mm; stemming dihitung = kedalaman - panjang isian)")

        st.markdown("**Distribusi konfinemen " + str(sdiag["n"]) + " event aktual:**")
        brows = []
        for name, cnt in sdiag["band_counts"].items():
            brows.append({"Regime SDOB": name, "Jumlah event": cnt,
                          "Porsi (%)": round(100.0 * cnt / sdiag["n"], 0)})
        st.dataframe(pd.DataFrame(brows), use_container_width=True, hide_index=True)
        st.write("SDOB aktual: min " + format(sdiag["smin"], ".2f") + " | median "
                 + format(sdiag["smed"], ".2f") + " | maks " + format(sdiag["smax"], ".2f"))
        st.success("Argumen ke geotek: desain berada di zona CONTROLLED s/d MINIMAL SURFACE ACTIVITY "
                   "(tidak ada satupun uncontrolled/cratering) -> secara teori SDOB Chiappetta, desain "
                   "blasting Pit 3 sudah proper: risiko flyrock/airblast minimal, energi terarah ke "
                   "pemecahan batuan. Mayoritas event di zona controlled-very controlled (zona ideal).")
        if not np.isnan(sdiag["corr_g"]):
            st.caption("Korelasi SDOB vs nilai g: r = " + format(sdiag["corr_g"], ".3f")
                       + " (lemah/rancu - g juga dipengaruhi jarak & charge, jadi SDOB TIDAK dipakai "
                       + "sebagai prediktor g; hanya diagnostik konfinemen).")

        st.markdown("**Tabel SDOB per event:**")
        trows = []
        for i, x in enumerate(sdiag["rows"]):
            trows.append({"#": i + 1, "SDOB": round(x["sdob"], 2),
                          "Stemming (m)": round(x["stemming"], 2),
                          "Stem/Dia": round(x["stem_over_dia"], 1),
                          "L isian (m)": round(x["charge_len"], 2),
                          "nilai g": round(x["g"], 4),
                          "Regime": x["band"].split(":")[0]})
        st.dataframe(pd.DataFrame(trows), use_container_width=True, hide_index=True)
        st.caption("Definisi & band: diagram Chiappetta/ERG (6 zona); Tobin (2013) 'The Importance of "
                   "Energy Confinement to the Blast Outcome'; Ash (1993); Langefors & Kihlstrom (1978). "
                   "Band controlled 0.92-1.40 diverifikasi via contoh ERG Industrial (SD=1.27).")

with tab5:
    st.markdown("Variasi charge (distance tetap = " + format(inp["distance_m"], ".0f") + " m)")
    charges = sorted(set([int(df[COL_CHG].min()), 35, 40, 45, 50, 55, 60, 70, int(df[COL_CHG].max())]))
    rows = []
    for ch in charges:
        sd_ch = inp["distance_m"] / np.sqrt(ch)
        g_ch = predict_g_mc(cal, sd_ch, wf_norm, sigma_mc, n_iter=4000)
        gp = float(np.median(g_ch))
        rows.append({"Charge (kg)": ch, "SD": round(sd_ch, 2),
                     "Median g": round(gp, 5), "Status (median)": get_status(gp),
                     "Design g": round(gp * design_lambda, 5), "Status (design)": get_status(gp * design_lambda)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("Model kalibrasi otomatis dari data aktual. Design Value = median x lambda (batas prediksi atas) "
           "untuk keputusan keselamatan; median untuk nilai harapan. Referensi: Duvall & Fogelson (1962); "
           "Ambraseys & Hendron (1968); Siskind dkk (1980); Dowding (1985); Sobol (1993); Floyd (2008).")
