# =====================================================================
# BLASTWAVE WEB - Prediksi Ground Vibration (g) Pit 3
# Aplikasi web (Streamlit) dari model Hybrid Monte Carlo + Weighted Factor
# + Residual Ratio + Design Value (P90) + Frekuensi-Resonansi.
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

# ---------------------------------------------------------------------
# DATABASE INTERNAL (bawaan) - 19 event aktual Pit 3.
# User tidak perlu upload; cukup isi parameter. Repo WAJIB Private.
# ---------------------------------------------------------------------
_COLS = ["Amaks (mm/s^s) Maks", "nilai_g", "charge_kg", "distance_m", "tie_up_type",
         "wall_echelon_ms", "freeface_echelon_ms", "controll_ms", "freeface_count",
         "geological_condt", "measuring_elevation", "row_number", "frekuensi_hz"]
_ROWS = [
    [258.11, 0.02631989517, 40, 270, "echelon", 109, 0, 67, 1, "normal", "higher", 3, 8.3],
    [1049, 0.1069682307, 30, 150, "echelon", 109, 0, 176, 1, "normal", "higher", 5, 9.8],
    [5198, 0.5300484875, 30, 80, "boxcut", 176, 176, 109, 1, "Fault", "higher", 10, 11.3],
    [226.13, 0.02305884272, 30, 300, "echelon", 176, 0, 109, 1, "normal", "higher", 4, 4.5],
    [387, 0.03946301744, 17, 180, "boxcut", 176, 176, 109, 2, "normal", "higher", 9, 49],
    [258.24, 0.02633315148, 50, 285, "echelon", 109, 0, 176, 2, "coal", "higher", 5, 6.4],
    [226.13, 0.02305884272, 55, 350, "echelon", 109, 0, 176, 2, "coal", "higher", 5, 8.3],
    [484, 0.04935426471, 57, 324, "echelon", 0, 109, 176, 2, "coal", "higher", 5, 7.9],
    [306.8, 0.03128489341, 60, 420, "echelon", 0, 67, 109, 1, "coal", "higher", 5, 8.1],
    [80.65, 0.008224011258, 60, 430, "boxcut", 42, 67, 109, 2, "coal", "higher", 4, 8.6],
    [161.42, 0.01646025911, 80, 500, "boxcut", 67, 42, 109, 0, "coal", "higher", 5, 3.8],
    [193.7, 0.01975190305, 90, 300, "boxcut", 67, 42, 109, 3, "coal", "normal", 6, 4.2],
    [1098, 0.1119648402, 82, 160, "boxcut", 67, 42, 109, 2, "coal", "lower", 9, 7.3],
    [936, 0.09544543753, 82, 160, "boxcut", 67, 42, 109, 0, "normal", "normal", 9, 6.3],
    [1195, 0.1218560875, 82, 160, "boxcut", 67, 42, 109, 0, "Fault", "higher", 9, 5.9],
    [96.85, 0.009875951523, 51, 454, "boxcut", 67, 42, 109, 2, "Fault", "normal", 7, 14.3],
    [323, 0.03293683368, 51, 320, "boxcut", 67, 42, 109, 2, "normal", "normal", 7, 4.9],
    [1130, 0.1152279321, 38, 176, "echelon", 42, 0, 67, 1, "Fault", "higher", 4, 8.8],
    [678, 0.06913675924, 38, 226, "echelon", 42, 0, 67, 1, "coal", "lower", 4, 12],
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
    st.title("Blastwave - Login")
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


def calibrate_frequency(df):
    if COL_FREQ not in df.columns:
        return None
    sub = df.dropna(subset=[COL_FREQ])
    sub = sub[sub[COL_FREQ] > 0]
    if len(sub) < 5:
        return None
    ln_f = np.log(sub[COL_FREQ].values)
    ln_sd = np.log(sub["scale_distance"].values)
    slope, intercept, r_val, _, _ = stats.linregress(ln_sd, ln_f)
    return dict(fk=float(np.exp(intercept)), fn=float(slope), r2=float(r_val ** 2),
                n=int(len(sub)), sub=sub)


def resonance_msgs(freq_hz, geo):
    msgs = []
    rock = GEO_TO_ROCK.get(geo)
    if rock:
        lo, hi = ROCK_NATURAL_FREQ[rock]
        if freq_hz <= hi:
            msgs.append(("warning", "RESONANSI: " + format(freq_hz, ".1f") + " Hz <= natural freq batuan '" + rock + "' (" + str(lo) + "-" + str(hi) + " Hz) -> potensi amplifikasi"))
        else:
            msgs.append(("ok", "OK vs batuan: " + format(freq_hz, ".1f") + " Hz di atas natural freq '" + rock + "' (" + str(lo) + "-" + str(hi) + " Hz)"))
    if freq_hz < SLOPE_FREQ_THRESHOLD:
        msgs.append(("warning", "SLOPE-RISK: " + format(freq_hz, ".1f") + " Hz < " + format(SLOPE_FREQ_THRESHOLD, ".0f") + " Hz (rentang paling berdampak ke lereng)"))
    else:
        msgs.append(("ok", "OK vs lereng: " + format(freq_hz, ".1f") + " Hz >= " + format(SLOPE_FREQ_THRESHOLD, ".0f") + " Hz"))
    return msgs


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
    fcal = calibrate_frequency(df)
    return df, cal, mean_wf, mae_loo, csf, dl, fcal


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Blastwave - Prediksi Getaran Pit 3", page_icon="boom", layout="wide")

if not check_password():
    st.stop()

st.title("Blastwave - Prediksi Ground Vibration (g) Pit 3")
st.caption("Hybrid Monte Carlo + Weighted Factor + Residual Ratio + Design Value (P90) + Frekuensi-Resonansi")

with st.sidebar:
    st.header("1. Data")
    st.caption("Aplikasi memakai database internal (bawaan). User cukup mengisi parameter di bawah.")
    with st.expander("Admin (opsional): perbarui database"):
        up = st.file_uploader("Upload Excel pengganti (format kolom sama)", type=["xlsx", "xls"])
    st.header("2. Pengaturan Model")
    method = st.selectbox("Metode residual ratio", ["geometric", "arithmetic"], index=0)
    g_target = st.number_input("Ambang aman g (G_TARGET)", value=0.030, step=0.005, format="%.3f")
    design_pct = st.selectbox("Design percentile", [90, 95, 75], index=0)
    design_method = st.selectbox("Metode Design Value", ["empirical", "normal"], index=0)
    use_loocv_sigma = st.checkbox("Pakai sigma LOOCV (lebih jujur)", value=True)
    n_iter = st.select_slider("Iterasi Monte Carlo", options=[2000, 5000, 10000, 20000], value=10000)

file_bytes = up.getvalue() if up is not None else None
try:
    df, cal, mean_wf, mae_loo, csf, dl, fcal = build_model(file_bytes, method)
except Exception as e:
    st.error("Gagal memproses data: " + str(e))
    st.stop()
st.caption("Sumber data: " + ("file upload (admin)" if up is not None else "database internal bawaan") +
           "  |  jumlah event: " + str(len(df)))

sigma_mc = dl["s_e"] if use_loocv_sigma else cal["sigma"]
design_lambda = dl["table"][design_pct]["emp" if design_method == "empirical" else "norm"]

# ---- Input parameter event ----
with st.sidebar:
    st.header("3. Parameter Event")
    inp = {}
    inp["distance_m"] = st.number_input("Distance (m)", value=float(np.median(df[COL_DIST].values)), min_value=1.0)
    inp["charge_kg"] = st.number_input("Charge per delay (kg)", value=float(np.median(df[COL_CHG].values)), min_value=1.0)
    for p in PARAMS_NUM:
        inp[p] = st.number_input(NICE[p], value=float(np.median(df[p].values)))
    for p in PARAMS_CAT:
        opts = sorted(str(x) for x in df[p].dropna().unique())
        inp[p] = st.selectbox(NICE[p], opts)

# ---- Prediksi ----
inp["scale_distance"] = inp["distance_m"] / np.sqrt(inp["charge_kg"])
wf_norm = compute_wf(inp, cal) / mean_wf
g_arr = predict_g_mc(cal, inp["scale_distance"], wf_norm, sigma_mc, int(n_iter))
g_med = float(np.median(g_arr))
g_q1, g_q3 = float(np.percentile(g_arr, 25)), float(np.percentile(g_arr, 75))
g_design = g_med * design_lambda

st.subheader("Hasil Prediksi")
c1, c2, c3 = st.columns(3)
c1.metric("Median g (harapan / mining)", format(g_med, ".5f"), get_status(g_med))
c2.metric("DESIGN VALUE P" + str(design_pct) + " (geotek)", format(g_design, ".5f"), get_status(g_design))
c3.metric("Faktor konservatisme", "x " + format(design_lambda, ".2f"), "sigma MC = " + format(sigma_mc, ".3f"))

st.markdown(
    "<div style='padding:10px;border-radius:8px;background:" + STATUS_COLOR[get_status(g_design)] +
    ";color:#000;font-weight:600'>Keputusan geotek pakai DESIGN VALUE = " + format(g_design, ".5f") +
    " (" + get_status(g_design) + "). Median (paling mungkin) = " + format(g_med, ".5f") +
    " (" + get_status(g_med) + ").</div>", unsafe_allow_html=True)

st.write("")
st.write("**Margin (IQR Q1-Q3):** " + format(g_q1, ".5f") + " ... " + format(g_q3, ".5f") +
         "  |  **WF normalized:** " + format(wf_norm, ".3f") +
         " (" + ("lebih berbahaya" if wf_norm > 1 else "lebih aman") + " dari rata-rata site)")

if inp["distance_m"] < 100 and str(inp["geological_condt"]) == "Fault":
    st.warning("REZIM Fault + near-field (<100 m): model cenderung MEREMEHKAN secara struktural -> beri buffer ekstra di luar Design Value.")

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
tab1, tab2, tab3, tab4 = st.tabs(["Faktor (Sobol + Heatmap)", "Validasi & CSF", "Frekuensi-Resonansi", "Sensitivity Charge"])

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

with tab3:
    if fcal is None:
        st.info("Kolom '" + COL_FREQ + "' belum ada/cukup di Excel -> analisis resonansi dilewati. "
                "Tambahkan kolom frekuensi dominan (Hz) untuk mengaktifkan.")
    else:
        f_pred = fcal["fk"] * inp["scale_distance"] ** (fcal["fn"])
        st.write("Kalibrasi freq vs SD: R2 = " + format(fcal["r2"], ".3f") + " (n=" + str(fcal["n"]) + ")")
        if fcal["r2"] < 0.3:
            st.caption("R2 rendah -> frekuensi lemah diprediksi dari SD (konsisten literatur). Pakai sebagai INDIKASI.")
        st.metric("Prediksi frekuensi dominan event ini", format(f_pred, ".1f") + " Hz")
        for kind, msg in resonance_msgs(f_pred, str(inp["geological_condt"])):
            (st.warning if kind == "warning" else st.success)(msg)
        sub = fcal["sub"]
        n_slope = int((sub[COL_FREQ] < SLOPE_FREQ_THRESHOLD).sum())
        st.write("Data aktual: " + str(n_slope) + "/" + str(fcal["n"]) + " event punya freq < " + format(SLOPE_FREQ_THRESHOLD, ".0f") + " Hz (zona sensitif lereng)")
        st.caption("Catatan: pemetaan geologi->kelas batuan & natural freq (Floyd 2008) WAJIB diverifikasi dgn uji site.")

with tab4:
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
