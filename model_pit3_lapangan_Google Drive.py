import math
import numpy as np
import pandas as pd
from scipy import stats
import requests
import io
import streamlit as st
import matplotlib.pyplot as plt

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Prediksi Getaran Pit 3", layout="wide")
st.title("Aplikasi Prediksi Getaran Blasting - PIT 3")

# =====================================================================
# [A] KONFIGURASI 
# =====================================================================
FILE_ID = '1xwJz5xrysYvEPK7lZMweL5rAAbBFkNA1'
DOWNLOAD_URL = f'https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=xlsx'
SHEET_NAME = 'Sheet1'

METHOD      = 'geometric'
G_TARGET    = 0.030
DESIGN_PCT  = 90
N_ITER      = 10000
SEED        = 42
G_GRAV      = 9806.65

SHOW_PLOTS  = True

# --- EVENT YANG MAU DIPREDIKSI ---
INPUT = {
    'distance_m': 255,
    'charge_kg': 30,
    'row_number': 5,
    'controll_ms': 109,
    'wall_echelon_ms': 42,
    'freeface_echelon_ms': 0,
    'freeface_count': 2,
    'geological_condt': 'Fault',
    'measuring_elevation': 'higher',
    'tie_up_type': 'boxcut',
}

# =====================================================================
# [B] & [C] & [D] FUNGSI-FUNGSI NUMERIK
# =====================================================================

COL_AMAKS = 'Amaks (mm/s^s) Maks'
COL_G     = 'nilai_g'
COL_DIST  = 'distance_m'
COL_CHG   = 'charge_kg'

PARAMS_CAT = ['geological_condt', 'tie_up_type', 'measuring_elevation', 'row_class', 'ffe_binned']
PARAMS_NUM = ['controll_binned', 'wall_echelon_ms', 'freeface_count']
ALL_PARAMS = PARAMS_CAT + PARAMS_NUM
CONTROLLABLE_OPS = ['tie_up_type', 'controll_binned', 'wall_echelon_ms', 'ffe_binned', 'freeface_count']

NICE = {
    'row_class': 'Row Number (kelas)', 'controll_binned': 'Control Delay (ms)',
    'wall_echelon_ms': 'Wall Echelon (ms)', 'ffe_binned': 'Freeface Echelon (ms)',
    'freeface_count': 'Freeface Count', 'geological_condt': 'Geological Condition',
    'tie_up_type': 'Tie-up Type', 'measuring_elevation': 'Measuring Elevation',
}

STATUS_THRESHOLDS = [(0.020, 'EXCELLENT'), (0.030, 'SAFE'), (0.100, 'MODERATE'),
                     (0.200, 'RISKY'), (float('inf'), 'EXTREMELY RISKY')]

def get_status(g):
    for thr, label in STATUS_THRESHOLDS:
        if g <= thr:
            return label
    return 'EXTREMELY RISKY'

def apply_binning(df):
    df = df.copy()
    df['row_class'] = df['row_number'].apply(
        lambda r: '1-3' if r <= 3 else ('4-7' if r <= 7 else '>7'))
    df['controll_binned'] = df['controll_ms'].replace({42: 67})
    df['ffe_binned'] = df['freeface_echelon_ms'].apply(
        lambda x: '0' if x == 0 else ('42-67' if x <= 67 else '>=109'))
    return df

def validate(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'freeface_count' in df.columns:
        df['freeface_count'] = df['freeface_count'].replace({0: 1})
    df = df.dropna(subset=[COL_AMAKS, COL_DIST, COL_CHG])
    df = df[(df[COL_AMAKS] > 0) & (df[COL_DIST] > 0) & (df[COL_CHG] > 0)].copy()
    df = df[df['row_number'] < 10].copy()
    df['scale_distance'] = df[COL_DIST] / np.sqrt(df[COL_CHG])
    df['Amaks_maks'] = df[COL_AMAKS]
    return apply_binning(df)

def _aggregate(values, method):
    arr = np.asarray(values, dtype=float)
    if method == 'geometric':
        return float(np.exp(np.mean(np.log(arr))))
    return float(np.mean(arr))

def calibrate(df, method=METHOD):
    ln_a  = np.log(df['Amaks_maks'].values)
    ln_sd = np.log(df['scale_distance'].values)
    slope, intercept, r_val, _, _ = stats.linregress(ln_sd, ln_a)
    k = float(np.exp(intercept))
    n = float(-slope)
    resid = ln_a - (intercept + slope * ln_sd)
    sigma = float(np.std(resid, ddof=1))
    r_sq  = float(r_val ** 2)
    df = df.copy()
    df['res_ratio'] = df['Amaks_maks'].values / (k * df['scale_distance'].values ** (-n))
    real_factor, counts, freq = {}, {}, {}
    n_tot = len(df)
    for p in ALL_PARAMS:
        rp, cp, fp = {}, {}, {}
        for key, vals in df.groupby(p)['res_ratio']:
            rp[key] = _aggregate(vals.values, method)
            cp[key] = len(vals)
            fp[key] = len(vals) / n_tot
        real_factor[p], counts[p], freq[p] = rp, cp, fp
    ranges = {p: max(v.values()) - min(v.values()) for p, v in real_factor.items()}
    total  = sum(ranges.values())
    weight = {p: (ranges[p] / total if total > 0 else 0.0) for p in ALL_PARAMS}
    return dict(k=k, n=n, sigma=sigma, r2=r_sq, real_factor=real_factor, weight=weight, counts=counts, freq=freq, df=df)

def _nearest_num(d, t):
    keys = np.array([float(x) for x in d.keys()])
    return list(d.values())[int(np.abs(keys - float(t)).argmin())]

def real_factor_value(p, v, cal):
    f = cal['real_factor'][p]
    return f.get(v, 1.0) if isinstance(v, str) else _nearest_num(f, v)

def compute_wf(row, cal):
    return sum(real_factor_value(p, row[p], cal) * cal['weight'][p] for p in ALL_PARAMS)

def mean_wf_of(df, cal):
    return float(np.mean([compute_wf(r, cal) for _, r in df.iterrows()]))

def median_g_event(inp, cal, mean_wf):
    sd = inp['distance_m'] / np.sqrt(inp['charge_kg'])
    return cal['k'] * sd ** (-cal['n']) * (compute_wf(inp, cal) / mean_wf) / G_GRAV

def predict_g_mc(cal, sd, wf_norm, sigma, n_iter=N_ITER, seed=SEED):
    rng = np.random.default_rng(seed)
    err = rng.normal(0.0, sigma, n_iter)
    return cal['k'] * (sd ** (-cal['n'])) * np.exp(err) * wf_norm / G_GRAV

def loocv_predictions(df, method=METHOD):
    pairs = []
    for i in list(df.index):
        train = df.drop(index=i)
        test  = df.loc[i]
        c  = calibrate(train, method)
        mw = mean_wf_of(train, c)
        gp = c['k'] * test['scale_distance'] ** (-c['n']) * (compute_wf(test, c) / mw) / G_GRAV
        pairs.append((float(test[COL_G]), float(gp)))
    return pairs

def design_limits(pairs):
    ratios = np.array([a / p for a, p in pairs if p > 0])
    logr = np.log(ratios)
    mu_e, s_e = float(np.mean(logr)), float(np.std(logr, ddof=1))
    table = {}
    for pct in (75, 90, 95):
        table[pct] = dict(emp=float(np.percentile(ratios, pct)))
    return dict(mu_e=mu_e, s_e=s_e, table=table)

def importance_metrics(cal, inp_row, mean_wf):
    weight = cal['weight']
    levels, g_levels = {}, {}
    for p in ALL_PARAMS:
        lv = sorted(cal['real_factor'][p], key=float) if p in PARAMS_NUM else sorted(cal['real_factor'][p], key=str)
        gl = []
        for val in lv:
            mod = dict(inp_row)
            mod[p] = val
            gl.append(median_g_event(mod, cal, mean_wf))
        levels[p], g_levels[p] = lv, gl
    C = {}
    for p in ALL_PARAMS:
        vals = list(cal['real_factor'][p].values())
        frs  = [cal['freq'][p][kk] for kk in cal['real_factor'][p].keys()]
        m    = sum(v * f for v, f in zip(vals, frs))
        C[p] = (weight[p] ** 2) * sum(f * (v - m) ** 2 for v, f in zip(vals, frs))
    s = sum(C.values())
    sobol = {p: (C[p] / s * 100 if s > 0 else 0.0) for p in ALL_PARAMS}
    oat_raw = {p: max(g_levels[p]) - min(g_levels[p]) for p in ALL_PARAMS}
    so = sum(oat_raw.values())
    oat = {p: (oat_raw[p] / so * 100 if so > 0 else 0.0) for p in ALL_PARAMS}
    return dict(sobol=sobol, oat=oat, levels=levels, g_levels=g_levels)

def recommend(inp, cal, mean_wf, df, target, design_lambda):
    g_med = median_g_event(inp, cal, mean_wf)
    g0 = g_med * design_lambda
    
    st.subheader("REKOMENDASI KEPUTUSAN")
    st.write(f"**Median g** = {g_med:.5f} | **DESIGN g (x{design_lambda:.2f})** = {g0:.5f}")
    st.write(f"**Ambang aman** <= {target:.3f}")
    
    if g0 <= target:
        st.success('STATUS: AMAN - tidak perlu perubahan desain.')
        return
    
    st.error('STATUS: DI ATAS AMBANG -> perlu mitigasi.')
    
    st.markdown("**(LANGKAH 1) Ubah parameter operasional ke level teraman:**")
    work = dict(inp)
    found = False
    for p in CONTROLLABLE_OPS:
        rf = cal['real_factor'][p]
        safe = min(rf, key=lambda kk: rf[kk])
        cur = _nearest_num(rf, inp[p]) if p in PARAMS_NUM else rf.get(inp[p], 1.0)
        if rf[safe] < cur - 1e-12:
            work[p] = safe
            st.write(f"- {NICE[p]}: {inp[p]} -> **{safe}**")
            found = True
            
    if not found:
        st.write("(tidak ada perubahan operasional yang menurunkan g)")
        
    gd_ops = median_g_event(work, cal, mean_wf) * design_lambda
    st.info(f"**=> DESIGN g setelah tweak:** {gd_ops:.5f}")
    
    st.markdown("**(LANGKAH 2) Kurangi charge per delay:**")
    if gd_ops > target:
        ch_t = work['charge_kg'] * (target / gd_ops) ** (2.0 / cal['n'])
        ch_min = float(df[COL_CHG].min())
        turun = (1 - ch_t / inp['charge_kg']) * 100
        st.write(f"- Charge: {inp['charge_kg']:.1f} kg -> **{ch_t:.1f} kg** (turun {turun:.0f}%)")
        if ch_t < ch_min:
            st.warning(f"di bawah rentang data (min {ch_min:.0f} kg) -> ekstrapolasi, kurang andal")
            
        final = dict(work); final['charge_kg'] = max(ch_t, ch_min)
        st.info(f"**=> Proyeksi Median:** {median_g_event(final, cal, mean_wf):.5f} | **DESIGN:** {median_g_event(final, cal, mean_wf) * design_lambda:.5f}")
        st.caption("CATATAN: charge turun memperbaiki getaran TAPI bisa memperburuk fragmentasi.")
    else:
        st.write("Tidak perlu - target tercapai hanya dengan tweak operasional.")

# =====================================================================
# [E] JALANKAN PROSES
# =====================================================================

with st.spinner("Mengunduh data dari Google Drive..."):
    try:
        response = requests.get(DOWNLOAD_URL)
        response.raise_for_status() 
        df_raw = pd.read_excel(io.BytesIO(response.content), sheet_name=SHEET_NAME, engine='openpyxl')
    except Exception as e:
        st.error(f"Gagal mengambil data dari Google Drive. Pastikan akses file adalah 'Anyone with the link'. Error: {e}")
        st.stop()

# Validasi dan Kalkulasi
df  = validate(df_raw)
cal = calibrate(df, METHOD)
df  = cal['df']
mean_wf = mean_wf_of(df, cal)

pairs   = loocv_predictions(df, METHOD)
mae_loo = float(np.mean([abs(a - p) for a, p in pairs]))
g_range = df[COL_G].max() - df[COL_G].min()
nmae    = mae_loo / g_range if g_range > 0 else float('nan')
n_under = sum(1 for a, p in pairs if p < a)
dl      = design_limits(pairs)
sigma_mc      = dl['s_e']                               
design_lambda = dl['table'][DESIGN_PCT]['emp']  

# --- Siapkan Input ---
INPUT['row_class'] = '1-3' if INPUT['row_number'] <= 3 else ('4-7' if INPUT['row_number'] <= 7 else '>7')
INPUT['controll_binned'] = 67 if INPUT['controll_ms'] == 42 else INPUT['controll_ms']
INPUT['ffe_binned'] = '0' if INPUT['freeface_echelon_ms'] == 0 else ('42-67' if INPUT['freeface_echelon_ms'] <= 67 else '>=109')

sd = INPUT['distance_m'] / np.sqrt(INPUT['charge_kg'])
wf_norm = compute_wf(INPUT, cal) / mean_wf
g_arr = predict_g_mc(cal, sd, wf_norm, sigma_mc, N_ITER)
g_med = float(np.median(g_arr))
g_mean = g_med * math.exp(sigma_mc ** 2 / 2.0)     
g_p10 = float(np.percentile(g_arr, 10))
g_p90 = float(np.percentile(g_arr, 90))
g_design = g_med * design_lambda
prob = 100.0 * float(np.mean(g_arr > G_TARGET))

# =====================================================================
# MENAMPILKAN HASIL KE WEB STREAMLIT
# =====================================================================

st.subheader("RINGKASAN KALIBRASI DATA AKTUAL")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Data", f"{len(df)} pengukuran")
col2.metric("MAE LOOCV", f"{mae_loo:.5f} g")
col3.metric("NMAE", f"{nmae * 100:.1f}%")

st.text(f"""
[REGRESI ln(Amaks) vs ln(SD)]
k = {cal['k']:.2f} | n = {cal['n']:.4f} | R2 = {cal['r2']:.4f}
sigma in-sample = {cal['sigma']:.4f} | sigma LOOCV = {dl['s_e']:.4f} (MC)
MEAN_WF = {mean_wf:.4f} | lambda P{DESIGN_PCT} = {design_lambda:.3f}
""")

st.divider()

st.subheader("PREDIKSI GETARAN (g)")
st.text(f"""
Distance {INPUT['distance_m']} m | Charge {INPUT['charge_kg']} kg | SD {sd:.2f}
Geologi {INPUT['geological_condt']} | Elev {INPUT['measuring_elevation']} | Tie-up {INPUT['tie_up_type']}
Row {INPUT['row_number']} | Control {INPUT['controll_ms']} ms | Wall ech {INPUT['wall_echelon_ms']} ms | FF ech {INPUT['freeface_echelon_ms']} ms | FF count {INPUT['freeface_count']}
WF normalized = {wf_norm:.3f}
""")

st.markdown(f"""
* **Prediksi g (harapan)** : `{g_mean:.5f}` ({get_status(g_mean)})
* **Median (P50)** : `{g_med:.5f}`
* **Rentang P10-P90** : `{g_p10:.5f}` - `{g_p90:.5f}`
* **Probabilitas g > {G_TARGET:.3f}** : `{prob:.1f}%`
* **DESIGN VALUE (P90)** : `{g_design:.5f}` ({get_status(g_design)})  *(<- acuan keputusan)*
""")

st.divider()

recommend(INPUT, cal, mean_wf, df, G_TARGET, design_lambda)

st.divider()

# Menampilkan Grafik
if SHOW_PLOTS:
    st.subheader("Visualisasi")
    imp = importance_metrics(cal, INPUT, mean_wf)
    order = sorted(ALL_PARAMS, key=lambda p: -imp['sobol'][p])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(g_arr, bins=60, color='#4a7fb5', edgecolor='white', linewidth=0.3)
    ax1.axvline(G_TARGET, color='#c0392b', lw=2.2, ls='--', label='Ambang ' + format(G_TARGET, '.3f'))
    ax1.axvline(g_med, color='#1a5276', lw=2, label='Median ' + format(g_med, '.4f'))
    ax1.axvline(g_design, color='#8e44ad', lw=2, ls=':', label='Design P90 ' + format(g_design, '.4f'))
    ax1.set_xlabel('Nilai g'); ax1.set_ylabel('Frekuensi')
    ax1.set_title(f'Distribusi Monte Carlo ({N_ITER:,} iterasi)')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.2)

    vals = [imp['sobol'][p] for p in order]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(order)))
    bars = ax2.barh(range(len(order)), vals, color=colors, edgecolor='black', linewidth=0.6)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([NICE[p] for p in order], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Kontribusi terhadap varians g - Sobol (%)')
    ax2.set_title('Kepentingan Faktor')
    for b, v in zip(bars, vals):
        ax2.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                 format(v, '.1f') + '%', va='center', fontsize=8, fontweight='bold')
    ax2.set_xlim(0, max(vals) * 1.18); ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    # PENTING: Gunakan st.pyplot() bukan plt.show()
    st.pyplot(fig)
