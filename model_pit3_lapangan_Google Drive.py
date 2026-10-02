# =====================================================================
# PREDIKSI GETARAN PIT 3 - VERSI LAPANGAN (Streamlit / Server)
# =====================================================================
# Logika NUMERIK IDENTIK dengan app.py (versi paper):
#   - binning di dalam tiap parameter (row_class, controll_binned, ffe_binned)
#   - record 10-row dibuang (di luar rentang operasi sekarang)
#   - freeface_count = 0 dikoreksi jadi 1
#   - sigma Monte Carlo dari LOOCV (bukan in-sample)
#   - Design Value P90 (median x lambda)
#   - Probabilitas melampaui ambang
#
# BEDA dengan app.py: script ini BACA EXCEL dari Google Drive.
# Tambah data baru di Excel -> jalankan ulang -> semua terhitung ulang.
# =====================================================================

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('[INFO] Google Drive ter-mount.')
except Exception:
    pass

import math
import numpy as np
import pandas as pd
from scipy import stats
import requests  # Ditambahkan untuk download file
import io        # Ditambahkan untuk membaca file di memory

# =====================================================================
# [A] KONFIGURASI - HANYA BAGIAN INI YANG PERLU DIUBAH
# =====================================================================

# Gunakan ID File dari link Google Drive Anda
FILE_ID = '1xwJz5xrysYvEPK7lZMweL5rAAbBFkNA1'
# Format URL khusus untuk memaksa download/export ke Excel
DOWNLOAD_URL = f'https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=xlsx'
SHEET_NAME = 'Sheet1'

METHOD      = 'geometric'   # geometric = disarankan
G_TARGET    = 0.030         # ambang aman
DESIGN_PCT  = 90            # P90 = disarankan
N_ITER      = 10000
SEED        = 42
G_GRAV      = 9806.65

SHOW_PLOTS  = True

# --- EVENT YANG MAU DIPREDIKSI (isi sesuai rencana peledakan) ---
INPUT = {
    'distance_m': 255,
    'charge_kg': 30,
    'row_number': 5,
    'controll_ms': 109,
    'wall_echelon_ms': 42,
    'freeface_echelon_ms': 0,
    'freeface_count': 2,
    'geological_condt': 'Fault',      # coal / normal / Fault
    'measuring_elevation': 'higher',  # lower / normal / higher
    'tie_up_type': 'boxcut',          # boxcut / echelon
}

# =====================================================================
# [B] KOLOM & PARAMETER (sama persis app.py)
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

SEP  = '=' * 66
SEP2 = '-' * 66

def get_status(g):
    for thr, label in STATUS_THRESHOLDS:
        if g <= thr:
            return label
    return 'EXTREMELY RISKY'

# =====================================================================
# [C] BINNING & VALIDASI (identik app.py)
# =====================================================================

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
    required = [COL_AMAKS, COL_G, COL_DIST, COL_CHG, 'geological_condt', 'tie_up_type',
                'measuring_elevation', 'row_number', 'controll_ms', 'wall_echelon_ms',
                'freeface_echelon_ms', 'freeface_count']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError('Kolom wajib tidak ada di Excel: ' + ', '.join(missing))
    n0 = len(df)
    df = df.dropna(subset=[COL_AMAKS, COL_DIST, COL_CHG])
    df = df[(df[COL_AMAKS] > 0) & (df[COL_DIST] > 0) & (df[COL_CHG] > 0)].copy()
    df = df[df['row_number'] < 10].copy()
    if len(df) < 5:
        raise ValueError('Data valid hanya ' + str(len(df)) + ' baris - terlalu sedikit.')
    if n0 - len(df) > 0:
        print('  [!] ' + str(n0 - len(df)) + ' baris dibuang (kosong/<=0/row>=10)')
    df['scale_distance'] = df[COL_DIST] / np.sqrt(df[COL_CHG])
    df['Amaks_maks'] = df[COL_AMAKS]
    return apply_binning(df)

# =====================================================================
# [D] KALIBRASI & PREDIKSI (identik app.py)
# =====================================================================

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
    return dict(k=k, n=n, sigma=sigma, r2=r_sq, real_factor=real_factor,
                weight=weight, counts=counts, freq=freq, df=df)

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
    print(SEP)
    print('   REKOMENDASI KEPUTUSAN')
    print(SEP)
    print('  Median g = ' + format(g_med, '.5f') + '  |  DESIGN g (x' + format(design_lambda, '.2f') + ') = ' + format(g0, '.5f'))
    print('  Ambang aman <= ' + format(target, '.3f'))
    if g0 <= target:
        print('  STATUS: AMAN - tidak perlu perubahan desain.')
        print('')
        return
    print('  STATUS: DI ATAS AMBANG -> perlu mitigasi.')
    print('')
    print('  [LANGKAH 1] Ubah parameter operasional ke level teraman:')
    work = dict(inp)
    found = False
    for p in CONTROLLABLE_OPS:
        rf = cal['real_factor'][p]
        safe = min(rf, key=lambda kk: rf[kk])
        cur = _nearest_num(rf, inp[p]) if p in PARAMS_NUM else rf.get(inp[p], 1.0)
        if rf[safe] < cur - 1e-12:
            work[p] = safe
            print('     - ' + NICE[p] + ': ' + str(inp[p]) + ' -> ' + str(safe))
            found = True
    if not found:
        print('     (tidak ada perubahan operasional yang menurunkan g)')
    gd_ops = median_g_event(work, cal, mean_wf) * design_lambda
    print('     => DESIGN g setelah tweak: ' + format(gd_ops, '.5f'))
    print('')
    print('  [LANGKAH 2] Kurangi charge per delay:')
    if gd_ops > target:
        ch_t = work['charge_kg'] * (target / gd_ops) ** (2.0 / cal['n'])
        ch_min = float(df[COL_CHG].min())
        turun = (1 - ch_t / inp['charge_kg']) * 100
        print('     - Charge: ' + format(inp['charge_kg'], '.1f') + ' kg -> ' + format(ch_t, '.1f') + ' kg (turun ' + format(turun, '.0f') + '%)')
        if ch_t < ch_min:
            print('     [!] di bawah rentang data (min ' + format(ch_min, '.0f') + ' kg) -> ekstrapolasi, kurang andal')
        final = dict(work); final['charge_kg'] = max(ch_t, ch_min)
        print('     => proyeksi: median ' + format(median_g_event(final, cal, mean_wf), '.5f')
              + ' | DESIGN ' + format(median_g_event(final, cal, mean_wf) * design_lambda, '.5f'))
        print('     CATATAN: charge turun memperbaiki getaran TAPI bisa memperburuk fragmentasi.')
    else:
        print('     Tidak perlu - target tercapai hanya dengan tweak operasional.')
    print('')

# =====================================================================
# [E] JALANKAN
# =====================================================================

print(''); print(SEP)
print('   KALIBRASI DARI DATA AKTUAL')
print(SEP)

try:
    print(f'  [INFO] Mengunduh data dari Google Drive (ID: {FILE_ID})...')
    response = requests.get(DOWNLOAD_URL)
    response.raise_for_status() # Cek apakah ada error HTTP (misal: 404 atau 403 Forbidden)
    
    # Membaca data langsung dari memory (BytesIO) menggunakan pandas
    df_raw = pd.read_excel(io.BytesIO(response.content), sheet_name=SHEET_NAME, engine='openpyxl')
    print('  [INFO] Berhasil mengunduh dan membaca data Excel.')
except Exception as e:
    print(f'\n[ERROR FATAL] Gagal mengambil data dari Google Drive.')
    print(f'Penyebab: {e}')
    print(f'Solusi: Pastikan file di Google Drive sudah di-setting "Anyone with the link can view".')
    exit() # Menghentikan script jika gagal download

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
sigma_mc      = dl['s_e']                               # sigma LOOCV
design_lambda = dl['table'][DESIGN_PCT]['emp']  # lambda P90 empiris

print('')
print('  ID File     : ' + str(FILE_ID))
print('  Jumlah data : ' + str(len(df)) + ' pengukuran')
print('  Range g     : ' + format(df[COL_G].min(), '.5f') + ' - ' + format(df[COL_G].max(), '.5f'))
print('')
print('  [REGRESI ln(Amaks) vs ln(SD)]')
print('  k = ' + format(cal['k'], '.2f') + ' | n = ' + format(cal['n'], '.4f')
      + ' | R2 = ' + format(cal['r2'], '.4f'))
print('  sigma in-sample = ' + format(cal['sigma'], '.4f')
      + ' | sigma LOOCV = ' + format(dl['s_e'], '.4f') + '  (yang dipakai MC)')
print('  MEAN_WF = ' + format(mean_wf, '.4f'))
print('')
print('  [VALIDASI LOOCV]')
print('  MAE  = ' + format(mae_loo, '.5f') + ' g')
print('  NMAE = ' + format(nmae * 100, '.1f') + '%')
print('  Underprediksi = ' + str(n_under) + '/' + str(len(pairs))
      + ' (' + format(100.0 * n_under / len(pairs), '.0f') + '%)')
print('  lambda P' + str(DESIGN_PCT) + ' = ' + format(design_lambda, '.3f'))

# --- siapkan input (binning otomatis, sama seperti app.py) ---
INPUT['row_class'] = '1-3' if INPUT['row_number'] <= 3 else ('4-7' if INPUT['row_number'] <= 7 else '>7')
INPUT['controll_binned'] = 67 if INPUT['controll_ms'] == 42 else INPUT['controll_ms']
INPUT['ffe_binned'] = '0' if INPUT['freeface_echelon_ms'] == 0 else ('42-67' if INPUT['freeface_echelon_ms'] <= 67 else '>=109')

sd = INPUT['distance_m'] / np.sqrt(INPUT['charge_kg'])
wf_norm = compute_wf(INPUT, cal) / mean_wf
g_arr = predict_g_mc(cal, sd, wf_norm, sigma_mc, N_ITER)
g_med = float(np.median(g_arr))
g_mean = g_med * math.exp(sigma_mc ** 2 / 2.0)     # koreksi Duan (1983)
g_p10 = float(np.percentile(g_arr, 10))
g_p90 = float(np.percentile(g_arr, 90))
g_design = g_med * design_lambda
prob = 100.0 * float(np.mean(g_arr > G_TARGET))

print(''); print(SEP)
print('   PREDIKSI GETARAN (g)')
print(SEP)
print('')
print('  Distance ' + str(INPUT['distance_m']) + ' m | Charge ' + str(INPUT['charge_kg'])
      + ' kg | SD ' + format(sd, '.2f'))
print('  Geologi ' + str(INPUT['geological_condt']) + ' | Elev ' + str(INPUT['measuring_elevation'])
      + ' | Tie-up ' + str(INPUT['tie_up_type']))
print('  Row ' + str(INPUT['row_number']) + ' (' + INPUT['row_class'] + ') | Control '
      + str(INPUT['controll_ms']) + ' ms | Wall ech ' + str(INPUT['wall_echelon_ms'])
      + ' ms | FF ech ' + str(INPUT['freeface_echelon_ms']) + ' ms | FF count ' + str(INPUT['freeface_count']))
print('  WF normalized = ' + format(wf_norm, '.3f')
      + (' (lebih berbahaya' if wf_norm > 1 else ' (lebih aman') + ' dari rata-rata site)')
print('')
print(SEP2)
print('  PREDIKSI g (harapan)      : ' + format(g_mean, '.5f') + '   ' + get_status(g_mean))
print('  Median (P50)              : ' + format(g_med, '.5f'))
print('  Rentang P10-P90           : ' + format(g_p10, '.5f') + ' - ' + format(g_p90, '.5f'))
print('  Probabilitas g > ' + format(G_TARGET, '.3f') + '   : ' + format(prob, '.1f') + '%')
print('  DESIGN VALUE (P90)        : ' + format(g_design, '.5f') + '   ' + get_status(g_design)
      + '   <- acuan keputusan')
print(SEP2)
print('')

recommend(INPUT, cal, mean_wf, df, G_TARGET, design_lambda)

# =====================================================================
# [F] KEPENTINGAN FAKTOR
# =====================================================================

imp = importance_metrics(cal, INPUT, mean_wf)
order = sorted(ALL_PARAMS, key=lambda p: -imp['sobol'][p])

print(SEP)
print('   KEPENTINGAN FAKTOR')
print(SEP)
print('')
print('  ' + 'Faktor'.ljust(24) + 'Weight'.rjust(9) + 'OAT'.rjust(9) + 'Sobol'.rjust(9))
print('  ' + '-' * 51)
for p in order:
    print('  ' + NICE[p].ljust(24)
          + (format(cal['weight'][p] * 100, '.1f') + '%').rjust(9)
          + (format(imp['oat'][p], '.1f') + '%').rjust(9)
          + (format(imp['sobol'][p], '.1f') + '%').rjust(9))
print('')

# =====================================================================
# [G] SENSITIVITAS CHARGE
# =====================================================================

print(SEP)
print('   SENSITIVITAS CHARGE (distance tetap ' + str(INPUT['distance_m']) + ' m)')
print(SEP)
print('')
print('  ' + 'Charge'.rjust(7) + 'SD'.rjust(8) + 'Median g'.rjust(11)
      + 'DESIGN g'.rjust(11) + '  Status (design)')
print('  ' + '-' * 55)
for ch in [20, 25, 30, 35, 40, 45, 50, 60, 70]:
    sd_ch = INPUT['distance_m'] / np.sqrt(ch)
    arr = predict_g_mc(cal, sd_ch, wf_norm, sigma_mc, 4000)
    gp = float(np.median(arr))
    gd = gp * design_lambda
    mark = '  <- rencana' if ch == INPUT['charge_kg'] else ''
    print('  ' + str(ch).rjust(7) + format(sd_ch, '.2f').rjust(8)
          + format(gp, '.5f').rjust(11) + format(gd, '.5f').rjust(11)
          + '  ' + get_status(gd) + mark)
print('')

# =====================================================================
# [H] GRAFIK
# =====================================================================

if SHOW_PLOTS:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(g_arr, bins=60, color='#4a7fb5', edgecolor='white', linewidth=0.3)
    ax1.axvline(G_TARGET, color='#c0392b', lw=2.2, ls='--', label='Ambang ' + format(G_TARGET, '.3f'))
    ax1.axvline(g_med, color='#1a5276', lw=2, label='Median ' + format(g_med, '.4f'))
    ax1.axvline(g_design, color='#8e44ad', lw=2, ls=':', label='Design P90 ' + format(g_design, '.4f'))
    ax1.set_xlabel('Nilai g'); ax1.set_ylabel('Frekuensi')
    ax1.set_title('Distribusi Monte Carlo (' + format(N_ITER, ',') + ' iterasi)')
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
    plt.show()

print(SEP)
print('   RINGKASAN')
print(SEP)
print('  Data      : ' + str(len(df)) + ' pengukuran (dari Google Drive)')
print('  k = ' + format(cal['k'], '.2f') + ' | n = ' + format(cal['n'], '.4f')
      + ' | R2 = ' + format(cal['r2'], '.4f') + ' | lambda = ' + format(design_lambda, '.3f'))
print('  MAE LOOCV = ' + format(mae_loo, '.5f') + ' | NMAE = ' + format(nmae * 100, '.1f') + '%')
print('  Prediksi g = ' + format(g_mean, '.5f') + ' | DESIGN = ' + format(g_design, '.5f')
      + ' | Prob lampau = ' + format(prob, '.1f') + '%')
print('')
