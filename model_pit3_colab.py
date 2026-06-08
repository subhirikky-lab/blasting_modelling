# =====================================================================
# HYBRID MONTE CARLO GROUND VIBRATION MODEL - PIT 3  (v3.2 - ALL-IN-ONE)
# Auto-mount Drive + Prediksi g + Design Value (P90) + Validasi LOOCV +
# Sensitivity + Importance (Sobol) + Diagnostik Resonansi + Visualisasi +
# RECOMMENDATION ENGINE
# =====================================================================
# Update data di Excel -> jalankan ulang -> semua parameter terhitung
# ulang otomatis. Logika numerik IDENTIK dengan web app (app.py).
# Paste-safe Colab: TANPA f-string, pakai str()+format().
#
# PERUBAHAN v3.2 (vs v3):
#   + DESIGN VALUE P90 = median x lambda (lambda empiris dari sebaran LOOCV)
#     -> untuk keputusan geotek; median tetap untuk nilai harapan mining.
#   + sigma Monte Carlo memakai sigma LOOCV (lebih jujur, bukan in-sample).
#   + DIAGNOSTIK RESONANSI frekuensi: AUDIT pita bahaya (Floyd 2008),
#     BUKAN prediksi (freq vs SD R2~0.002 = tidak valid sbg prediktor).
#   + Anotasi DISARANKAN pada pilihan metode.
#
# TIGA METRIK KEPENTINGAN FAKTOR (untuk paper):
#   (A) weight model  w_p ~ range_p          (bobot pencampur internal)
#   (B) OAT g-range   ~ range_p^2            (sensitivitas lokal 1 event)
#   (C) Sobol orde-1  S_p ~ w_p^2 * Var(f_p) (variance-based, GLOBAL) [UTAMA]
# Referensi: Duvall & Fogelson (1962); Ambraseys & Hendron (1968);
#   Siskind dkk (1980); Dowding (1985); Hasanipanah dkk (2017);
#   Sobol (1993); Saltelli dkk (2008); Floyd (2008); Lucca (2003).
# =====================================================================

# --- Auto-mount Google Drive (otomatis jika dijalankan di Google Colab) ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('[INFO] Google Drive ter-mount otomatis.')
except Exception:
    pass  # bukan di Colab -> lewati tanpa error

import numpy as np
import pandas as pd
from scipy import stats

# =====================================================================
# [A] KONFIGURASI - HANYA BAGIAN INI YANG PERLU DIUBAH
# =====================================================================

DATA_FILE  = r'/content/drive/MyDrive/DATA_BLASTING/nilai_g_pit3.xlsx'   # <- ganti path/nama file jika perlu
SHEET_NAME = 'Sheet1'               # <- ganti jika sheet berbeda

REAL_FACTOR_METHOD = 'geometric'    # DISARANKAN 'geometric'. 'arithmetic' hanya utk uji sensitivitas.
SHRINKAGE_PRIOR    = 0.0
MIN_SAMPLES        = 3

# --- Pengaturan Design Value & Monte Carlo ---
USE_LOOCV_SIGMA = True              # DISARANKAN True: sigma MC dari sebaran LOOCV (lebih jujur dari in-sample).
DESIGN_PCT      = 90                # DISARANKAN 90 (P90). P95 rapuh di n kecil -> pakai metode 'normal'.
DESIGN_METHOD   = 'empirical'       # DISARANKAN 'empirical'. 'normal' = asumsi lognormal (cross-check).

SHOW_PLOTS = True
N_ITER     = 10000
SEED       = 42
G_GRAV     = 9806.65                # mm/s^2

G_TARGET   = 0.030                  # ambang aman; design g > ini -> muncul rekomendasi

STATUS_THRESHOLDS = [
    (0.020, 'EXCELLENT'), (0.030, 'SAFE'), (0.100, 'MODERATE'),
    (0.200, 'RISKY'), (float('inf'), 'EXTREMELY RISKY'),
]

COL_AMAKS = 'Amaks (mm/s^s) Maks'
COL_G     = 'nilai_g'
COL_DIST  = 'distance_m'
COL_CHG   = 'charge_kg'
COL_FREQ  = 'frekuensi_hz'

PARAMS_CAT = ['geological_condt', 'tie_up_type', 'measuring_elevation']
PARAMS_NUM = ['row_number', 'controll_ms', 'wall_echelon_ms', 'freeface_echelon_ms', 'freeface_count']
ALL_PARAMS = PARAMS_CAT + PARAMS_NUM

# Parameter yang MASIH BISA DIUBAH setelah lubang dibor (untuk rekomendasi).
CONTROLLABLE_OPS = ['tie_up_type', 'controll_ms', 'wall_echelon_ms', 'freeface_echelon_ms', 'freeface_count']

NICE = {
    'row_number': 'Row Number', 'controll_ms': 'Control Delay (ms)',
    'wall_echelon_ms': 'Wall Echelon (ms)', 'freeface_echelon_ms': 'Freeface Echelon (ms)',
    'freeface_count': 'Freeface Count', 'geological_condt': 'Geological Condition',
    'tie_up_type': 'Tie-up Type', 'measuring_elevation': 'Measuring Elevation',
}

# --- Diagnostik resonansi (Floyd 2008) ---
ROCK_NATURAL_FREQ    = {'weak': (3, 7), 'blocky': (7, 15), 'strong': (15, 23)}
GEO_TO_ROCK          = {'coal': 'weak', 'normal': 'blocky', 'Fault': 'weak'}
SLOPE_FREQ_THRESHOLD = 40.0

# --- Scaled Depth of Burial (SDOB) - ERG/Tobin (2013) ---
# SD = (stemming + 5d) / W10^(1/3); W10 = rho * (pi/4 d^2) * 10d.
# Verifikasi contoh artikel (stem=5m, d=229mm, rho=1210) -> SD=1.27 (controlled).
COL_DEPTH     = 'depth_m'
COL_HOLE_DIA  = 'hole_diameter_mm'
RHO_EXPLOSIVE = 1150.0   # kg/m^3 (= 1.15 g/cc), emulsi Pit 3
# 6 zona SDOB sesuai diagram Chiappetta/ERG:
SDOB_BANDS = [
    ('Uncontrolled (0-0.60): flyrock & airblast hebat', 0.0, 0.60),
    ('Cratering (0.60-0.92): fragmentasi sangat halus', 0.60, 0.92),
    ('Controlled (0.92-1.40): fragmentasi baik, getaran/airblast wajar', 0.92, 1.40),
    ('Very controlled (1.40-1.80): frag lebih kasar, TANPA flyrock', 1.40, 1.80),
    ('Minimal surface (1.80-2.40): gangguan permukaan kecil', 1.80, 2.40),
    ('Insignificant (>2.40): efek permukaan tak berarti', 2.40, float('inf')),
]

INPUT = {
    'distance_m': 200, 'charge_kg': 37, 'row_number': 5, 'controll_ms': 67,
    'wall_echelon_ms': 42, 'freeface_echelon_ms': 0, 'freeface_count': 1,
    'geological_condt': 'normal', 'measuring_elevation': 'higher', 'tie_up_type': 'echelon',
    'depth_m': 7.45, 'hole_diameter_mm': 200,   # untuk diagnostik SDOB (tidak mengubah prediksi g)
}

SEP  = '=' * 64
SEP2 = '-' * 64

# =====================================================================
# [B] FUNGSI UTILITAS
# =====================================================================

def get_status(g):
    for thr, label in STATUS_THRESHOLDS:
        if g <= thr:
            return label
    return 'EXTREMELY RISKY'

def load_and_validate(path, sheet):
    try:
        df = pd.read_excel(path, sheet_name=sheet, header=0)
    except FileNotFoundError:
        raise FileNotFoundError("File '" + str(path) + "' tidak ditemukan. Periksa path di bagian KONFIGURASI.")
    df.columns = df.columns.str.strip()

    required = [COL_AMAKS, COL_G, COL_DIST, COL_CHG] + ALL_PARAMS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("Kolom wajib tidak ada di Excel: " + ", ".join(missing) + " | Kolom tersedia: " + str(list(df.columns)))

    n0 = len(df)
    df = df.dropna(subset=[COL_AMAKS, COL_DIST, COL_CHG])
    df = df[(df[COL_AMAKS] > 0) & (df[COL_DIST] > 0) & (df[COL_CHG] > 0)].copy()
    dropped = n0 - len(df)
    if dropped > 0:
        print("  [!] " + str(dropped) + " baris dibuang (Amaks/distance/charge kosong atau <= 0)")
    if len(df) < 5:
        raise ValueError("Data valid hanya " + str(len(df)) + " baris - terlalu sedikit.")

    df['scale_distance'] = df[COL_DIST] / np.sqrt(df[COL_CHG])
    df['Amaks_maks']     = df[COL_AMAKS]
    return df

def _aggregate(values, method):
    arr = np.asarray(values, dtype=float)
    if method == 'geometric':
        return float(np.exp(np.mean(np.log(arr))))
    return float(np.mean(arr))

def calibrate(df, method=REAL_FACTOR_METHOD, prior=SHRINKAGE_PRIOR):
    ln_a  = np.log(df['Amaks_maks'].values)
    ln_sd = np.log(df['scale_distance'].values)
    slope, intercept, r_val, _, _ = stats.linregress(ln_sd, ln_a)

    k     = float(np.exp(intercept))
    n     = float(-slope)
    resid = ln_a - (intercept + slope * ln_sd)
    sigma = float(np.std(resid, ddof=1))
    r_sq  = float(r_val ** 2)

    df = df.copy()
    df['res_ratio'] = df['Amaks_maks'].values / (k * df['scale_distance'].values ** (-n))

    real_factor, counts, freq = {}, {}, {}
    n_tot = len(df)
    for p in ALL_PARAMS:
        groups = df.groupby(p)['res_ratio']
        rp, cp, fp = {}, {}, {}
        for key, vals in groups:
            m  = _aggregate(vals.values, method)
            nc = len(vals)
            m  = (nc * m + prior * 1.0) / (nc + prior) if prior > 0 else m
            rp[key], cp[key], fp[key] = m, nc, nc / n_tot
        real_factor[p], counts[p], freq[p] = rp, cp, fp

    ranges = {p: max(v.values()) - min(v.values()) for p, v in real_factor.items()}
    total  = sum(ranges.values())
    weight = {p: (ranges[p] / total if total > 0 else 0.0) for p in ALL_PARAMS}

    return dict(k=k, n=n, sigma=sigma, r2=r_sq, real_factor=real_factor,
                weight=weight, counts=counts, freq=freq, ranges=ranges, df=df)

def _nearest_num(d, t):
    keys = np.array([float(x) for x in d.keys()])
    return list(d.values())[int(np.abs(keys - float(t)).argmin())]

def compute_wf(row, cal):
    wf = 0.0
    for p in ALL_PARAMS:
        f = cal['real_factor'][p]
        v = row[p]
        fval = f.get(v, 1.0) if isinstance(v, str) else _nearest_num(f, v)
        wf += fval * cal['weight'][p]
    return wf

def predict_g_mc(cal, scale_distance, wf_norm, sigma_mc, n_iter=N_ITER, seed=SEED):
    rng = np.random.default_rng(seed)
    err = rng.normal(0.0, sigma_mc, n_iter)
    return cal['k'] * (scale_distance ** (-cal['n'])) * np.exp(err) * wf_norm / G_GRAV

def median_g_event(inp, cal, mean_wf):
    """Median MC deterministik untuk satu event dict (punya distance_m & charge_kg)."""
    sd = inp['distance_m'] / np.sqrt(inp['charge_kg'])
    return cal['k'] * sd ** (-cal['n']) * (compute_wf(inp, cal) / mean_wf) / G_GRAV

def median_g(row, cal, mean_wf):
    """Median MC pakai key 'sd' (untuk importance_metrics)."""
    return cal['k'] * row['sd'] ** (-cal['n']) * (compute_wf(row, cal) / mean_wf) / G_GRAV

def loocv_predictions(df):
    """Pasangan (aktual, prediksi) leave-one-out -> untuk MAE & lambda jujur."""
    pairs = []
    for i in list(df.index):
        train = df.drop(index=i)
        test  = df.loc[i]
        c  = calibrate(train)
        mw = np.mean([compute_wf(r, c) for _, r in train.iterrows()])
        gp = c['k'] * test['scale_distance'] ** (-c['n']) * (compute_wf(test, c) / mw) / G_GRAV
        pairs.append((float(test[COL_G]), float(gp)))
    return pairs

def design_limits(pairs):
    """Faktor konservatisme lambda dari sebaran rasio aktual/prediksi LOOCV.
    empirical = persentil langsung (tanpa asumsi); normal = asumsi lognormal.
    Juga mengembalikan s_e = sigma LOOCV (std log-rasio) untuk MC."""
    ratios = np.array([a / p for a, p in pairs if p > 0])
    logr   = np.log(ratios)
    mu_e, s_e = float(np.mean(logr)), float(np.std(logr, ddof=1))
    table = {}
    for pct in (75, 90, 95):
        z = float(stats.norm.ppf(pct / 100.0))
        lam_emp  = float(np.percentile(ratios, pct))
        lam_norm = float(np.exp(mu_e + z * s_e))
        table[pct] = dict(emp=lam_emp, norm=lam_norm,
                          under_emp=100.0 * float(np.mean(ratios > lam_emp)),
                          under_norm=100.0 * float(np.mean(ratios > lam_norm)))
    return dict(mu_e=mu_e, s_e=s_e, table=table, ratios=ratios)

def frequency_diagnostics(df):
    """DIAGNOSTIK RESONANSI MURNI (bukan prediktor).
    Temuan Pit 3: freq vs scale-distance R2 ~ 0.002 (nol) -> frekuensi TIDAK
    valid diprediksi dari SD. Frekuensi diposisikan sbg AUDIT pita bahaya
    dari event aktual (Floyd 2008; zona resonansi lereng)."""
    if COL_FREQ not in df.columns:
        return None
    sub = df.dropna(subset=[COL_FREQ])
    sub = sub[sub[COL_FREQ] > 0]
    if len(sub) < 1:
        return None
    freqs = sub[COL_FREQ].values.astype(float)
    n = int(len(freqs))
    band_defs = [('Weak (3-7 Hz)', 3.0, 7.0), ('Blocky (7-15 Hz)', 7.0, 15.0),
                 ('Strong (15-23 Hz)', 15.0, 23.0), ('Aman (>23 Hz)', 23.0, float('inf'))]
    bands = []
    for name, lo, hi in band_defs:
        c = int(np.sum((freqs > lo) & (freqs <= hi)))
        bands.append(dict(name=name, count=c, pct=100.0 * c / n if n > 0 else 0.0))
    n_slope = int(np.sum(freqs < SLOPE_FREQ_THRESHOLD))
    geo_audit = {}
    if 'geological_condt' in sub.columns:
        for geo in sorted(str(x) for x in sub['geological_condt'].dropna().unique()):
            g_sub = sub[sub['geological_condt'].astype(str) == geo]
            gf = g_sub[COL_FREQ].values.astype(float)
            if len(gf) == 0:
                continue
            rock = GEO_TO_ROCK.get(geo)
            rng  = ROCK_NATURAL_FREQ.get(rock) if rock else None
            in_res = int(np.sum(gf <= rng[1])) if rng is not None else 0
            geo_audit[geo] = dict(rock=rock, rng=rng, n=int(len(gf)), in_res=in_res,
                                  fmin=float(np.min(gf)), fmax=float(np.max(gf)))
    return dict(n=n, fmin=float(np.min(freqs)), fmax=float(np.max(freqs)),
                fmed=float(np.median(freqs)), bands=bands, n_slope=n_slope, geo_audit=geo_audit)

def compute_sdob(charge_kg, depth_m, hole_dia_mm, rho=RHO_EXPLOSIVE):
    """Scaled Depth of Burial (ERG/Tobin 2013): SD = (stemming + 5d)/W10^(1/3)."""
    try:
        ch = float(charge_kg); dep = float(depth_m); d = float(hole_dia_mm) / 1000.0
    except Exception:
        return None
    if ch <= 0 or dep <= 0 or d <= 0:
        return None
    lin_density = rho * (np.pi / 4.0) * d * d        # kg/m
    charge_len = ch / lin_density                    # panjang kolom isian (m)
    stemming = dep - charge_len                       # tinggi stemming (m)
    w10 = lin_density * (10.0 * d)                    # massa peledak dlm 10x diameter (kg)
    sdob = (stemming + 5.0 * d) / (w10 ** (1.0 / 3.0))
    band = 'n/a'
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
        s['g'] = float(r[COL_G]) if COL_G in df.columns else float('nan')
        rows.append(s)
    if not rows:
        return None
    sdobs = np.array([x['sdob'] for x in rows])
    band_counts = {}
    for name, _, _ in SDOB_BANDS:
        band_counts[name] = sum(1 for x in rows if x['band'] == name)
    gs = np.array([x['g'] for x in rows])
    corr = float(np.corrcoef(sdobs, gs)[0, 1]) if len(rows) > 2 else float('nan')
    return dict(rows=rows, n=len(rows), smin=float(sdobs.min()), smax=float(sdobs.max()),
                smed=float(np.median(sdobs)), band_counts=band_counts, corr_g=corr)


    weight = cal['weight']
    A = {p: weight[p] for p in ALL_PARAMS}
    B, levels, g_levels = {}, {}, {}
    for p in ALL_PARAMS:
        lv = sorted(cal['real_factor'][p], key=float) if p in PARAMS_NUM else sorted(cal['real_factor'][p], key=str)
        gl = []
        for val in lv:
            mod = dict(inp_row)
            mod[p] = val
            gl.append(median_g(mod, cal, mean_wf))
        levels[p], g_levels[p] = lv, gl
        B[p] = max(gl) - min(gl)
    C = {}
    for p in ALL_PARAMS:
        vals = list(cal['real_factor'][p].values())
        frs  = [cal['freq'][p][kk] for kk in cal['real_factor'][p].keys()]
        m    = sum(v * f for v, f in zip(vals, frs))
        var  = sum(f * (v - m) ** 2 for v, f in zip(vals, frs))
        C[p] = (weight[p] ** 2) * var

    def pct(d):
        s = sum(d.values())
        return {p: (d[p] / s * 100 if s > 0 else 0.0) for p in d}

    return dict(A=pct(A), B=pct(B), C=pct(C), levels=levels, g_levels=g_levels)

def recommend_design(inp, cal, mean_wf, df, design_lambda, target=G_TARGET):
    """Jika DESIGN g (median x lambda) > target, beri rekomendasi perubahan desain."""
    g_med = median_g_event(inp, cal, mean_wf)
    g0    = g_med * design_lambda
    print(SEP)
    print('   RECOMMENDATION ENGINE - PENGAMBILAN KEPUTUSAN')
    print(SEP)
    print('  Median g  = ' + format(g_med, '.6f') + '  |  DESIGN g (x' + format(design_lambda, '.2f') + ') = ' + format(g0, '.6f'))
    print('  Target aman <= ' + format(target, '.4f') + '  (keputusan geotek pakai DESIGN g)')

    if g0 <= target:
        print('  STATUS: AMAN. Tidak perlu perubahan desain.')
        print('')
        return

    print('  STATUS: DI ATAS AMBANG -> perlu mitigasi. Saran (urut prioritas):')
    print('')

    # --- LANGKAH 1: tweak operasional ---
    print('  [LANGKAH 1] Ubah parameter operasional ke level paling aman (data aktual):')
    work = dict(inp)
    cand = []
    for p in CONTROLLABLE_OPS:
        rf = cal['real_factor'][p]
        safe = min(rf, key=lambda kk: rf[kk])
        cur_rf = _nearest_num(rf, inp[p]) if p in PARAMS_NUM else rf.get(inp[p], 1.0)
        if rf[safe] < cur_rf - 1e-12:
            cand.append((p, inp[p], safe, cal['counts'][p][safe]))
    if not cand:
        print('     (tidak ada perubahan operasional yang menurunkan g)')
    else:
        for p, cur, safe, ns in cand:
            work[p] = safe
            note = '  [data tipis: n=' + str(ns) + ']' if ns < MIN_SAMPLES else ''
            print('     - ' + NICE[p] + ': ' + str(cur) + ' -> ' + str(safe) + note)
    g_ops_design = median_g_event(work, cal, mean_wf) * design_lambda
    print('     => DESIGN g setelah tweak operasional: ' + format(g_ops_design, '.6f'))
    print('')

    # --- LANGKAH 2: kurangi charge (g ~ charge^(n/2)) ---
    print('  [LANGKAH 2] Kurangi maximum charge per delay (lever paling ampuh):')
    n = cal['n']
    if g_ops_design > target:
        ch_now = work['charge_kg']
        ch_target = ch_now * (target / g_ops_design) ** (2.0 / n)
        ch_min_data = float(df[COL_CHG].min())
        print('     - Charge: ' + str(round(inp['charge_kg'], 1)) + ' kg -> ' + format(ch_target, '.1f') + ' kg (turun ' + format((1 - ch_target / inp['charge_kg']) * 100, '.0f') + '%)')
        chk = dict(work); chk['charge_kg'] = ch_target
        print('       cek DESIGN g: ' + format(median_g_event(chk, cal, mean_wf) * design_lambda, '.6f'))
        if ch_target < ch_min_data:
            print('       [!] charge target ' + format(ch_target, '.1f') + ' kg di BAWAH rentang data (min ' + format(ch_min_data, '.0f') + ' kg) -> ekstrapolasi, kurang andal')
        print('     CATATAN: penurunan charge memperbaiki getaran TAPI berpotensi')
        print('              memperburuk fragmentasi/produktivitas (trade-off geotek vs mining).')
    else:
        print('     Tidak perlu - target sudah tercapai hanya dgn tweak operasional.')
    print('')

    print('  [LANGKAH 3] Catatan keputusan:')
    print('     - distance, geologi, row number, elevasi titik ukur = TIDAK bisa diubah.')
    print('     - Jika charge target tidak realistis, pertimbangkan: pre-split,')
    print('       pembagian deck/delay lebih halus, atau penjadwalan ulang.')

    final = dict(work)
    if g_ops_design > target:
        final['charge_kg'] = max(ch_target, ch_min_data)
    print('')
    print('  PROYEKSI bila saran diterapkan (charge >= batas data):')
    print('     median g = ' + format(median_g_event(final, cal, mean_wf), '.6f'))
    print('     DESIGN g = ' + format(median_g_event(final, cal, mean_wf) * design_lambda, '.6f'))
    print('')

# =====================================================================
# [C] KALIBRASI
# =====================================================================

print(''); print('' + SEP)
print('   AUTO-KALIBRASI DARI DATA AKTUAL')
print(SEP)

df  = load_and_validate(DATA_FILE, SHEET_NAME)
cal = calibrate(df)
df  = cal['df']
mean_wf = float(np.mean([compute_wf(r, cal) for _, r in df.iterrows()]))
n_data  = len(df)

# Validasi LOOCV + faktor konservatisme (lambda) + sigma LOOCV
pairs   = loocv_predictions(df)
mae_loo = float(np.mean([abs(a - p) for a, p in pairs]))
dl      = design_limits(pairs)
sigma_mc      = dl['s_e'] if USE_LOOCV_SIGMA else cal['sigma']
design_lambda = dl['table'][DESIGN_PCT]['emp' if DESIGN_METHOD == 'empirical' else 'norm']
fdiag   = frequency_diagnostics(df)
sdiag   = sdob_diagnostics(df)

print(''); print('  File        : ' + str(DATA_FILE))
print('  Jumlah data : ' + str(n_data) + ' event')
print('  Range g     : ' + format(df[COL_G].min(), '.5f') + ' - ' + format(df[COL_G].max(), '.5f'))
print('  Metode WF   : ' + REAL_FACTOR_METHOD + ' (DISARANKAN: geometric)')

print(''); print('  [REGRESI ln(Amaks) vs ln(SD)]')
print('  k = ' + format(cal['k'], '.4f') + '   n = ' + format(cal['n'], '.6f') + '   sigma_in = ' + format(cal['sigma'], '.6f') + '   sigma_LOOCV = ' + format(dl['s_e'], '.6f') + '   R2 = ' + format(cal['r2'], '.4f'))

print(''); print('  [REAL FACTOR - ' + REAL_FACTOR_METHOD + ' mean residual ratio per kategori]')
for p in ALL_PARAMS:
    items = ', '.join(str(kk) + ':' + format(vv, '.3f') for kk, vv in sorted(cal['real_factor'][p].items(), key=lambda x: str(x[0])))
    print('  ' + p.ljust(22) + ': ' + items)

# Warning kategori tipis
thin = []
print(''); print('  [JUMLAH SAMPEL PER KATEGORI]  (ambang tipis: < ' + str(MIN_SAMPLES) + ')')
for p in PARAMS_CAT:
    parts = []
    for kk, c in sorted(cal['counts'][p].items(), key=lambda x: str(x[0])):
        flag = ' (!)' if c < MIN_SAMPLES else ''
        parts.append(str(kk) + ':' + str(c) + flag)
        if c < MIN_SAMPLES:
            thin.append(str(p) + '=' + str(kk) + ' (n=' + str(c) + ')')
    print('  ' + p.ljust(22) + ': ' + ', '.join(parts))
if thin:
    print('  [!] Kategori tipis (real_factor kurang andal, perlu data tambahan): ' + '; '.join(thin))

# Validasi jujur (in-sample sbg pembanding; LOOCV sbg laporan utama)
g_val = (cal['k'] * df['scale_distance'].values ** (-cal['n']) * (df.apply(lambda r: compute_wf(r, cal), axis=1).values / mean_wf)) / G_GRAV
mae_in  = float(np.mean(np.abs(df[COL_G].values - g_val)))
corr_in = float(np.corrcoef(df[COL_G].values, g_val)[0, 1])

print(''); print('  MEAN_WF_AKTUAL = ' + format(mean_wf, '.6f'))
print('  [VALIDASI] In-sample MAE = ' + format(mae_in, '.6f') + ' (r=' + format(corr_in, '.4f') + ')  |  LOOCV MAE = ' + format(mae_loo, '.6f') + ' <- error jujur out-of-sample (UTAMA)')

# --- Faktor konservatisme (lambda) ---
print(''); print('  [FAKTOR KONSERVATISME lambda - dari sebaran rasio aktual/pred LOOCV]')
print('  Persentil  lambda_emp  underpred_emp%   lambda_norm  underpred_norm%')
for pct in (75, 90, 95):
    t = dl['table'][pct]
    print('   P' + str(pct).ljust(4) + format(t['emp'], '.2f').rjust(9) + format(t['under_emp'], '.0f').rjust(14) + format(t['norm'], '.2f').rjust(14) + format(t['under_norm'], '.0f').rjust(15))
print('  Dipakai (DISARANKAN): P' + str(DESIGN_PCT) + ' ' + DESIGN_METHOD + ' -> lambda = ' + format(design_lambda, '.3f'))
print('  CATATAN setelan: geometric + empirical P90 = konfigurasi disarankan untuk keputusan.')
print('  "arithmetic"/"normal" hanya untuk uji sensitivitas / cross-check, bukan produksi.')

# =====================================================================
# [D] PREDIKSI EVENT BARU
# =====================================================================

def check_extrapolation(inp, df):
    warns = []
    for p in PARAMS_NUM:
        lo, hi = df[p].min(), df[p].max()
        if inp[p] < lo or inp[p] > hi:
            warns.append(str(p) + '=' + str(inp[p]) + ' di luar rentang [' + str(lo) + '-' + str(hi) + ']')
    for p in PARAMS_CAT:
        if inp[p] not in set(df[p].unique()):
            warns.append(str(p) + "='" + str(inp[p]) + "' tidak ada di data aktual")
    for col, key in [(COL_CHG, 'charge_kg'), (COL_DIST, 'distance_m')]:
        lo, hi = df[col].min(), df[col].max()
        if not (lo <= inp[key] <= hi):
            warns.append(str(key) + '=' + str(inp[key]) + ' di luar rentang [' + str(lo) + '-' + str(hi) + ']')
    return warns

INPUT['scale_distance'] = INPUT['distance_m'] / np.sqrt(INPUT['charge_kg'])
INPUT['sd']             = INPUT['scale_distance']
wf_raw_input  = compute_wf(INPUT, cal)
wf_norm_input = wf_raw_input / mean_wf

g_arr  = predict_g_mc(cal, INPUT['scale_distance'], wf_norm_input, sigma_mc)
g_pred = float(np.median(g_arr))
g_q1   = float(np.percentile(g_arr, 25))
g_q3   = float(np.percentile(g_arr, 75))
g_best  = float(np.percentile(g_arr, 10))
g_worst = float(np.percentile(g_arr, 90))
g_design = g_pred * design_lambda
prob_exceed = 100.0 * float(np.mean(g_arr > G_TARGET))   # probabilitas melampaui ambang
warns = check_extrapolation(INPUT, df)

print(''); print('' + SEP)
print('   PREDIKSI GROUND VIBRATION (g) - BLASTING PIT 3')
print('   Data kalibrasi: ' + str(n_data) + ' event  |  Iterasi: ' + format(N_ITER, ','))
print(SEP)
print(''); print('  Distance ' + str(INPUT['distance_m']) + ' m | Charge ' + str(INPUT['charge_kg']) + ' kg | SD ' + format(INPUT['scale_distance'], '.3f') + ' | Row ' + str(INPUT['row_number']))
print('  Geological ' + str(INPUT['geological_condt']) + ' | Elev ' + str(INPUT['measuring_elevation']) + ' | Tie-up ' + str(INPUT['tie_up_type']))
print('  Control ' + str(INPUT['controll_ms']) + ' ms | Wall echelon ' + str(INPUT['wall_echelon_ms']) + ' ms | Freeface echelon ' + str(INPUT['freeface_echelon_ms']) + ' ms | Freeface count ' + str(INPUT['freeface_count']))
print(''); print('  WF raw = ' + format(wf_raw_input, '.4f') + ' | WF normalized = ' + format(wf_norm_input, '.4f') + ' (' + ('lebih berbahaya' if wf_norm_input > 1 else 'lebih aman') + ' dari rata-rata site)')
if warns:
    print(''); print('  [!] PERINGATAN EKSTRAPOLASI (prediksi kurang andal):')
    for w in warns:
        print('      - ' + w)
if INPUT['distance_m'] < 100 and str(INPUT['geological_condt']) == 'Fault':
    print(''); print('  [!] REZIM Fault + near-field (<100 m): model cenderung MEREMEHKAN -> beri buffer ekstra.')

print(''); print('' + SEP2)
print('   HASIL PREDIKSI - ' + format(N_ITER, ',') + ' SIMULASI MONTE CARLO')
print(SEP2)
print('')
print('  Prediksi terbaik  (P10)     :  ' + str(round(g_best, 6)) + '      Status: ' + get_status(g_best))
print('  Paling mungkin    (P50)     :  ' + str(round(g_pred, 6)) + '      Status: ' + get_status(g_pred))
print('  Prediksi terburuk (P90)     :  ' + str(round(g_worst, 6)) + '      Status: ' + get_status(g_worst))
print('  PROBABILITAS g > ' + format(G_TARGET, '.3f') + '     :  ' + format(prob_exceed, '.1f') + '%   <- ukuran risiko paling konkret')
print('  DESIGN VALUE P' + str(DESIGN_PCT) + ' (x' + format(design_lambda, '.2f') + ') :  ' + str(round(g_design, 6)) + '      Status: ' + get_status(g_design))
print('  (sigma MC = ' + format(sigma_mc, '.3f') + ', ' + ('LOOCV' if USE_LOOCV_SIGMA else 'in-sample') + ')')
print('')
print(SEP2)
print('   RINGKASAN UNTUK RAPAT')
print(SEP2)
print('')
print('  Rentang P10-P90 (80% kemungkinan): ' + str(round(g_best, 6)) + ' - ' + str(round(g_worst, 6)))
print('  Probabilitas melampaui ambang ' + format(G_TARGET, '.3f') + ' = ' + format(prob_exceed, '.1f') + '%')
print('  Acuan keputusan keselamatan: P90 = ' + str(round(g_worst, 6)) + ' atau DESIGN = ' + str(round(g_design, 6)))
print('  Perkiraan error model (LOOCV): +/- ' + format(mae_loo, '.4f'))
print('  CATATAN: P50 adalah nilai tengah; menurut sifat median ~separuh kejadian aktual')
print('           dapat berada di atasnya. Untuk Fault/near-field, pakai P90/DESIGN sbg acuan.')
print('')

# --- RECOMMENDATION ENGINE: aktif jika DESIGN g melebihi ambang aman ---
recommend_design(INPUT, cal, mean_wf, df, design_lambda, target=G_TARGET)

# =====================================================================
# [E] SENSITIVITY ANALYSIS - VARIASI CHARGE
# =====================================================================

print(SEP)
print('   SENSITIVITY ANALYSIS - VARIASI CHARGE (Distance tetap ' + str(INPUT['distance_m']) + ' m)')
print(SEP)
print(''); print('  ' + 'Charge'.rjust(6) + ' | ' + 'SD'.rjust(8) + ' | ' + 'Median g'.rjust(11) + ' | ' + 'DESIGN g'.rjust(11) + ' | Status(design)')
print('  ' + '-' * 6 + '-+-' + '-' * 8 + '-+-' + '-' * 11 + '-+-' + '-' * 11 + '-+-' + '-' * 15)
for ch in [35, 40, 45, 50, 55, 60, 65, 70]:
    sd_ch = INPUT['distance_m'] / np.sqrt(ch)
    g_ch  = predict_g_mc(cal, sd_ch, wf_norm_input, sigma_mc, n_iter=5000, seed=SEED)
    gp = float(np.median(g_ch))
    gdz = gp * design_lambda
    mark = '  <- BASE' if ch == INPUT['charge_kg'] else ''
    print('  ' + str(ch).rjust(6) + ' | ' + format(sd_ch, '.3f').rjust(8) + ' | ' + str(round(gp, 6)).rjust(11) + ' | ' + str(round(gdz, 6)).rjust(11) + ' | ' + get_status(gdz) + mark)

# =====================================================================
# [F] KEPENTINGAN FAKTOR - 3 METRIK (untuk paper)
# =====================================================================

imp = importance_metrics(cal, INPUT, mean_wf)
order = sorted(ALL_PARAMS, key=lambda p: -imp['C'][p])

print(''); print('' + SEP)
print('   KEPENTINGAN FAKTOR - 3 METRIK (definisi eksplisit utk paper)')
print(SEP)
print('  (A) Weight model  : w_p ~ range_p           - bobot pencampur internal')
print('  (B) OAT g-range   : ~ range_p^2             - sensitivitas LOKAL 1 event')
print('  (C) Sobol orde-1  : S_p ~ w_p^2 * Var(f_p)  - variance-based, GLOBAL  [UTAMA]')
print(''); print('  ' + 'Faktor'.ljust(24) + '(A) Weight'.rjust(11) + '(B) OAT'.rjust(10) + '(C) Sobol'.rjust(11))
print('  ' + '-' * 24 + '-' * 11 + '-' * 10 + '-' * 11)
for p in order:
    print('  ' + NICE[p].ljust(24) + (format(imp['A'][p], '.1f') + '%').rjust(11) + (format(imp['B'][p], '.1f') + '%').rjust(10) + (format(imp['C'][p], '.1f') + '%').rjust(11))
print(''); print('  -> Untuk paper, laporkan kolom (C) Sobol sebagai kontribusi utama.')

# =====================================================================
# [G] DIAGNOSTIK RESONANSI (frekuensi = AUDIT, BUKAN prediksi)
# =====================================================================

if fdiag is not None:
    print(''); print('' + SEP)
    print('   DIAGNOSTIK RESONANSI - frekuensi sbg AUDIT pita bahaya (BUKAN prediksi)')
    print(SEP)
    print('  Frekuensi dominan TIDAK dapat diprediksi dari scale-distance (R2 ~ 0.002,')
    print('  praktis nol) maupun parameter desain -> diposisikan sebagai audit pita bahaya.')
    print(''); print('  Frekuensi dominan: min ' + format(fdiag['fmin'], '.1f') + ' | median ' + format(fdiag['fmed'], '.1f') + ' | maks ' + format(fdiag['fmax'], '.1f') + ' Hz')
    print('  Distribusi pita Floyd (2008):')
    for b in fdiag['bands']:
        print('    ' + b['name'].ljust(18) + str(b['count']) + ' event (' + format(b['pct'], '.0f') + '%)')
    print('  ' + str(fdiag['n_slope']) + '/' + str(fdiag['n']) + ' event < ' + format(SLOPE_FREQ_THRESHOLD, '.0f') + ' Hz (zona resonansi lereng).')
    print('  Audit resonansi per geologi (freq event aktual vs natural freq batuan):')
    for geo, a in fdiag['geo_audit'].items():
        rng_txt = (str(a['rng'][0]) + '-' + str(a['rng'][1]) + ' Hz') if a['rng'] else '(tidak dipetakan)'
        print('    ' + str(geo).ljust(8) + 'rock=' + str(a['rock']).ljust(8) + 'nat=' + rng_txt.ljust(10) + 'n=' + str(a['n']) + ' | freq ' + format(a['fmin'], '.1f') + '-' + format(a['fmax'], '.1f') + ' | di zona resonansi: ' + str(a['in_res']))
    print('  Catatan: freq tergantung geologi/jarak/delay (non-linear; Lucca 2003);')
    print('  R2~0 terhadap SD adalah wajar. Pustaka: Floyd 2008, Kumar 2020, Hasanipanah 2015.')

# =====================================================================
# [G2] DIAGNOSTIK SDOB (Scaled Depth of Burial) - ERG/Tobin 2013
# =====================================================================

s_in = compute_sdob(INPUT['charge_kg'], INPUT['depth_m'], INPUT['hole_diameter_mm'])
if s_in is not None:
    print(''); print('' + SEP)
    print('   DIAGNOSTIK SDOB - konfinemen energi (ERG/Tobin 2013), bukan prediktor g')
    print(SEP)
    print('  SD = (stemming + 5d) / W10^(1/3) ; W10 = massa peledak dlm 10x diameter.')
    print('  Densitas emulsi = ' + format(RHO_EXPLOSIVE, '.0f') + ' kg/m3, diameter = ' + format(INPUT['hole_diameter_mm'], '.0f') + ' mm.')
    print(''); print('  Event input (charge ' + str(INPUT['charge_kg']) + ' kg @ kedalaman ' + format(INPUT['depth_m'], '.2f') + ' m):')
    print('    panjang isian = ' + format(s_in['charge_len'], '.2f') + ' m | stemming (dihitung) = ' + format(s_in['stemming'], '.2f') + ' m | stem/dia = ' + format(s_in['stem_over_dia'], '.1f'))
    print('    SDOB = ' + format(s_in['sdob'], '.3f') + '  -> ' + s_in['band'])
if sdiag is not None:
    print(''); print('  Distribusi konfinemen ' + str(sdiag['n']) + ' event aktual:')
    for name, _, _ in SDOB_BANDS:
        c = sdiag['band_counts'].get(name, 0)
        print('    ' + name.split(':')[0].ljust(22) + ': ' + str(c) + ' event (' + format(100.0 * c / sdiag['n'], '.0f') + '%)')
    print('  SDOB aktual: min ' + format(sdiag['smin'], '.2f') + ' | median ' + format(sdiag['smed'], '.2f') + ' | maks ' + format(sdiag['smax'], '.2f'))
    print('  Korelasi SDOB vs g: r = ' + format(sdiag['corr_g'], '.3f') + ' (lemah/rancu -> g jg dipengaruhi jarak & charge;')
    print('    SDOB TIDAK dipakai sbg prediktor g, hanya diagnostik konfinemen).')
    print('  ARGUMEN GEOTEK: desain di zona controlled s/d minimal-surface (tdk ada uncontrolled/cratering)')
    print('    -> secara teori SDOB Chiappetta, desain proper: risiko flyrock/airblast minimal, energi ke batuan.')
    print('  Tabel SDOB per event:')
    print('    #   SDOB   stemming  stem/dia  L_isian   nilai_g   regime')
    for i, x in enumerate(sdiag['rows']):
        print('    ' + str(i + 1).rjust(2) + format(x['sdob'], '.2f').rjust(7) + format(x['stemming'], '.2f').rjust(9)
              + format(x['stem_over_dia'], '.1f').rjust(9) + format(x['charge_len'], '.2f').rjust(9)
              + format(x['g'], '.4f').rjust(10) + '   ' + x['band'].split(':')[0])
    print('  Pustaka: Tobin (2013); Ash (1993); Langefors & Kihlstrom (1978).')

# =====================================================================
# [H] VISUALISASI - BAR CHART (Sobol) + HEATMAP
# =====================================================================

if SHOW_PLOTS:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    levels, g_levels = imp['levels'], imp['g_levels']

    def active_level(p):
        if p in PARAMS_NUM:
            keys = np.array([float(x) for x in levels[p]])
            return levels[p][int(np.abs(keys - float(INPUT[p])).argmin())]
        return INPUT[p]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 13), gridspec_kw={'height_ratios': [1, 1.25]})
    fig.suptitle('Analisis Faktor Penentu Nilai g - Blasting Pit 3', fontsize=15, fontweight='bold', y=0.985)

    labels = [NICE[p] + '  [' + str(active_level(p)) + ']' for p in order]
    vals   = [imp['C'][p] for p in order]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(order)))
    bars   = ax1.barh(range(len(order)), vals, color=colors, edgecolor='black', linewidth=0.6)
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('Kontribusi terhadap varians nilai g - indeks Sobol orde-1 (%)', fontsize=11)
    ax1.set_title('Kontribusi Faktor (GLOBAL)   median g input = ' + format(median_g_event(INPUT, cal, mean_wf), '.5f') + '   |  [ ] = kondisi input aktif', fontsize=11, pad=8)
    for b, v in zip(bars, vals):
        ax1.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, format(v, '.1f') + '%', va='center', fontsize=9, fontweight='bold')
    ax1.set_xlim(0, max(vals) * 1.15)
    ax1.grid(axis='x', alpha=0.3)

    max_lv = max(len(levels[p]) for p in order)
    mat = np.full((len(order), max_lv), np.nan)
    for i, p in enumerate(order):
        for j, g in enumerate(g_levels[p]):
            mat[i, j] = g
    cmap2 = plt.cm.RdYlGn_r.copy()
    cmap2.set_bad('white')
    im = ax2.imshow(np.ma.masked_invalid(mat), aspect='auto', cmap=cmap2)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([NICE[p] for p in order], fontsize=10)
    ax2.set_xticks(range(max_lv))
    ax2.set_xticklabels(['Lvl ' + str(j + 1) for j in range(max_lv)], fontsize=9)
    ax2.set_title('Heatmap Sensitivitas - prediksi g tiap level (merah=tinggi/berisiko, hijau=rendah/aman, kotak biru=kondisi input)', fontsize=10, pad=8)
    for i, p in enumerate(order):
        act = active_level(p)
        for j, val in enumerate(levels[p]):
            ax2.text(j, i - 0.18, str(val), ha='center', va='center', fontsize=8, fontweight='bold', color='black')
            ax2.text(j, i + 0.22, format(g_levels[p][j], '.4f'), ha='center', va='center', fontsize=7.5, color='black')
            if val == act:
                ax2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2.5))
    cbar = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cbar.set_label('Prediksi nilai g', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.show()

print(''); print('' + SEP)
print('   PARAMETER KALIBRASI (auto-computed dari ' + str(n_data) + ' data aktual)')
print(SEP)
print('  k = ' + format(cal['k'], '.4f') + ' | n = ' + format(cal['n'], '.6f') + ' | sigma_in = ' + format(cal['sigma'], '.6f') + ' | sigma_LOOCV = ' + format(dl['s_e'], '.6f'))
print('  MEAN_WF = ' + format(mean_wf, '.6f') + ' | R2 = ' + format(cal['r2'], '.4f'))
print('  MAE in-sample = ' + format(mae_in, '.6f') + ' | MAE LOOCV = ' + format(mae_loo, '.6f'))
print('  lambda P' + str(DESIGN_PCT) + ' (' + DESIGN_METHOD + ') = ' + format(design_lambda, '.3f'))
print('  Semua parameter dihitung otomatis dari file: ' + str(DATA_FILE))
print('')
