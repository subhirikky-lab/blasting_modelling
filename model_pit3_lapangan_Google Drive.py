# =====================================================================
# APLIKASI PREDIKSI GETARAN BLASTING - PIT 3          (model v5)
# =====================================================================
# Mesin model:
#   1. Atenuasi per geologi   : A = k_geo * SD^(-n_geo)  (Fault, coal, normal)
#   2. Faktor jumlah baris    : pengali langsung per kelas baris (1-3, 4-7, 8-9)
#   3. Weighting factor       : faktor operasional lain (tie-up, delay, freeface,
#                               elevasi titik ukur), rata-rata berbobot
#   4. Monte Carlo + design value (lambda P90) dari validasi per-event
#
# Data diambil dari Google Drive (Sheet1). Hanya blast dalam batas kalibrasi
# yang dipakai: baris <= 9 dan jarak >= 86 m.
#
# Deploy: Streamlit Cloud. requirements.txt minimal:
#   streamlit, pandas, numpy, scipy, plotly, openpyxl, requests
# =====================================================================

import io
import os
import math
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy import stats
import plotly.graph_objects as go

# =====================================================================
# [A] KONFIGURASI
# =====================================================================
FILE_ID = '1xwJz5xrysYvEPK7lZMweL5rAAbBFkNA1'
DOWNLOAD_URLS = [   # dicoba berurutan: Google Sheets, lalu file .xlsx biasa di Drive
    f'https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=xlsx',
    f'https://drive.google.com/uc?export=download&id={FILE_ID}',
]
SHEET_NAME = 'Sheet1'
REFRESH_MIN = 5                                   # data Drive diambil ulang tiap n menit
LOCAL_DATA = os.environ.get('PIT3_LOCAL_DATA')   # opsional: path file lokal untuk uji

# Urutan kolom template upload (sama dengan sheet di Drive)
TEMPLATE_COLS = ['Tanggal', 'PPA (mm/s^s) Trans', 'PPA (mm/s^s) Long', 'PPA (mm/s^s) Maks',
                 'Amaks (mm/s^s) Trans', 'Amaks (mm/s^s) Long', 'Amaks (mm/s^s) Maks', 'nilai_g',
                 'charge_kg', 'distance_m', 'hole_diameter_mm', 'burden_m', 'spacing_m', 'depth_m',
                 'tie_up_type', 'wall_echelon_ms', 'freeface_echelon_ms', 'controll_ms',
                 'freeface_count', 'geological_condt', 'measuring_elevation', 'row_number', 'frekuensi_hz']

G_TARGET = 0.030            # batas aman (k_h dari geoteknik)
MAX_ROWS = 9                # batas kalibrasi
MIN_DIST = 86               # batas kalibrasi (m)
MIN_SAMPLES = 5             # level dengan data < ini dianggap tipis
MIN_GEO_FOR_OWN_N = 8       # geologi dengan data < ini memakai n bersama
DESIGN_PCT = 90             # persentil design value
N_ITER = 10000
SEED = 42
G_GRAV = 9806.65            # mm/s^2

ROW_AS_MULTIPLIER = True    # True: jumlah baris = pengali langsung (v5)
LAMBDA_PER_GEOLOGI = False  # True: lambda & sigma per geologi

COL_AMAKS, COL_G, COL_DIST, COL_CHG = 'Amaks (mm/s^s) Maks', 'nilai_g', 'distance_m', 'charge_kg'
GEO = 'geological_condt'
REQUIRED = [COL_AMAKS, COL_G, COL_DIST, COL_CHG, GEO, 'row_number', 'tie_up_type',
            'measuring_elevation', 'controll_ms', 'wall_echelon_ms',
            'freeface_echelon_ms', 'freeface_count']
OPS_CAT = ['tie_up_type', 'measuring_elevation', 'ffe_binned']
OPS_NUM = ['controll_binned', 'wall_echelon_ms', 'freeface_count']
ROW_CLASSES = ['1-3', '4-7', '8-9']

NICE = {
    GEO: 'Geologi', 'row_class': 'Jumlah baris', 'tie_up_type': 'Tie-up',
    'measuring_elevation': 'Elevasi titik ukur', 'ffe_binned': 'Freeface echelon (ms)',
    'controll_binned': 'Control delay (ms)', 'wall_echelon_ms': 'Wall echelon (ms)',
    'freeface_count': 'Freeface count',
}

# Faktor yang bisa dilihat di tab Data Historis: label -> (kolom, urutan level)
HIST_FACTORS = {
    'Geologi': (GEO, ['Fault', 'coal', 'normal']),
    'Jumlah baris': ('row_class', ROW_CLASSES),
    'Freeface count': ('freeface_count', None),
    'Wall echelon (ms)': ('wall_echelon_ms', None),
    'Freeface echelon (ms)': ('ffe_binned', ['0', '42-67', '>=109']),
    'Control delay (ms)': ('controll_binned', None),
    'Tie-up': ('tie_up_type', ['boxcut', 'echelon']),
    'Elevasi titik ukur': ('measuring_elevation', ['lower', 'normal', 'higher']),
}

# Parameter yang masih bisa diubah setelah lubang dibor:
# (kunci input, parameter model, fungsi level-model -> nilai input)
CONTROLLABLE = [
    ('tie_up_type', 'tie_up_type', lambda v: v),
    ('controll_ms', 'controll_binned', lambda v: int(float(v))),
    ('wall_echelon_ms', 'wall_echelon_ms', lambda v: int(float(v))),
    ('freeface_echelon_ms', 'ffe_binned', lambda v: {'0': 0, '42-67': 42, '>=109': 109}[v]),
    ('freeface_count', 'freeface_count', lambda v: int(float(v))),
]

# Warna (palet kategorikal tervalidasi + bentuk marker sebagai penanda kedua)
# amber · teal · plum · rust · indigo — lolos uji keterbacaan & buta warna (terang & gelap)
PALETTE = ['#c27c0e', '#008a7e', '#8e4fa0', '#c4532f', '#4a5fc1']
ACCENT = PALETTE[0]          # warna utama (amber)
DIV_LOW, DIV_MID, DIV_HIGH = '#008a7e', '#ebe7df', '#c4532f'   # di bawah batas · sekitar · di atas
SYMBOLS = ['circle', 'square', 'triangle-up', 'diamond', 'triangle-down']
GEO_COLOR = {'Fault': PALETTE[0], 'coal': PALETTE[1], 'normal': PALETTE[2]}
GEO_SYMBOL = {'Fault': SYMBOLS[0], 'coal': SYMBOLS[1], 'normal': SYMBOLS[2]}
MUTED = 'rgba(128,128,128,0.9)'
GRIDC = 'rgba(128,128,128,0.18)'

# Status: (batas atas, label, warna, ikon, warna teks badge)
STATUS = [
    (0.020, 'EXCELLENT', '#0ca30c', '✅', '#ffffff'),
    (0.030, 'SAFE', '#0ca30c', '✅', '#ffffff'),
    (0.100, 'MODERATE', '#fab219', '⚠️', '#0b0b0b'),
    (0.200, 'RISKY', '#ec835a', '🔶', '#0b0b0b'),
    (float('inf'), 'EXTREMELY RISKY', '#d03b3b', '⛔', '#ffffff'),
]

DELAY_OPTIONS = [0, 17, 25, 42, 67, 109, 176]


# =====================================================================
# [B] TAMPILAN DASAR
# =====================================================================
st.set_page_config(page_title='Prediksi Getaran Pit 3', page_icon='💥', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.kpi {border-radius: 12px; padding: 14px 16px 12px; background: rgba(128,128,128,0.08);
      border-left: 6px solid var(--c); height: 100%;}
.kpi .lbl {font-size: 0.80rem; opacity: 0.75; margin-bottom: 2px; text-transform: uppercase;
           letter-spacing: 0.03em;}
.kpi .val {font-size: 1.85rem; font-weight: 700; line-height: 1.15;}
.kpi .sub {font-size: 0.84rem; opacity: 0.85; margin-top: 6px;}
.badge {display: inline-block; padding: 1px 10px; border-radius: 999px; font-size: 0.74rem;
        font-weight: 700; background: var(--c); color: var(--t); margin-left: 6px;
        vertical-align: middle;}
.banner {border-radius: 12px; padding: 12px 18px; margin: 14px 0 6px;
         background: rgba(128,128,128,0.08); border-left: 6px solid var(--c);}
.banner b {font-size: 1.02rem;}
.step {border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
       background: rgba(128,128,128,0.07);}
.muted {opacity: 0.75; font-size: 0.86rem;}
/* aksen amber untuk tab aktif (berlaku juga tanpa config.toml) */
div[role="tab"][aria-selected="true"], button[role="tab"][aria-selected="true"] {color: #c27c0e !important;}
div[role="tab"][aria-selected="true"] p, button[role="tab"][aria-selected="true"] p {color: #c27c0e !important;}
[data-baseweb="tab-highlight"], [data-testid="stTabHighlight"] {background-color: #c27c0e !important;}
h1 {letter-spacing: -0.01em;}
h3 {border-bottom: 2px solid rgba(194,124,14,0.35); padding-bottom: 4px;}
</style>
""", unsafe_allow_html=True)


def status_of(g):
    for thr, label, color, icon, tcol in STATUS:
        if g <= thr:
            return dict(label=label, color=color, icon=icon, tcol=tcol)
    return dict(label='EXTREMELY RISKY', color='#d03b3b', icon='⛔', tcol='#ffffff')


def kpi(label, value, sub='', g_for_status=None, color=ACCENT):
    badge = ''
    if g_for_status is not None:
        s = status_of(g_for_status)
        color = s['color']
        badge = '<span class="badge" style="--c:%s;--t:%s">%s %s</span>' % (
            s['color'], s['tcol'], s['icon'], s['label'])
    st.markdown('<div class="kpi" style="--c:%s"><div class="lbl">%s</div>'
                '<div class="val">%s%s</div><div class="sub">%s</div></div>'
                % (color, label, value, badge, sub), unsafe_allow_html=True)


def show_chart(fig):
    try:
        st.plotly_chart(fig, width='stretch', theme='streamlit')
    except (TypeError, st.errors.StreamlitAPIException):
        st.plotly_chart(fig, use_container_width=True, theme='streamlit')


def show_table(df):
    try:
        st.dataframe(df, width='stretch', hide_index=True)
    except (TypeError, st.errors.StreamlitAPIException):
        st.dataframe(df, use_container_width=True, hide_index=True)


def style_fig(fig, title=None, height=420, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=56 if title else 20, b=90 if legend else 10),
        title=dict(text=title, x=0, xanchor='left', font=dict(size=15)) if title else None,
        showlegend=legend,
        legend=dict(orientation='h', yanchor='top', y=-0.16, x=0, xanchor='left', title=None),
        hoverlabel=dict(font_size=12), font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRIDC, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRIDC, zeroline=False)
    return fig


def ref_hline(fig, y, text, log_y=False, side='left'):
    """Garis acuan horizontal + label. Pada sumbu log, posisi label Plotly
    harus dalam log10 — ditangani di sini."""
    fig.add_hline(y=y, line=dict(color=MUTED, width=1.5, dash='dash'))
    fig.add_annotation(x=0.01 if side == 'left' else 0.99, xref='paper',
                       y=math.log10(y) if log_y else y, yref='y', text=text, showarrow=False,
                       xanchor='left' if side == 'left' else 'right', yanchor='bottom',
                       font=dict(size=11, color=MUTED))


# =====================================================================
# [C] DATA
# =====================================================================
@st.cache_data(ttl=REFRESH_MIN * 60, show_spinner=False)
def fetch_remote(urls):
    """Ambil file Excel dari Google Drive. Return (isi file, waktu ambil WIB)."""
    errors = []
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            if r.content[:2] != b'PK':          # file xlsx selalu diawali 'PK' (zip)
                raise ValueError('bukan file Excel (kemungkinan halaman login/izin)')
            wib = pd.Timestamp.now(tz='Asia/Jakarta').strftime('%d/%m/%Y %H:%M')
            return r.content, wib
        except Exception as e:
            errors.append('%s -> %s' % (url.split('/')[2], e))
    raise RuntimeError('; '.join(errors))


def read_excel_any(src):
    """Baca Sheet1; kalau nama sheet beda, pakai sheet pertama."""
    data = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    xl = pd.ExcelFile(data, engine='openpyxl')
    sheet = SHEET_NAME if SHEET_NAME in xl.sheet_names else xl.sheet_names[0]
    return xl.parse(sheet)


def template_bytes():
    buf = io.BytesIO()
    pd.DataFrame(columns=TEMPLATE_COLS).to_excel(buf, sheet_name=SHEET_NAME, index=False)
    return buf.getvalue()


def parse_dates(series):
    try:
        return pd.to_datetime(series, dayfirst=True, errors='coerce', format='mixed')
    except (TypeError, ValueError):
        return pd.to_datetime(series, dayfirst=True, errors='coerce')


def last_blast_date(d):
    """Tanggal di baris terakhir (data diisi berurutan) + tanggal yang tampak salah ketik."""
    dt = parse_dates(d['tanggal'])
    last = dt.iloc[-1].strftime('%d/%m/%Y') if pd.notna(dt.iloc[-1]) else str(d['tanggal'].iloc[-1])
    today = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None).normalize()
    future = [x.strftime('%d/%m/%Y') for x in dt[dt > today]]
    return last, future


def _date_label(v):
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, 'strftime'):
        try:
            return v.strftime('%d/%m/%y')
        except Exception:
            return str(v)
    return str(v).strip()


@st.cache_data(show_spinner=False)
def prepare(raw):
    df = raw.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError('Kolom wajib tidak ada di data: ' + ', '.join(missing))
    df['tanggal'] = df['Tanggal'].apply(_date_label) if 'Tanggal' in df.columns else ''
    reason = pd.Series('', index=df.index)
    incomplete = df[REQUIRED].isna().any(axis=1)
    reason[incomplete] = 'data tidak lengkap'
    num = df[[COL_AMAKS, COL_G, COL_DIST, COL_CHG]].apply(pd.to_numeric, errors='coerce')
    bad = (~incomplete) & ((num <= 0).any(axis=1) | num.isna().any(axis=1))
    reason[bad] = 'nilai nol / tidak valid'
    ok = reason == ''
    rows = pd.to_numeric(df['row_number'], errors='coerce')
    dist = pd.to_numeric(df[COL_DIST], errors='coerce')
    reason[ok & (rows > MAX_ROWS)] = 'di luar batas: baris > %d' % MAX_ROWS
    reason[ok & (rows <= MAX_ROWS) & (dist < MIN_DIST)] = 'di luar batas: jarak < %d m' % MIN_DIST

    excluded = df[reason != ''].copy()
    excluded['alasan'] = reason[reason != '']
    d = df[reason == ''].copy()
    for c in [COL_AMAKS, COL_G, COL_DIST, COL_CHG, 'row_number', 'controll_ms',
              'wall_echelon_ms', 'freeface_echelon_ms', 'freeface_count']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d[GEO] = d[GEO].astype(str).str.strip()
    d['tie_up_type'] = d['tie_up_type'].astype(str).str.strip()
    d['measuring_elevation'] = d['measuring_elevation'].astype(str).str.strip()
    d['freeface_count'] = d['freeface_count'].replace({0: 1})
    d['sd'] = d[COL_DIST] / np.sqrt(d[COL_CHG])
    d['A'] = d[COL_AMAKS].astype(float)
    d = add_bins(d)
    d['event'] = d['tanggal'].astype(str) + '_' + d[COL_CHG].astype(str)
    return d.reset_index(drop=True), excluded.reset_index(drop=True)


def add_bins(d):
    d = d.copy()
    d['row_class'] = d['row_number'].apply(lambda r: '1-3' if r <= 3 else ('4-7' if r <= 7 else '8-9'))
    d['controll_binned'] = d['controll_ms'].replace({42: 67})
    d['ffe_binned'] = d['freeface_echelon_ms'].apply(
        lambda x: '0' if x == 0 else ('42-67' if x <= 67 else '>=109'))
    return d


# =====================================================================
# [D] MODEL
# =====================================================================
def _gm(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def _nearest(d, t):
    keys = np.array([float(x) for x in d])
    return list(d.values())[int(np.abs(keys - float(t)).argmin())]


def fit_model(d):
    """Kalibrasi lengkap dari data d."""
    sl, ic, r, *_ = stats.linregress(np.log(d['sd']), np.log(d['A']))
    base = {'*': (float(np.exp(ic)), float(-sl), float(r ** 2), len(d))}
    n_common = float(-sl)
    for g, x in d.groupby(GEO):
        if len(x) >= MIN_GEO_FOR_OWN_N:
            s, i, rr, *_ = stats.linregress(np.log(x['sd']), np.log(x['A']))
            base[g] = (float(np.exp(i)), float(-s), float(rr ** 2), len(x))
        else:
            k_g = float(np.exp(np.mean(np.log(x['A']) + n_common * np.log(x['sd']))))
            base[g] = (k_g, n_common, float('nan'), len(x))

    b = np.array([_kn(g, base)[0] * s ** (-_kn(g, base)[1]) for g, s in zip(d[GEO], d['sd'])])
    rr = d['A'].values / b
    rowf, rowc = {}, {}
    if ROW_AS_MULTIPLIER:
        for c, v in pd.Series(rr).groupby(d['row_class'].values):
            rowf[c], rowc[c] = _gm(v.values), len(v)
        rr = rr / np.array([rowf[c] for c in d['row_class']])
    params_cat = OPS_CAT + ([] if ROW_AS_MULTIPLIER else ['row_class'])
    params = params_cat + OPS_NUM
    x = d.assign(rr=rr)
    rf, cnt, frq = {}, {}, {}
    for p in params:
        g = x.groupby(p)['rr']
        rf[p] = {k: _gm(v.values) for k, v in g}
        cnt[p] = {k: len(v) for k, v in g}
        frq[p] = {k: len(v) / len(x) for k, v in g}
    rng = {p: max(v.values()) - min(v.values()) for p, v in rf.items()}
    tot = sum(rng.values())
    w = {p: rng[p] / tot if tot else 0.0 for p in params}
    m = dict(base=base, rowf=rowf, rowc=rowc, rf=rf, cnt=cnt, frq=frq, w=w,
             params=params, params_cat=params_cat)
    m['mean_wf'] = float(np.mean([_wf(row, m) for _, row in d.iterrows()]))
    return m


def _kn(geo, base):
    k, n, *_ = base.get(geo, base['*'])
    return k, n


def _wf(row, m):
    s = 0.0
    for p in m['params']:
        f, v = m['rf'][p], row[p]
        s += (f.get(v, 1.0) if p in m['params_cat'] else _nearest(f, v)) * m['w'][p]
    return s


def components(row, m):
    """Pecahan prediksi: baseline geologi, faktor baris, faktor operasional."""
    k, n = _kn(row[GEO], m['base'])
    g_base = k * row['sd'] ** (-n) / G_GRAV
    f_row = m['rowf'].get(row['row_class'], 1.0) if ROW_AS_MULTIPLIER else 1.0
    f_ops = _wf(row, m) / m['mean_wf']
    return g_base, f_row, f_ops


def predict_median(row, m):
    g_base, f_row, f_ops = components(row, m)
    return g_base * f_row * f_ops


@st.cache_data(show_spinner=False)
def build(d):
    """Kalibrasi + validasi per event (tiap blast diprediksi model tanpa blast itu)."""
    m = fit_model(d)
    pred = np.empty(len(d))
    for e in pd.unique(d['event']):
        te = np.where(d['event'].values == e)[0]
        mm = fit_model(d.drop(index=te))
        for i in te:
            pred[i] = predict_median(d.iloc[i], mm)
    ratio = d[COL_G].values / pred
    fac = {'*': dict(n=len(ratio), lam=float(np.percentile(ratio, DESIGN_PCT)),
                     sigma=float(np.std(np.log(ratio), ddof=1)), bias=float(np.median(ratio)))}
    for g in sorted(d[GEO].unique()):
        r = ratio[d[GEO].values == g]
        fac[g] = dict(n=len(r), lam=float(np.percentile(r, DESIGN_PCT)),
                      sigma=float(np.std(np.log(r), ddof=1)) if len(r) > 1 else fac['*']['sigma'],
                      bias=float(np.median(r)))
    sl, ic, *_ = stats.linregress(np.log(d['sd']), np.log(d['A']))
    return m, pred, fac, (float(np.exp(ic)), float(-sl))


def factor_for(geo, fac):
    return fac.get(geo, fac['*']) if LAMBDA_PER_GEOLOGI else fac['*']


def input_row(inp):
    row = pd.Series(add_bins(pd.DataFrame([inp])).iloc[0])
    row['sd'] = inp['distance_m'] / math.sqrt(inp['charge_kg'])
    return row


def predict_event(inp, m, fac):
    row = input_row(inp)
    f = factor_for(inp[GEO], fac)
    g_base, f_row, f_ops = components(row, m)
    med = g_base * f_row * f_ops
    sims = med * np.exp(np.random.default_rng(SEED).normal(0.0, f['sigma'], N_ITER))
    return dict(row=row, sd=row['sd'], g_base=g_base, f_row=f_row, f_ops=f_ops, median=med,
                mean=med * math.exp(f['sigma'] ** 2 / 2),
                p10=float(np.percentile(sims, 10)), p90=float(np.percentile(sims, 90)),
                prob=float(np.mean(sims > G_TARGET)), design=med * f['lam'],
                lam=f['lam'], sigma=f['sigma'], sims=sims)


def max_charge(distance, inp, m, factor):
    """Charge per delay maksimum agar median*factor <= G_TARGET (desain operasional tetap)."""
    row = input_row(dict(inp, distance_m=distance))
    k, n = _kn(inp[GEO], m['base'])
    _, f_row, f_ops = components(row, m)
    sd_req = (k * f_row * f_ops * factor / (G_TARGET * G_GRAV)) ** (1.0 / n)
    return (distance / sd_req) ** 2


# =====================================================================
# [D2] KOORDINAT - baca berbagai format -> UTM 48S
# =====================================================================
# Format: Lat/Lon desimal, desimal + arah, DMS, derajat-menit desimal,
#         Easting-Northing, MGRS, UTM dengan zona.
UTM_ZONE = 48  # zona kerja (UTM 48 Selatan, WGS84)
ZONE = UTM_ZONE

# ---------------------------------------------------------------- lat/lon -> UTM (WGS84)
_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)


def latlon_to_utm(lat, lon, zone=ZONE):
    """Transverse Mercator (rumus Snyder), akurat ~mm di dalam zona."""
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    phi, lam = math.radians(lat), math.radians(lon)
    n = _A / math.sqrt(1 - _E2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = _EP2 * math.cos(phi) ** 2
    a = math.cos(phi) * (lam - lon0)
    m = _A * ((1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256) * phi
              - (3 * _E2 / 8 + 3 * _E2 ** 2 / 32 + 45 * _E2 ** 3 / 1024) * math.sin(2 * phi)
              + (15 * _E2 ** 2 / 256 + 45 * _E2 ** 3 / 1024) * math.sin(4 * phi)
              - (35 * _E2 ** 3 / 3072) * math.sin(6 * phi))
    x = _K0 * n * (a + (1 - t + c) * a ** 3 / 6 + (5 - 18 * t + t ** 2 + 72 * c - 58 * _EP2) * a ** 5 / 120)
    y = _K0 * (m + n * math.tan(phi) * (a ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
                                        + (61 - 58 * t + t ** 2 + 600 * c - 330 * _EP2) * a ** 6 / 720))
    easting = x + 500000.0
    northing = y + (10000000.0 if lat < 0 else 0.0)
    return easting, northing


# ---------------------------------------------------------------- MGRS -> UTM
_BANDS = 'CDEFGHJKLMNPQRSTUVWX'
_ROW_LETTERS = 'ABCDEFGHJKLMNPQRSTUV'


def mgrs_to_utm(zone, band, square, e_str, n_str):
    e_str, n_str = e_str.strip(), n_str.strip()
    if len(e_str) != len(n_str) or not 1 <= len(e_str) <= 5:
        raise ValueError('angka easting & northing MGRS harus sama panjang (1–5 digit)')
    scale = 10 ** (5 - len(e_str))
    e_in, n_in = int(e_str) * scale, int(n_str) * scale
    zone = int(zone)
    band, col, row = band.upper(), square[0].upper(), square[1].upper()
    s = zone % 6 or 6
    cols = {1: 'ABCDEFGH', 2: 'JKLMNPQR', 3: 'STUVWXYZ', 4: 'ABCDEFGH', 5: 'JKLMNPQR', 6: 'STUVWXYZ'}[s]
    if col not in cols:
        raise ValueError("huruf kolom MGRS '%s' tidak cocok untuk zona %d" % (col, zone))
    easting = (cols.index(col) + 1) * 100000 + e_in
    if row not in _ROW_LETTERS:
        raise ValueError("huruf baris MGRS '%s' tidak dikenal" % row)
    offset = 5 if zone % 2 == 0 else 0
    n100k = ((_ROW_LETTERS.index(row) - offset) % 20) * 100000 + n_in
    if band not in _BANDS:
        raise ValueError("lintang band MGRS '%s' tidak dikenal" % band)
    lat_band_min = -80 + 8 * _BANDS.index(band)
    lon_cm = (zone - 1) * 6 - 180 + 3
    _, n_min = latlon_to_utm(lat_band_min, lon_cm, zone)
    if lat_band_min < 0 and n_min >= 10000000:   # tepat di ekuator
        n_min -= 10000000
    northing = n100k
    while northing < n_min - 100000:
        northing += 2000000
    return float(easting), float(northing)


# ---------------------------------------------------------------- parser umum
FORMATS = {
    'auto': ('Otomatis (deteksi format)', '-3.753802, 103.864167  atau  373871, 9585004'),
    'dec': ('Lat, Lon desimal', '-3.753802, 103.864167'),
    'dec_h': ('Lat, Lon desimal + arah (S/E)', '3.753802 S, 103.864167 E'),
    'dms': ("Derajat-menit-detik (DMS)", '3°45\'13.7" S, 103°51\'51.0" E'),
    'ddm': ('Derajat-menit desimal', "3°45.228' S, 103°51.850' E"),
    'en': ('Easting, Northing (UTM 48S)', '373871, 9585004'),
    'mgrs': ('MGRS', '48M UA 73871 85004'),
    'utm': ('UTM dengan zona', '48 S 373871 E 9585004 N'),
}

_NUM = r'[+-]?\d+(?:[.,]\d+)?'
_ANGLE = re.compile(r'^\s*(?P<h1>[NSEW])?\s*(?P<d>[+-]?\d+(?:\.\d+)?)\s*(?:°|º|d|\s)?\s*'
                    r'(?:(?P<m>\d+(?:\.\d+)?)\s*(?:\'|′|m)\s*)?'
                    r'(?:(?P<s>\d+(?:\.\d+)?)\s*(?:"|″|\'\'|s)\s*)?(?P<h2>[NSEW])?\s*$')


def _clean(text):
    t = str(text).strip().upper()
    for a, b in [('’', "'"), ('‘', "'"), ('`', "'"), ('“', '"'), ('”', '"'), ('″', '"'), ('′', "'"),
                 ('º', '°'), (';', ',')]:
        t = t.replace(a, b)
    return re.sub(r'\s+', ' ', t)


def _parse_angle(part):
    m = _ANGLE.match(part)
    if not m:
        raise ValueError("bagian '%s' tidak terbaca" % part.strip())
    deg = float(m.group('d'))
    mins = float(m.group('m')) if m.group('m') else 0.0
    secs = float(m.group('s')) if m.group('s') else 0.0
    if mins >= 60 or secs >= 60:
        raise ValueError('menit/detik harus < 60')
    val = abs(deg) + mins / 60 + secs / 3600
    if deg < 0 or str(m.group('d')).startswith('-'):
        val = -val
    hemi = m.group('h1') or m.group('h2')
    if hemi in ('S', 'W'):
        val = -abs(val)
    elif hemi in ('N', 'E'):
        val = abs(val)
    kind = 'dms' if m.group('s') else ('ddm' if m.group('m') else ('dec_h' if hemi else 'dec'))
    return val, hemi, kind


def parse_coord(text, fmt='auto', zone=ZONE):
    """Return dict(e, n, kind, desc). Raise ValueError bila tidak terbaca."""
    if text is None or not str(text).strip():
        raise ValueError('kosong')
    t = _clean(text)

    # MGRS: 48M UA 73871 85004  atau 48MUA7387185004
    m = re.match(r'^(\d{1,2})\s*([C-HJ-NP-X])\s*([A-HJ-NP-Z]{2})\s*(\d+)\s+(\d+)$', t) or \
        re.match(r'^(\d{1,2})([C-HJ-NP-X])([A-HJ-NP-Z]{2})(\d{2,10})$', t)
    if m:
        if len(m.groups()) == 4:
            digits = m.group(4)
            if len(digits) % 2:
                raise ValueError('digit MGRS harus genap')
            e_s, n_s = digits[:len(digits) // 2], digits[len(digits) // 2:]
        else:
            e_s, n_s = m.group(4), m.group(5)
        if int(m.group(1)) != zone:
            raise ValueError('zona %s berbeda dengan zona kerja %d' % (m.group(1), zone))
        e, n = mgrs_to_utm(m.group(1), m.group(2), m.group(3), e_s, n_s)
        return _result(e, n, 'mgrs', fmt)

    # UTM dengan zona: 48 373871 E 9585004 S | 48 S 373871 E 9585004 N | 48M 373871 E 9585004 N
    m = re.match(r'^(\d{1,2})\s*([A-Z])?\s+(\d+(?:\.\d+)?)\s*(?:E|MT)?\s*,?\s+(\d+(?:\.\d+)?)\s*(?:[NS]|MU)?$', t)
    if m and float(m.group(3)) > 1000:
        if int(m.group(1)) != zone:
            raise ValueError('zona %s berbeda dengan zona kerja %d' % (m.group(1), zone))
        return _result(float(m.group(3)), float(m.group(4)), 'utm', fmt)

    # dua bagian: pisah koma; kalau tidak ada koma, coba pisah spasi untuk angka polos
    parts = [p for p in t.split(',') if p.strip()]
    if len(parts) != 2:
        nums = re.findall(r'^\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*$', t)
        if nums:
            parts = list(nums[0])
        else:
            raise ValueError('pisahkan dua nilai dengan koma, contoh: %s' % FORMATS.get(fmt, FORMATS['auto'])[1])

    plain = [re.match(r'^\s*[+-]?\d+(?:\.\d+)?\s*$', p) for p in parts]
    if all(plain) and all(abs(float(p)) > 1000 for p in parts):
        e, n = float(parts[0]), float(parts[1])
        if n < e:   # tertukar (Northing, Easting)
            e, n = n, e
        return _result(e, n, 'en', fmt)

    (lat, h1, k1), (lon, h2, k2) = _parse_angle(parts[0]), _parse_angle(parts[1])
    if h1 in ('E', 'W') or h2 in ('N', 'S'):     # ditulis Lon, Lat
        lat, lon = lon, lat
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError('lintang/bujur di luar rentang')
    kind = k1 if k1 == k2 else max(k1, k2, key=lambda k: ['dec', 'dec_h', 'ddm', 'dms'].index(k))
    e, n = latlon_to_utm(lat, lon, zone)
    return _result(e, n, kind, fmt, lat=lat, lon=lon)


def _result(e, n, kind, fmt, lat=None, lon=None):
    if fmt not in ('auto', kind):
        raise ValueError("terbaca sebagai '%s', bukan '%s'" % (FORMATS[kind][0], FORMATS[fmt][0]))
    return dict(e=e, n=n, kind=kind, desc=FORMATS[kind][0], lat=lat, lon=lon)


# =====================================================================
# [E] SIDEBAR
# =====================================================================
st.sidebar.markdown('## 📂 Sumber data')
SRC_DRIVE, SRC_UPLOAD = '🔄 Otomatis dari Google Drive', '📤 Upload file manual'
data_mode = st.sidebar.radio('Sumber data', [SRC_DRIVE, SRC_UPLOAD], label_visibility='collapsed')
uploaded = None
if data_mode == SRC_DRIVE:
    if st.sidebar.button('Muat ulang data sekarang', icon='🔄'):
        fetch_remote.clear()
        st.rerun()
    st.sidebar.caption('Setiap Excel di Google Drive diubah, aplikasi menghitung ulang otomatis '
                       '(paling lambat %d menit, atau langsung dengan tombol di atas).' % REFRESH_MIN)
else:
    uploaded = st.sidebar.file_uploader('File Excel (.xlsx)', type=['xlsx'],
                                        help='Format kolom sama dengan sheet di Google Drive.')
    st.sidebar.download_button('Unduh template kosong', template_bytes(), file_name='template_data_pit3.xlsx',
                               icon='📄', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
data_status = st.sidebar.container()
st.sidebar.divider()

ELEV_OPTIONS = ['lower', 'normal', 'higher']
st.session_state.setdefault('in_distance', 255.0)
st.session_state.setdefault('in_elev', 'higher')

st.sidebar.markdown('## ⚙️ Rencana peledakan')
INPUT = {
    'distance_m': st.sidebar.number_input('Jarak ke titik ukur (m)', min_value=1.0, step=5.0, key='in_distance',
                                          help='Isi manual, atau hitung dari koordinat di tab 📍 Jarak & sketsa.'),
    'charge_kg': st.sidebar.number_input('Charge per delay (kg)', min_value=1.0, value=30.0, step=1.0),
    'row_number': int(st.sidebar.number_input('Jumlah baris', min_value=1, max_value=15, value=5, step=1)),
    'geological_condt': st.sidebar.selectbox('Geologi', ['Fault', 'coal', 'normal'], index=0),
}
st.sidebar.markdown('#### Tie-up & delay')
INPUT['tie_up_type'] = st.sidebar.selectbox('Tie-up', ['boxcut', 'echelon'], index=0)
INPUT['controll_ms'] = st.sidebar.selectbox('Control delay (ms)', DELAY_OPTIONS, index=DELAY_OPTIONS.index(109))
INPUT['wall_echelon_ms'] = st.sidebar.selectbox('Wall echelon (ms)', DELAY_OPTIONS, index=DELAY_OPTIONS.index(42))
INPUT['freeface_echelon_ms'] = st.sidebar.selectbox('Freeface echelon (ms)', DELAY_OPTIONS, index=0)
INPUT['freeface_count'] = st.sidebar.selectbox('Freeface count', [1, 2, 3], index=1,
                                               help='1 = atas saja · 2 = atas + depan · 3 = atas + depan + samping')
INPUT['measuring_elevation'] = st.sidebar.selectbox('Elevasi titik ukur', ELEV_OPTIONS, key='in_elev')



# =====================================================================
# [F] MUAT DATA & KALIBRASI
# =====================================================================
if data_mode == SRC_UPLOAD and uploaded is None:
    st.markdown('# 💥 Prediksi Getaran Blasting — Pit 3')
    st.info('**Mode upload manual.** Unggah file Excel di sidebar kiri untuk mulai menghitung. '
            'Formatnya sama dengan sheet di Google Drive (nama sheet `%s`). Kalau perlu, unduh '
            'template kosong di bawah kotak upload.' % SHEET_NAME, icon='📤')
    data_status.caption('Belum ada file yang diunggah.')
    st.stop()

with st.spinner('Membaca data dan mengkalibrasi model ...'):
    try:
        if data_mode == SRC_UPLOAD:
            raw = read_excel_any(uploaded.getvalue())
            source, fetched = 'file unggahan (%s)' % uploaded.name, None
        elif LOCAL_DATA:
            raw = read_excel_any(LOCAL_DATA)
            source, fetched = 'Google Drive', pd.Timestamp.now(tz='Asia/Jakarta').strftime('%d/%m/%Y %H:%M')
        else:
            content, fetched = fetch_remote(tuple(DOWNLOAD_URLS))
            raw = read_excel_any(content)
            source = 'Google Drive'
        df, excluded = prepare(raw)
    except Exception as e:
        if data_mode == SRC_DRIVE:
            st.error('**Gagal mengambil data dari Google Drive.** Pastikan file dibagikan sebagai '
                     '"Anyone with the link – Viewer", atau pindah ke **Upload file manual** di sidebar.'
                     '\n\nDetail: %s' % e, icon='⛔')
        else:
            st.error('**File tidak bisa dibaca.** Pastikan formatnya sama dengan template. Detail: %s' % e,
                     icon='⛔')
        st.stop()
    if len(df) < 15:
        st.error('Data valid hanya %d blast — terlalu sedikit untuk kalibrasi.' % len(df))
        st.stop()
    model, cv_pred, fac, sd_line = build(df)

last_date, future_dates = last_blast_date(df)
data_status.success('**%d blast** dipakai · %d disisihkan  \nData terakhir: **%s**%s'
                    % (len(df), len(excluded), last_date,
                       '  \nDiambil: %s WIB' % fetched if fetched else ''), icon='✅')
if future_dates:
    data_status.warning('Ada tanggal di masa depan (%s) — kemungkinan salah ketik di Excel.'
                       % ', '.join(future_dates), icon='📅')

res = predict_event(INPUT, model, fac)
geos = sorted(df[GEO].unique())
mae = float(np.mean(np.abs(df[COL_G].values - cv_pred)))
ch_min, ch_max = float(df[COL_CHG].min()), float(df[COL_CHG].max())

# Peringatan input
warns = []
if INPUT['row_number'] > MAX_ROWS:
    warns.append('Jumlah baris %d di atas batas kalibrasi (maks %d). Prediksi memakai faktor 8–9 baris '
                 'dan kemungkinan masih terlalu kecil.' % (INPUT['row_number'], MAX_ROWS))
if INPUT['distance_m'] < MIN_DIST:
    warns.append('Jarak %.0f m di bawah batas kalibrasi (%d m) — hasil ekstrapolasi.' % (INPUT['distance_m'], MIN_DIST))
if not (ch_min <= INPUT['charge_kg'] <= ch_max):
    warns.append('Charge %.0f kg di luar rentang data (%.0f–%.0f kg).' % (INPUT['charge_kg'], ch_min, ch_max))
if not (df['sd'].min() <= res['sd'] <= df['sd'].max()):
    warns.append('Scaled distance %.1f di luar rentang data (%.1f–%.1f).' % (res['sd'], df['sd'].min(), df['sd'].max()))
if INPUT[GEO] not in geos:
    warns.append("Geologi '%s' belum ada di data." % INPUT[GEO])
for key, col in [('controll_ms', 'controll_ms'), ('wall_echelon_ms', 'wall_echelon_ms'),
                 ('freeface_echelon_ms', 'freeface_echelon_ms')]:
    if float(INPUT[key]) not in set(df[col].astype(float)):
        warns.append('%s %s ms belum pernah dipakai di data — dipakai level terdekat.' % (NICE.get(col, col), INPUT[key]))


# =====================================================================
# [G] HEADER
# =====================================================================
st.markdown('# 💥 Prediksi Getaran Blasting — Pit 3')
st.markdown('<span class="muted">Model v5 · atenuasi per geologi · faktor jumlah baris · '
            'Monte Carlo %s iterasi · %d blast dari %s · batas g = %.2f</span>'
            % (format(N_ITER, ','), len(df), source, G_TARGET), unsafe_allow_html=True)

tab_pred, tab_dist, tab_hist, tab_factor, tab_val = st.tabs(
    ['🎯 Prediksi', '📍 Jarak & sketsa', '📈 Data historis per faktor', '🧭 Faktor pengaruh', '✅ Validasi model'])


# =====================================================================
# TAB 1 - PREDIKSI
# =====================================================================
with tab_pred:
    cm_design = max_charge(INPUT['distance_m'], INPUT, model, res['lam'])
    cm_median = max_charge(INPUT['distance_m'], INPUT, model, 1.0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi('Median g', '%.4f' % res['median'], 'Nilai paling mungkin · rentang P10–P90 %.4f–%.4f'
            % (res['p10'], res['p90']), g_for_status=res['median'])
    with c2:
        kpi('Design value (P%d)' % DESIGN_PCT, '%.4f' % res['design'],
            'Acuan keputusan geoteknik · median × %.2f' % res['lam'], g_for_status=res['design'])
    with c3:
        pc = '#d03b3b' if res['prob'] > 0.5 else ('#fab219' if res['prob'] > 0.1 else '#0ca30c')
        kpi('Peluang g > %.2f' % G_TARGET, '%.0f%%' % (100 * res['prob']),
            '%.0f dari 100 peledakan dengan desain ini' % (100 * res['prob']), color=pc)
    with c4:
        note = '⚠ di bawah charge terkecil di data' if cm_design < ch_min else 'design value tetap ≤ %.2f' % G_TARGET
        kpi('Charge maks aman', '%.1f kg' % cm_design,
            '%s · median ≤ %.2f: %.1f kg' % (note, G_TARGET, cm_median), color=ACCENT)

    s = status_of(res['design'])
    if res['design'] <= G_TARGET:
        msg = 'Desain aman sampai design value — tidak perlu perubahan.'
    elif res['median'] <= G_TARGET:
        msg = 'Median masih di bawah batas, tapi design value melewatinya. Perlu mitigasi untuk keputusan geoteknik.'
    else:
        msg = 'Median sudah di atas batas. Desain perlu diubah sebelum ditembak.'
    st.markdown('<div class="banner" style="--c:%s"><b>%s %s</b> — %s</div>'
                % (s['color'], s['icon'], s['label'], msg), unsafe_allow_html=True)
    for w in warns:
        st.warning(w, icon='⚠️')

    left, right = st.columns([1.15, 1])
    with left:
        sims = res['sims']
        hi = float(np.percentile(sims, 99.5))
        fig = go.Figure(go.Histogram(x=sims[sims <= hi], nbinsx=60, marker_color=PALETTE[0],
                                     marker_line=dict(color='rgba(255,255,255,0.9)', width=1),
                                     hovertemplate='g %{x}<br>%{y} simulasi<extra></extra>', name='simulasi'))
        left_side = G_TARGET <= res['median']
        for x, lab, dash, ypos, anchor in [
                (G_TARGET, 'batas %.2f' % G_TARGET, 'dot', 1.0, 'right' if left_side else 'left'),
                (res['median'], 'median %.4f' % res['median'], 'solid', 0.90, 'left' if left_side else 'right'),
                (res['design'], 'design %.4f' % res['design'], 'dash', 1.0, 'left')]:
            fig.add_vline(x=x, line=dict(color=MUTED, width=2, dash=dash))
            fig.add_annotation(x=x, y=ypos, xref='x', yref='paper', text=' %s ' % lab, showarrow=False,
                               xanchor=anchor, yanchor='top', font=dict(size=11),
                               bgcolor='rgba(255,255,255,0.75)')
        fig.update_xaxes(title='g')
        fig.update_yaxes(title='jumlah simulasi')
        show_chart(style_fig(fig, 'Sebaran %s simulasi Monte Carlo' % format(N_ITER, ','), 380, legend=False))

    with right:
        st.markdown('##### Kenapa hasilnya segini?')
        steps = [
            ('Jarak & charge + geologi %s' % INPUT[GEO], res['g_base'], 'SD %.1f · k=%.0f, n=%.2f'
             % ((res['sd'],) + _kn(INPUT[GEO], model['base']))),
            ('× jumlah baris (%s)' % res['row']['row_class'], res['g_base'] * res['f_row'],
             'faktor ×%.2f' % res['f_row']),
            ('× faktor operasional → median', res['median'], 'faktor ×%.2f' % res['f_ops']),
            ('× λ → design value', res['design'], 'faktor ×%.2f' % res['lam']),
        ]
        fig = go.Figure()
        labels = [s_[0] for s_ in steps][::-1]
        vals = [s_[1] for s_ in steps][::-1]
        notes = [s_[2] for s_ in steps][::-1]
        cols = [PALETTE[1]] + [ACCENT] * (len(vals) - 1)   # design value (paling atas) dibedakan
        fig.add_trace(go.Bar(y=labels, x=vals, orientation='h', marker_color=cols,
                             text=['%.4f  (%s)' % (v, n_) for v, n_ in zip(vals, notes)],
                             textposition='outside', cliponaxis=False,
                             hovertemplate='%{y}<br>g = %{x:.4f}<extra></extra>'))
        fig.add_vline(x=G_TARGET, line=dict(color=MUTED, width=1.5, dash='dot'),
                      annotation_text='batas', annotation_position='top')
        fig.update_xaxes(title='g', range=[0, max(vals) * 1.75])
        fig.update_layout(bargap=0.35)
        show_chart(style_fig(fig, None, 300, legend=False))
        st.markdown('<span class="muted">Dibaca dari atas ke bawah: tiap baris menambahkan satu faktor. '
                    'Batang tosca = design value, garis titik-titik = batas %.2f.</span>' % G_TARGET,
                    unsafe_allow_html=True)

    # ---------------- REKOMENDASI
    st.markdown('### 🛠️ Rekomendasi')
    if res['design'] <= G_TARGET:
        st.success('Design value sudah ≤ %.2f. Tidak perlu perubahan desain.' % G_TARGET)
    else:
        work = dict(INPUT)
        changes = []
        for raw_key, p, to_raw in CONTROLLABLE:
            if p not in model['rf']:
                continue
            rf, cnt = model['rf'][p], model['cnt'][p]
            cur_val = input_row(work)[p]
            cur_rf = rf.get(cur_val, 1.0) if p in model['params_cat'] else _nearest(rf, cur_val)
            ok = [k for k in rf if cnt[k] >= MIN_SAMPLES]
            if not ok:
                continue
            best = min(ok, key=lambda k: rf[k])
            if rf[best] < cur_rf - 1e-9:
                new_val = to_raw(best)
                changes.append((NICE[p], INPUT[raw_key], new_val, cnt[best]))
                work[raw_key] = new_val
        r1 = predict_event(work, model, fac)
        _, n_geo = _kn(INPUT[GEO], model['base'])
        ch_des = work['charge_kg'] * (G_TARGET / r1['design']) ** (2.0 / n_geo)
        ch_med = work['charge_kg'] * (G_TARGET / r1['median']) ** (2.0 / n_geo) if r1['median'] > G_TARGET else None

        a, b = st.columns(2)
        with a:
            st.markdown('<div class="step"><b>Langkah 1 — ubah tie-up/delay ke level paling aman</b><br>'
                        '<span class="muted">Hanya level yang sudah dipakai ≥ %d blast.</span></div>'
                        % MIN_SAMPLES, unsafe_allow_html=True)
            if changes:
                for nm, old, new, n_ in changes:
                    st.markdown('- **%s**: %s → **%s** &nbsp;<span class="muted">(%d blast)</span>'
                                % (nm, old, new, n_), unsafe_allow_html=True)
            else:
                st.markdown('- Tidak ada perubahan tie-up/delay yang menurunkan g.')
            st.markdown('Hasil: median **%.4f** · design value **%.4f**' % (r1['median'], r1['design']))
            if INPUT['row_number'] >= 8:
                st.info('Round %d baris termasuk kelas 8–9 baris (faktor ×%.2f). Memecah jadi ≤ 7 baris '
                        'menurunkan g sekitar %.0f%%.'
                        % (INPUT['row_number'], model['rowf'].get('8-9', 1.0),
                           100 * (1 - model['rowf'].get('4-7', 1.0) / model['rowf'].get('8-9', 1.0))))
        with b:
            st.markdown('<div class="step"><b>Langkah 2 — kurangi charge per delay</b><br>'
                        '<span class="muted">g ∝ charge^(n/2), n %s = %.2f</span></div>'
                        % (INPUT[GEO], n_geo), unsafe_allow_html=True)
            if ch_med is not None:
                st.markdown('- Agar **median** ≤ %.2f: %.1f → **%.1f kg**' % (G_TARGET, INPUT['charge_kg'], ch_med))
            else:
                st.markdown('- Median sudah ≤ %.2f setelah langkah 1.' % G_TARGET)
            st.markdown('- Agar **design value** ≤ %.2f: %.1f → **%.1f kg**' % (G_TARGET, INPUT['charge_kg'], ch_des))
            if ch_des < ch_min:
                st.warning('%.1f kg di bawah charge terkecil di data (%.0f kg) — ekstrapolasi, kurang andal.'
                           % (ch_des, ch_min))
            st.caption('Charge lebih kecil menurunkan getaran tapi bisa memperburuk fragmentasi. '
                       'Alternatif: deck/delay lebih halus, pre-split, atau jadwal ulang terhadap lereng.')

    # ---------------- SENSITIVITAS & CHARGE MAKS
    st.markdown('### 📉 Kalau charge diubah')
    a, b = st.columns([1.2, 1])
    with a:
        charges = np.arange(10, 95, 5)
        meds, dess = [], []
        for ch in charges:
            rc = predict_event(dict(INPUT, charge_kg=float(ch)), model, fac)
            meds.append(rc['median']); dess.append(rc['design'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=charges, y=meds, mode='lines+markers', name='median',
                                 line=dict(color=PALETTE[0], width=2), marker=dict(size=8),
                                 hovertemplate='%{x} kg<br>median %{y:.4f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=charges, y=dess, mode='lines+markers', name='design value',
                                 line=dict(color=PALETTE[1], width=2), marker=dict(size=8, symbol='square'),
                                 hovertemplate='%{x} kg<br>design %{y:.4f}<extra></extra>'))
        ref_hline(fig, G_TARGET, 'batas %.2f' % G_TARGET, log_y=False, side='left')
        fig.add_vline(x=INPUT['charge_kg'], line=dict(color=MUTED, width=1, dash='dash'),
                      annotation_text='input', annotation_position='top')
        fig.update_xaxes(title='charge per delay (kg)')
        fig.update_yaxes(title='g')
        show_chart(style_fig(fig, 'Jarak %.0f m · %s · desain operasional tetap' % (INPUT['distance_m'], INPUT[GEO]), 380))
    with b:
        st.markdown('##### Charge maksimum per delay (kg)')
        st.markdown('<span class="muted">Desain operasional sama dengan input. Angka kiri: median ≤ %.2f · '
                    'kanan: design value ≤ %.2f. ⚠ = di bawah charge terkecil di data.</span>'
                    % (G_TARGET, G_TARGET), unsafe_allow_html=True)
        rows_ = []
        for dist in [100, 125, 150, 175, 200, 250, 300, 350, 400, 450]:
            item = {'Jarak (m)': dist}
            for g in geos:
                gi = dict(INPUT, geological_condt=g)
                lam_g = factor_for(g, fac)['lam']
                cmed = max_charge(dist, gi, model, 1.0)
                cdes = max_charge(dist, gi, model, lam_g)
                item[g] = '%.1f / %.1f%s' % (cmed, cdes, ' ⚠' if cdes < ch_min else '')
            rows_.append(item)
        show_table(pd.DataFrame(rows_))


# =====================================================================
# TAB - JARAK & SKETSA (opsional; prediksi tetap jalan tanpa ini)
# =====================================================================
def _set_from_calc(dist, elev):
    """Callback tombol: isi jarak & elevasi di sidebar."""
    st.session_state['in_distance'] = float(round(dist, 1))
    st.session_state['in_elev'] = elev


def _seg_dist(px, py, ax, ay, bx, by):
    """Jarak titik P ke ruas AB."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy), qx, qy


def _nice_len(span):
    for v in [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000]:
        if v >= span / 6:
            return v
    return 5000


FMT_KEYS = list(FORMATS.keys())

with tab_dist:
    st.markdown('Hitung **jarak horizontal** dari sudut blok peledakan ke alat ukur, sebelum data diolah '
                'di Blastware. Bagian ini opsional — kalau dikosongkan, prediksi tetap memakai jarak '
                'yang diisi manual di sidebar.')

    colA, colB = st.columns([1.15, 1])
    with colA:
        st.markdown('##### 💥 Lokasi blasting — sudut blok')
        fmt_b = st.selectbox('Format koordinat blasting', FMT_KEYS, key='fmt_blast',
                             format_func=lambda k: FORMATS[k][0])
        st.caption('Contoh: `%s`' % FORMATS[fmt_b][1])
        n_corner = int(st.number_input('Jumlah sudut blok', min_value=1, max_value=12, value=4, step=1,
                                       key='n_corner'))
        corner_txt = []
        cc = st.columns(2)
        for i in range(n_corner):
            with cc[i % 2]:
                corner_txt.append(('S%d' % (i + 1),
                                   st.text_input('Sudut S%d' % (i + 1), key='corner_%d' % i,
                                                 placeholder=FORMATS[fmt_b][1])))
    with colB:
        st.markdown('##### 📟 Alat ukur')
        fmt_i = st.selectbox('Format koordinat alat ukur', FMT_KEYS, key='fmt_inst',
                             format_func=lambda k: FORMATS[k][0])
        inst_txt = st.text_input('Koordinat alat ukur', key='inst_coord', placeholder=FORMATS[fmt_i][1])
        c1_, c2_ = st.columns(2)
        with c1_:
            inst_id = st.text_input('ID alat (opsional)', key='inst_id', placeholder='mis. UM16793')
        with c2_:
            blast_date = st.date_input('Tanggal blasting', key='blast_date', format='DD/MM/YYYY')

    # ---------------- baca koordinat
    corners, problems = [], []
    for name, txt in corner_txt:
        txt = (txt or '').strip()
        if not txt:
            continue
        try:
            p = parse_coord(txt, fmt_b)
            corners.append(dict(name=name, e=p['e'], n=p['n'], kind=p['desc']))
        except ValueError as e:
            problems.append('%s: %s' % (name, e))
    inst = None
    if inst_txt.strip():
        try:
            inst = parse_coord(inst_txt, fmt_i)
        except ValueError as e:
            problems.append('Alat ukur: %s' % e)

    for p_ in problems:
        st.warning(p_, icon='⚠️')

    if inst is None or not corners:
        need = []
        if not corners:
            need.append('minimal satu koordinat sudut blok')
        if inst is None:
            need.append('koordinat alat ukur')
        st.info('Isi %s untuk menghitung jarak. Semua titik diubah ke UTM zona %dS (WGS84), '
                'jadi format blasting dan alat ukur boleh berbeda.' % (' dan '.join(need), UTM_ZONE), icon='📍')
    else:
        for c in corners:
            c['d'] = math.hypot(c['e'] - inst['e'], c['n'] - inst['n'])
        nearest = min(corners, key=lambda c: c['d'])
        names = [c['name'] for c in corners]

        st.divider()
        a, b = st.columns([1, 1.6])
        with a:
            st.markdown('##### Jarak tiap sudut ke alat ukur')
            show_table(pd.DataFrame([{
                'Titik': c['name'], 'Easting': round(c['e'], 1), 'Northing': round(c['n'], 1),
                'Jarak horizontal (m)': round(c['d'], 1),
                'Catatan': 'terdekat' if c is nearest else ''} for c in corners]))
            pick = st.selectbox('Sudut yang dipakai', names, index=names.index(nearest['name']),
                                format_func=lambda nm: '%s — %.1f m%s' % (
                                    nm, next(c['d'] for c in corners if c['name'] == nm),
                                    '  (terdekat)' if nm == nearest['name'] else ''))
            chosen = next(c for c in corners if c['name'] == pick)
            elev = st.radio('Elevasi alat ukur terhadap lokasi blasting', ELEV_OPTIONS, horizontal=True,
                            index=ELEV_OPTIONS.index(st.session_state.get('in_elev', 'higher')),
                            key='calc_elev')
            sd_val = chosen['d'] / math.sqrt(INPUT['charge_kg'])
            m1, m2 = st.columns(2)
            with m1:
                kpi('Jarak horizontal', '%.1f m' % chosen['d'], 'ke %s' % chosen['name'])
            with m2:
                kpi('Scaled distance', '%.1f' % sd_val, 'charge %.0f kg/delay (sidebar)' % INPUT['charge_kg'])

            # tepi blok bisa lebih dekat dari sudut
            if len(corners) >= 2:
                ring = corners + ([corners[0]] if len(corners) >= 3 else [])
                edge = min((_seg_dist(inst['e'], inst['n'], p1['e'], p1['n'], p2['e'], p2['n'])
                            for p1, p2 in zip(ring[:-1], ring[1:])), key=lambda t: t[0])
                if nearest['d'] - edge[0] > 5:
                    st.caption('Catatan: tepi blok terdekat berjarak %.1f m dari alat ukur — %.1f m lebih dekat '
                               'dari sudut terdekat. Kalau ada lubang di sepanjang tepi itu, jarak sebenarnya '
                               'bisa lebih pendek.' % (edge[0], nearest['d'] - edge[0]))

            st.button('➡️ Pakai jarak & elevasi ini untuk prediksi', type='primary',
                      on_click=_set_from_calc, args=(chosen['d'], elev))
            if abs(st.session_state.get('in_distance', 0) - round(chosen['d'], 1)) < 0.05 and \
                    st.session_state.get('in_elev') == elev:
                st.success('Sudah dipakai di tab Prediksi: %.1f m, %s.' % (chosen['d'], elev), icon='✅')

            row = pd.DataFrame([{
                'tanggal': blast_date.strftime('%d/%m/%Y') if blast_date else '',
                'id_alat': inst_id, 'alat_easting': round(inst['e'], 1), 'alat_northing': round(inst['n'], 1),
                'titik_blast': chosen['name'], 'blast_easting': round(chosen['e'], 1),
                'blast_northing': round(chosen['n'], 1), 'distance_m': round(chosen['d'], 1),
                'charge_kg': INPUT['charge_kg'], 'scale_distance': round(sd_val, 2),
                'measuring_elevation': elev, 'zona_utm': '%dS' % UTM_ZONE,
            }])
            st.download_button('⬇️ Unduh baris untuk data master (CSV)', row.to_csv(index=False).encode('utf-8'),
                               file_name='jarak_%s.csv' % (blast_date.strftime('%Y%m%d') if blast_date else 'blast'),
                               mime='text/csv')

        with b:
            fig = go.Figure()
            xs = [c['e'] for c in corners]
            ys = [c['n'] for c in corners]
            if len(corners) >= 3:
                fig.add_trace(go.Scatter(x=xs + [xs[0]], y=ys + [ys[0]], mode='lines', fill='toself',
                                         fillcolor='rgba(194,124,14,0.16)', line=dict(color=ACCENT, width=2),
                                         name='blok blasting', hoverinfo='skip'))
            elif len(corners) == 2:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=ACCENT, width=2),
                                         name='blok blasting', hoverinfo='skip'))
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='markers+text', name='sudut blok', text=[c['name'] for c in corners],
                textposition='top center', textfont=dict(size=12),
                marker=dict(color=ACCENT, size=11, symbol='square', line=dict(color='white', width=1.5)),
                customdata=[[c['name'], c['d']] for c in corners],
                hovertemplate='<b>%{customdata[0]}</b><br>E %{x:,.1f}<br>N %{y:,.1f}'
                              '<br>jarak %{customdata[1]:.1f} m<extra></extra>'))
            fig.add_trace(go.Scatter(
                x=[chosen['e']], y=[chosen['n']], mode='markers', name='sudut dipakai', hoverinfo='skip',
                marker=dict(color='rgba(0,0,0,0)', size=22, line=dict(color=ACCENT, width=3))))
            fig.add_trace(go.Scatter(
                x=[inst['e']], y=[inst['n']], mode='markers+text', name='alat ukur',
                text=['Alat ukur' + (' ' + inst_id if inst_id else '')], textposition='bottom center',
                marker=dict(color=PALETTE[1], size=16, symbol='triangle-up', line=dict(color='white', width=1.5)),
                hovertemplate='<b>Alat ukur</b><br>E %{x:,.1f}<br>N %{y:,.1f}<extra></extra>'))
            fig.add_trace(go.Scatter(
                x=[inst['e'], chosen['e']], y=[inst['n'], chosen['n']], mode='lines', showlegend=False,
                line=dict(color='rgba(60,60,60,0.85)', width=2, dash='dash'), hoverinfo='skip'))
            fig.add_annotation(x=(inst['e'] + chosen['e']) / 2, y=(inst['n'] + chosen['n']) / 2,
                               text='<b>%.1f m</b>' % chosen['d'], showarrow=False, font=dict(size=13),
                               bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(60,60,60,0.5)', borderpad=3)

            allx, ally = xs + [inst['e']], ys + [inst['n']]
            span = max(max(allx) - min(allx), max(ally) - min(ally), 40.0)
            cx, cy = (max(allx) + min(allx)) / 2, (max(ally) + min(ally)) / 2
            half = span * 0.62
            x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
            # skala
            L = _nice_len(2 * half)
            sx, sy = x0 + 0.06 * 2 * half, y0 + 0.06 * 2 * half
            fig.add_shape(type='line', x0=sx, x1=sx + L, y0=sy, y1=sy, line=dict(color='rgba(40,40,40,0.9)', width=4))
            fig.add_annotation(x=sx + L / 2, y=sy, yshift=12, text='%d m' % L, showarrow=False, font=dict(size=11))
            # panah utara
            fig.add_annotation(x=0.95, y=0.93, ax=0.95, ay=0.78, xref='x domain', yref='y domain',
                               axref='x domain', ayref='y domain', showarrow=True, arrowhead=2, arrowsize=1.4,
                               arrowwidth=2.5, arrowcolor='rgba(40,40,40,0.9)', text='')
            fig.add_annotation(x=0.95, y=0.94, xref='x domain', yref='y domain', text='<b>U</b>',
                               showarrow=False, yanchor='bottom', font=dict(size=13))
            fig.update_xaxes(range=[x0, x1], title='Easting (m) — UTM %dS' % UTM_ZONE, tickformat=',d')
            fig.update_yaxes(range=[y0, y1], title='Northing (m)', tickformat=',d', scaleanchor='x', scaleratio=1)
            show_chart(style_fig(fig, 'Sketsa lokasi — %s · %s' % (
                blast_date.strftime('%d/%m/%Y') if blast_date else '', elev), 560))
            st.caption('Ikon kamera di pojok kanan atas grafik untuk menyimpan sketsa sebagai gambar PNG.')


# =====================================================================
# TAB 2 - DATA HISTORIS PER FAKTOR
# =====================================================================
with tab_hist:
    st.markdown('Setiap titik adalah satu pengukuran blast. Pilih faktor untuk melihat bagaimana '
                'getaran terukur berbeda menurut level faktor itu.')
    fac_label = st.selectbox('Faktor', list(HIST_FACTORS.keys()), index=1)
    col, order = HIST_FACTORS[fac_label]
    levels = order if order else sorted(df[col].dropna().unique(), key=float)
    levels = [lv for lv in levels if (df[col] == lv).any()]
    k_all, n_all = sd_line
    d = df.copy()
    d['rasio'] = d['A'] / (k_all * d['sd'] ** (-n_all))

    def lv_txt(v):
        return str(int(v)) if isinstance(v, (int, float, np.floating)) and float(v).is_integer() else str(v)

    hover = ('<b>%{customdata[0]}</b><br>jarak %{customdata[1]:.0f} m · charge %{customdata[2]:.0f} kg'
             '<br>g = %{customdata[3]:.4f} · %{customdata[4]} baris · %{customdata[5]}<extra></extra>')

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        xs_all = np.linspace(d['sd'].min(), d['sd'].max(), 40)
        for i, lv in enumerate(levels):
            x = d[d[col] == lv]
            colr, sym = PALETTE[i % len(PALETTE)], SYMBOLS[i % len(SYMBOLS)]
            fig.add_trace(go.Scatter(
                x=x['sd'], y=x['A'], mode='markers', name='%s (n=%d)' % (lv_txt(lv), len(x)),
                marker=dict(color=colr, symbol=sym, size=10, line=dict(color='white', width=1)),
                customdata=np.stack([x['tanggal'], x[COL_DIST], x[COL_CHG], x[COL_G],
                                     x['row_number'].astype(int), x[GEO]], axis=1),
                hovertemplate=hover, legendgroup=str(lv)))
            if len(x) >= MIN_GEO_FOR_OWN_N and x['sd'].nunique() > 2:
                s_, i_, *_ = stats.linregress(np.log(x['sd']), np.log(x['A']))
                xs = np.linspace(x['sd'].min(), x['sd'].max(), 30)
                fig.add_trace(go.Scatter(x=xs, y=np.exp(i_) * xs ** s_, mode='lines', showlegend=False,
                                         line=dict(color=colr, width=2), legendgroup=str(lv),
                                         hovertemplate='%s: A = %.0f·SD^-%.2f<extra></extra>'
                                         % (lv_txt(lv), np.exp(i_), -s_)))
        ref_hline(fig, G_TARGET * G_GRAV, 'batas g = %.2f' % G_TARGET, log_y=True, side='left')
        fig.update_xaxes(type='log', title='Scaled distance SD = R/√W')
        fig.update_yaxes(type='log', title='A maks (mm/s²)')
        show_chart(style_fig(fig, 'Atenuasi menurut %s' % fac_label.lower(), 460))
        st.markdown('<span class="muted">Garis tren digambar untuk level dengan ≥ %d blast.</span>'
                    % MIN_GEO_FOR_OWN_N, unsafe_allow_html=True)

    with b:
        fig = go.Figure()
        rng_ = np.random.default_rng(1)
        summary = []
        for i, lv in enumerate(levels):
            x = d[d[col] == lv]
            colr, sym = PALETTE[i % len(PALETTE)], SYMBOLS[i % len(SYMBOLS)]
            jitter = i + rng_.uniform(-0.18, 0.18, len(x))
            fig.add_trace(go.Scatter(
                x=jitter, y=x['rasio'], mode='markers', showlegend=False,
                marker=dict(color=colr, symbol=sym, size=10, opacity=0.85, line=dict(color='white', width=1)),
                customdata=np.stack([x['tanggal'], x[COL_DIST], x[COL_CHG], x[COL_G],
                                     x['row_number'].astype(int), x[GEO]], axis=1),
                hovertemplate=hover))
            med = _gm(x['rasio'])
            fig.add_trace(go.Scatter(x=[i - 0.32, i + 0.32], y=[med, med], mode='lines', showlegend=False,
                                     line=dict(color=colr, width=4),
                                     hovertemplate='%s: rasio rata-rata %.2f<extra></extra>' % (lv_txt(lv), med)))
            fig.add_annotation(x=i, y=1.0, yref='paper', yanchor='bottom', showarrow=False,
                               text='n=%d%s' % (len(x), ' · tipis' if len(x) < MIN_SAMPLES else ''),
                               font=dict(size=11))
            summary.append(dict(level=lv_txt(lv), n=len(x), ratio=med, g_med=float(x[COL_G].median())))
        ref_hline(fig, 1.0, 'sesuai perkiraan jarak & charge', log_y=True, side='right')
        fig.update_xaxes(tickmode='array', tickvals=list(range(len(levels))),
                         ticktext=[lv_txt(lv) for lv in levels], title=fac_label, showgrid=False)
        fig.update_yaxes(type='log', title='g aktual ÷ prediksi scaled distance')
        show_chart(style_fig(fig, 'Lebih / kurang bergetar dari perkiraan jarak & charge', 460, legend=False))
        st.markdown('<span class="muted">Garis tebal = rata-rata geometrik tiap level. Di atas garis putus '
                    '= getaran lebih besar dari perkiraan scaled distance saja.</span>', unsafe_allow_html=True)

    tbl = pd.DataFrame([{
        fac_label: s_['level'], 'Jumlah blast': s_['n'], 'Median g': round(s_['g_med'], 4),
        'Rasio rata-rata': round(s_['ratio'], 2),
        'Dibanding perkiraan SD': '%+.0f%%' % (100 * (s_['ratio'] - 1)),
        'Catatan': 'data tipis (< %d blast)' % MIN_SAMPLES if s_['n'] < MIN_SAMPLES else '',
    } for s_ in summary])
    show_table(tbl)

    solid = [s_ for s_ in summary if s_['n'] >= MIN_SAMPLES]
    if len(solid) >= 2:
        hi_ = max(solid, key=lambda s_: s_['ratio'])
        lo_ = min(solid, key=lambda s_: s_['ratio'])
        st.info('**Ringkasan %s:** blast dengan %s = %s rata-rata **%+.0f%%** dibanding perkiraan jarak & charge, '
                'sedangkan %s = %s **%+.0f%%**. Selisih keduanya %.0f%%.'
                % (fac_label.lower(), fac_label.lower(), hi_['level'], 100 * (hi_['ratio'] - 1),
                   fac_label.lower(), lo_['level'], 100 * (lo_['ratio'] - 1),
                   100 * (hi_['ratio'] / lo_['ratio'] - 1)))
    thin_lv = [s_['level'] for s_ in summary if s_['n'] < MIN_SAMPLES]
    if thin_lv:
        st.caption('Level dengan data < %d blast (%s) belum cukup untuk disimpulkan.' % (MIN_SAMPLES, ', '.join(thin_lv)))


# =====================================================================
# TAB 3 - FAKTOR PENGARUH
# =====================================================================
with tab_factor:
    st.markdown('Seberapa besar prediksi **untuk desain Anda** berubah kalau satu faktor diganti, '
                'faktor lain tetap.')
    row0 = res['row']
    items = []
    lv_geo = []
    for g in geos:
        r_ = row0.copy(); r_[GEO] = g
        lv_geo.append((g, predict_median(r_, model)))
    items.append((NICE[GEO], lv_geo))
    lv_row = []
    for c in ROW_CLASSES:
        if c in model['rowf'] or not ROW_AS_MULTIPLIER:
            r_ = row0.copy(); r_['row_class'] = c
            lv_row.append((c, predict_median(r_, model)))
    items.append((NICE['row_class'], lv_row))
    for p in [q for q in model['params'] if q != 'row_class']:
        lv = sorted(model['rf'][p], key=str) if p in model['params_cat'] else sorted(model['rf'][p], key=float)
        out = []
        for v in lv:
            r_ = row0.copy(); r_[p] = v
            out.append((v, predict_median(r_, model)))
        items.append((NICE[p], out))
    items.sort(key=lambda t: -(max(v for _, v in t[1]) - min(v for _, v in t[1])))

    a, b = st.columns([1, 1.25])
    with a:
        names = [t[0] for t in items][::-1]
        spans = [max(v for _, v in t[1]) - min(v for _, v in t[1]) for t in items][::-1]
        fig = go.Figure(go.Bar(y=names, x=spans, orientation='h', marker_color=PALETTE[0],
                               text=['±%.4f' % (s_ / 2) for s_ in spans], textposition='outside',
                               cliponaxis=False, hovertemplate='%{y}<br>rentang g %{x:.4f}<extra></extra>'))
        fig.update_xaxes(title='rentang median g antar level', range=[0, max(spans) * 1.3])
        show_chart(style_fig(fig, 'Urutan pengaruh untuk desain ini', 420, legend=False))
    with b:
        max_lv = max(len(t[1]) for t in items)
        z = np.full((len(items), max_lv), np.nan)
        txt = np.full((len(items), max_lv), '', dtype=object)
        for i, (nm, lvls) in enumerate(items):
            p_key = [k for k, v in NICE.items() if v == nm][0]
            cnts = ({g: int((df[GEO] == g).sum()) for g in geos} if p_key == GEO else
                    (model['rowc'] if p_key == 'row_class' and ROW_AS_MULTIPLIER else model['cnt'].get(p_key, {})))
            for j, (v, g_) in enumerate(lvls):
                z[i, j] = g_
                n_ = cnts.get(v)
                mark = '' if n_ is None or n_ >= MIN_SAMPLES else ' ·tipis'
                is_cur = str(v) == str(row0[p_key]) if isinstance(v, str) else abs(float(v) - float(row0[p_key])) < 1e-9
                txt[i, j] = '%s%s%s<br>%.4f' % ('▶ ' if is_cur else '', lv_txt(v), mark, g_)
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        lo_, hi_ = min(zmin, G_TARGET * 0.98), max(zmax, G_TARGET * 1.02)
        mid = (G_TARGET - lo_) / (hi_ - lo_)
        fig = go.Figure(go.Heatmap(
            z=z, text=txt, texttemplate='%{text}', textfont=dict(size=11), zmin=lo_, zmax=hi_,
            colorscale=[[0, DIV_LOW], [mid, DIV_MID], [1, DIV_HIGH]], xgap=3, ygap=3,
            hovertemplate='%{text}<extra></extra>', colorbar=dict(title='median g', thickness=12)))
        fig.update_yaxes(tickvals=list(range(len(items))), ticktext=[t[0] for t in items],
                         autorange='reversed', showgrid=False)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        show_chart(style_fig(fig, 'Median g tiap level (▶ = desain input)', 420, legend=False))
        st.markdown('<span class="muted">Hijau tosca = di bawah %.2f · krem = sekitar batas · merah bata = di atas. '
                    '"tipis" = level dengan data < %d blast.</span>' % (G_TARGET, MIN_SAMPLES),
                    unsafe_allow_html=True)

    with st.expander('Faktor jumlah baris dan konstanta geologi'):
        rr_ = [{'Kelas baris': c, 'Faktor': round(model['rowf'][c], 3), 'Jumlah blast': model['rowc'][c]}
               for c in ROW_CLASSES if c in model['rowf']]
        if rr_:
            show_table(pd.DataFrame(rr_))
        kk = [{'Geologi': g, 'k': round(model['base'][g][0], 0), 'n': round(model['base'][g][1], 3),
               'R²': round(model['base'][g][2], 3), 'Jumlah blast': model['base'][g][3]} for g in geos]
        show_table(pd.DataFrame(kk))


# =====================================================================
# TAB 4 - VALIDASI
# =====================================================================
with tab_val:
    st.markdown('Setiap blast diprediksi oleh model yang dikalibrasi **tanpa** blast itu '
                '(validasi per event). Inilah gambaran akurasi untuk blast yang belum terjadi.')
    ratio = df[COL_G].values / cv_pred
    lam_used = np.array([factor_for(g, fac)['lam'] for g in df[GEO]])
    covered = cv_pred * lam_used >= df[COL_G].values
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi('MAE', '%.4f g' % mae, 'rata-rata selisih prediksi vs aktual')
    with c2:
        kpi('NMAE', '%.1f%%' % (100 * mae / (df[COL_G].max() - df[COL_G].min())), 'MAE ÷ rentang g data')
    with c3:
        kpi('Prediksi kekecilan', '%d / %d' % ((ratio > 1).sum(), len(df)), 'ideal ± separuh')
    with c4:
        kpi('Design value menutupi', '%.0f%%' % (100 * covered.mean()),
            '%d dari %d blast · target %d%%' % (covered.sum(), len(df), DESIGN_PCT))

    a, b = st.columns([1.2, 1])
    with a:
        hi = float(max(df[COL_G].max(), cv_pred.max()) * 1.1)
        lam = fac['*']['lam']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode='lines', name='tepat (1:1)',
                                 line=dict(color=MUTED, width=1.5), hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[0, hi / lam], y=[0, hi], mode='lines', name='design value (×%.2f)' % lam,
                                 line=dict(color=MUTED, width=1.5, dash='dash'), hoverinfo='skip'))
        for g in geos:
            x = df[df[GEO] == g]
            pv = cv_pred[df[GEO].values == g]
            fig.add_trace(go.Scatter(
                x=pv, y=x[COL_G], mode='markers', name='%s (n=%d)' % (g, len(x)),
                marker=dict(color=GEO_COLOR.get(g, PALETTE[3]), symbol=GEO_SYMBOL.get(g, 'diamond'),
                            size=10, line=dict(color='white', width=1)),
                customdata=np.stack([x['tanggal'], x[COL_DIST], x[COL_CHG], x['row_number'].astype(int)], axis=1),
                hovertemplate='<b>%{customdata[0]}</b> · %{customdata[1]:.0f} m · %{customdata[2]:.0f} kg · '
                              '%{customdata[3]} baris<br>prediksi %{x:.4f} · aktual %{y:.4f}<extra></extra>'))
        fig.update_xaxes(title='prediksi g (median)', range=[0, hi])
        fig.update_yaxes(title='g terukur', range=[0, hi])
        show_chart(style_fig(fig, 'Prediksi vs aktual', 480))
    with b:
        st.markdown('##### Keseimbangan prediksi')
        st.markdown('<span class="muted">Rasio aktual ÷ prediksi. 1,00 = tepat · > 1 = model kekecilan · '
                    '< 1 = model kebesaran.</span>', unsafe_allow_html=True)
        bias = [{'Kelompok': 'Geologi: %s' % g, 'Blast': int((df[GEO] == g).sum()),
                 'Rasio median': round(float(np.median(ratio[df[GEO].values == g])), 2),
                 'MAE': round(float(np.mean(np.abs(df[COL_G].values - cv_pred)[df[GEO].values == g])), 4)}
                for g in geos]
        bias += [{'Kelompok': 'Baris: %s' % c, 'Blast': int((df['row_class'] == c).sum()),
                  'Rasio median': round(float(np.median(ratio[df['row_class'].values == c])), 2),
                  'MAE': round(float(np.mean(np.abs(df[COL_G].values - cv_pred)[df['row_class'].values == c])), 4)}
                 for c in ROW_CLASSES if (df['row_class'] == c).any()]
        bias.append({'Kelompok': 'Semua', 'Blast': len(df), 'Rasio median': round(float(np.median(ratio)), 2),
                     'MAE': round(mae, 4)})
        show_table(pd.DataFrame(bias))
        st.markdown('##### Konstanta model')
        show_table(pd.DataFrame([
            {'Parameter': 'λ design value (P%d)' % DESIGN_PCT, 'Nilai': '%.3f' % fac['*']['lam']},
            {'Parameter': 'σ Monte Carlo', 'Nilai': '%.3f' % fac['*']['sigma']},
            {'Parameter': 'Mode λ', 'Nilai': 'per geologi' if LAMBDA_PER_GEOLOGI else 'tunggal'},
            {'Parameter': 'Jumlah baris', 'Nilai': 'pengali langsung' if ROW_AS_MULTIPLIER else 'dalam weighting factor'},
        ]))
