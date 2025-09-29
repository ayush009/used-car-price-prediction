# app.py — DriveWorth • Used Car Value Studio
# Supercharged Streamlit UI: glassmorphism, animated header, themed controls, dependent dropdowns, background image

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from pathlib import Path
import base64
import requests
import time

# ============ CONFIG ============
st.set_page_config(page_title="DriveWorth • Used Car Value Studio", page_icon="🚗", layout="centered")

# --- Paths (change as needed) ---
MODEL_PATH = Path("results/models/RF_app_best_model.pkl")
# Portable CSV (RAW GitHub). If you want local, point to your file path.
CSV_PATH = "https://raw.githubusercontent.com/ayush009/used-car-price-prediction/main/data/Used_Car_Price_Prediction.csv"

# Background image (unchanged per your request)
BG_URL_RAW = "https://raw.githubusercontent.com/ayush009/used-car-price-prediction/main/Images/tao-yuan-tGTwk6JBBok-unsplash.jpg"

# Accent palette
ACCENT      = "#10b981"  # emerald
ACCENT_SOFT = "rgba(16,185,129,0.25)"
INK         = "#e5f8f1"
PANEL       = "rgba(17,24,39,0.55)"
BORDER      = "rgba(16,185,129,0.45)"


# ===== Live FX: INR → EUR (cached 5 min) =====
@st.cache_data(ttl=300, show_spinner=False)
def fetch_fx_inr_eur():
    """Returns (rate_inr_to_eur, rate_eur_to_inr, iso_timestamp)."""
    urls = [
        "https://api.exchangerate.host/latest?base=INR&symbols=EUR",
        "https://api.frankfurter.app/latest?from=INR&to=EUR",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            if "rates" in data and "EUR" in data["rates"]:
                rate = float(data["rates"]["EUR"])
                ts = data.get("date") or data.get("time_last_update_utc") or time.strftime("%Y-%m-%d")
                return rate, (1.0 / rate), ts
        except Exception:
            continue
    fallback = 0.011
    return fallback, (1.0 / fallback), "offline-fallback"


# ============ BACKGROUND ============
def set_background_from_url(url: str, dim: float = 0.40):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.content
        b64 = base64.b64encode(data).decode()
    except Exception as e:
        st.error("Failed to load background image. (GitHub RAW URL)")
        with st.expander("Image error details"):
            st.code(repr(e))
        b64 = ""

    dim = max(0.0, min(1.0, float(dim)))
    css = f"""
    <style>
      :root {{
        --accent: {ACCENT};
        --accent-soft: {ACCENT_SOFT};
        --ink: {INK};
        --panel: {PANEL};
        --border: {BORDER};
      }}

      .stApp {{
        background:
          linear-gradient(rgba(0,0,0,{dim}), rgba(0,0,0,{dim})),
          url("data:image/jpg;base64,{b64}") center/cover no-repeat fixed;
      }}
      .stApp [data-testid="stHeader"] {{ background: transparent; }}
      .stApp .block-container {{
        background: rgba(0,0,0,0.06);
        border-radius: 18px;
        padding-top: 0.6rem;
        padding-bottom: 1.0rem;
      }}

      .dw-hero {{ display:flex; align-items:center; gap:.6rem; margin:.25rem 0 1.0rem 0; }}
      .dw-title {{
        font-size: clamp(2.0rem, 3.8vw, 3.4rem);
        font-weight: 800; line-height:1.05; letter-spacing:.4px; margin:0;
        background: linear-gradient(90deg, var(--ink), var(--accent), #60a5fa);
        -webkit-background-clip:text; background-clip:text; color:transparent;
        animation: hueRotate 10s linear infinite; text-shadow:0 8px 30px rgba(0,0,0,.35);
      }}
      @keyframes hueRotate {{ 0% {{filter:hue-rotate(0)}}
                              100% {{filter:hue-rotate(360deg)}} }}

      .dw-card {{
        background: var(--panel); border:1px solid var(--border);
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border-radius:18px; padding:18px 18px 10px 18px; box-shadow:0 10px 26px rgba(0,0,0,.28);
      }}
      .dw-card h2, .dw-card h3, .dw-card p, .dw-card label {{ color: var(--ink) !important; }}

      .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {{
        background: rgba(2,6,23,0.55); border:1px solid rgba(148,163,184,0.22); color:var(--ink); border-radius:12px;
      }}
      .stSelectbox:hover > div > div, .stTextInput:hover > div > div, .stNumberInput:hover > div > div {{
        border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);
      }}
      .stNumberInput button, .stSelectbox svg, .stTextInput svg {{ color: var(--ink); }}

      .stButton > button {{
        background: linear-gradient(135deg, var(--accent), #34d399); color:#06281b; border:none; font-weight:800;
        border-radius:12px; padding:.7rem 1.05rem; box-shadow:0 12px 24px rgba(16,185,129,.35);
      }}
      .stButton > button:hover {{ transform: translateY(-1px); filter: brightness(1.03); }}

      .stAlert > div {{ background: rgba(2,6,23,0.65); border:1px solid rgba(148,163,184,0.25); color: var(--ink); }}

      .dw-price {{
        font-size: clamp(1.6rem, 3.2vw, 2.4rem); font-weight:900; color:var(--ink); text-align:center;
        padding:.6rem 1rem; border:2px dashed var(--border); border-radius:16px; background: rgba(2,6,23,0.55);
      }}
      .dw-kicker {{ text-align:center; color:var(--ink); opacity:.88; font-size:.9rem; }}

      .dw-fadein {{ animation: fadeInUp .6s ease both; }}
      @keyframes fadeInUp {{ 0% {{opacity:0; transform:translateY(10px)}} 100% {{opacity:1; transform:none}} }}

      /* dropdown menu visible + readable */
      html, body, .stApp, .stApp .block-container {{ overflow: visible !important; }}
      .stApp [data-baseweb="popover"] {{ z-index: 999999 !important; }}
      .stApp [data-baseweb="menu"] {{
        background: rgba(2,6,23,0.92); border:1px solid rgba(148,163,184,0.35);
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
      }}
      .stApp [data-baseweb="menu"] [role="option"] {{ color: var(--ink); }}
      .stApp [data-baseweb="menu"] [role="option"][aria-selected="true"],
      .stApp [data-baseweb="menu"] [role="option"]:hover {{ background: rgba(16,185,129,0.18); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background_from_url(BG_URL_RAW, dim=0.40)


# ============ HEADER ============
st.markdown(
    """
    <div class="dw-hero dw-fadein">
      <div style="font-size:2.2rem; filter: drop-shadow(0 6px 12px rgba(0,0,0,.35));">🚗</div>
      <h1 class="dw-title">DriveWorth — Used Car Value Studio</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="dw-card dw-fadein" style="margin-top:.2rem;">
      <p style="margin:0;">
        Get a fast, data-driven estimate of your vehicle’s resale value — all dropdowns pull valid options from the dataset and update dependently.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============ DATA LOAD (ONE TIME, CACHED) ============
@st.cache_data(show_spinner=False)
def load_df(csv_path):
    """Load & normalize the dataset; adds helper columns and typed owners/year."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return pd.DataFrame()

    # normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # ensure essential columns exist
    needed = [
        "make", "model", "fuel_type", "transmission", "body_type",
        "registered_city", "registered_state", "total_owners",
        "kms_run", "year"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # normalize text columns
    for c in ["make", "model", "fuel_type", "transmission", "body_type", "registered_city", "registered_state"]:
        df[c] = df[c].astype(str).str.lower().str.strip()

    # owners: extract first int (handles "2 owners") and clip to >=1
    owners_raw = df["total_owners"].astype(str).str.extract(r"(\d+)", expand=False)
    df["total_owners"] = pd.to_numeric(owners_raw, errors="coerce").clip(lower=1)

    # kms
    df["kms_run"] = pd.to_numeric(df["kms_run"], errors="coerce")

    # year / yr_mfr
    if "yr_mfr" in df.columns:
        df["yr_mfr"] = pd.to_numeric(df["yr_mfr"], errors="coerce")
    else:
        df["yr_mfr"] = pd.to_numeric(df["year"], errors="coerce")

    return df

df_all = load_df(CSV_PATH)


# ============ HELPERS ============
def filtered(df, **conds):
    """Return df filtered by equality on provided columns (strings lowercased)."""
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for col, val in conds.items():
        if val is None or val == "":
            continue
        if col not in df.columns:
            continue
        mask &= (df[col] == str(val).lower().strip())
    return df.loc[mask]

def options_from(df, col):
    if df.empty or col not in df.columns:
        return []
    opts = sorted(x for x in df[col].dropna().unique().tolist() if str(x).strip() and str(x).lower().strip() != "nan")
    return opts

def owners_options(df):
    if df.empty or "total_owners" not in df.columns:
        return [1, 2, 3]
    vals = (
        df["total_owners"].dropna().astype(int).clip(lower=1).unique().tolist()
    )
    vals = sorted(set(vals)) or [1, 2, 3]
    return vals

def safe_index(options, value):
    try:
        return options.index(value)
    except Exception:
        return 0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============ MODEL ============
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH.as_posix())

try:
    pipe = load_model()
except Exception as _e:
    pipe = None
    st.error("Model failed to load — check MODEL_PATH.")
    with st.expander("Show error details"):
        st.code(repr(_e))


# ============ FORM ============
st.markdown('<div class="dw-card dw-fadein">', unsafe_allow_html=True)
st.subheader("Enter Vehicle Details", anchor=False)

# 1) Make → Model
all_makes = options_from(df_all, "make")
make = st.selectbox("Car Brand (Make)", options=all_makes or ["mercedes"], index=safe_index(all_makes, "mercedes"))
df_make = filtered(df_all, make=make)

models_for_make = options_from(df_make, "model")
model = st.selectbox("Car Model", options=models_for_make or ["c-class"], index=safe_index(models_for_make, "c-class"))
df_car = filtered(df_make, model=model)

# 2) Fuel / Transmission / Body (from rows for the chosen make+model)
fuels = options_from(df_car, "fuel_type")
fuel_type = st.selectbox("Fuel Type", options=fuels or ["petrol"])

transmissions = options_from(df_car, "transmission")
transmission = st.selectbox("Transmission", options=transmissions or ["manual"])

bodies = options_from(df_car, "body_type")
body_type = st.selectbox("Body Type", options=bodies or ["sedan"])

# 3) Owners (from chosen make+model and F/T/B filter)
df_car_ftb = filtered(df_car, fuel_type=fuel_type, transmission=transmission, body_type=body_type)
owner_opts = owners_options(df_car_ftb if not df_car_ftb.empty else df_car)
total_owners = st.selectbox("Number of Previous Owners", options=owner_opts, index=min(len(owner_opts)//2, len(owner_opts)-1))

# 4) State → City (dataset-driven)
states = options_from(df_all, "registered_state")
registered_state = st.selectbox("Registered State", options=states or ["berlin"])
df_state = filtered(df_all, registered_state=registered_state)

cities = options_from(df_state, "registered_city")
city = st.selectbox("Registered City", options=cities or ["berlin"])

# 5) Year & KMs — bounds from the filtered car subset (robust clamping)
this_year = datetime.datetime.now().year

yr_series = df_car_ftb["yr_mfr"].dropna().astype(int) if not df_car_ftb.empty else df_car["yr_mfr"].dropna().astype(int)
yr_min = int(yr_series.min()) if not yr_series.empty else 1990
yr_max = int(yr_series.max()) if not yr_series.empty else this_year
year_lo = max(1990, yr_min)
year_hi = max(year_lo, yr_max)
year_default = clamp(2018, year_lo, year_hi)

yr_mfr = st.number_input(
    "Year of Manufacture",
    min_value=int(year_lo),
    max_value=int(year_hi),
    value=int(year_default),
    step=1,
)

kms_series = df_car_ftb["kms_run"].dropna().astype(int) if not df_car_ftb.empty else df_car["kms_run"].dropna().astype(int)
kms_min = int(kms_series.min()) if not kms_series.empty else 0
kms_max = int(kms_series.max()) if not kms_series.empty else 1_000_000
kms_lo = max(0, kms_min)
kms_hi = max(kms_lo + 500, kms_max)  # ensure hi >= lo + step
kms_default = clamp(45_000, kms_lo, kms_hi)

kms_run = st.number_input(
    "Kilometers Driven",
    min_value=int(kms_lo),
    max_value=int(kms_hi),
    value=int(kms_default),
    step=500,
)

st.markdown("</div>", unsafe_allow_html=True)  # /dw-card


# Live preview
car_age = max(0, datetime.datetime.now().year - int(yr_mfr))
st.markdown(
    f"""
    <div class="dw-card dw-fadein" style="margin-top:12px;">
      <h3 style="margin:.1rem 0 .6rem 0;">Quick Preview</h3>
      <p style="margin:.2rem 0;">
        <b>Make/Model:</b> {make.title()} {model.upper()} •
        <b>Fuel:</b> {fuel_type.title()} •
        <b>Transmission:</b> {transmission.title()} •
        <b>Body:</b> {body_type.title()}
      </p>
      <p style="margin:.2rem 0;">
        <b>Year:</b> {yr_mfr} (<i>{car_age}-year old</i>) •
        <b>KMs:</b> {kms_run:,.0f} •
        <b>Owners:</b> {int(total_owners)}
      </p>
      <p style="margin:.2rem 0;">
        <b>Registration:</b> {city.title()}, {registered_state.upper()}
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Build input row (exact columns your model expects)
row = {
    "yr_mfr": int(yr_mfr),
    "kms_run": float(kms_run),
    "make": make.lower().strip(),
    "model": model.lower().strip(),
    "fuel_type": fuel_type.lower().strip(),
    "transmission": transmission.lower().strip(),
    "body_type": body_type.lower().strip(),
    "total_owners": int(total_owners),
    "city": city.lower().strip(),
    "registered_state": registered_state.lower().strip(),
    "car_age": car_age,
}
X_one = pd.DataFrame([row])


# ============ PREDICT ============
clicked = st.button("Estimate Value", use_container_width=True)
if clicked:
    if pipe is None:
        st.error("No model loaded — cannot predict. Verify your model path.")
    else:
        try:
            pred_inr = float(pipe.predict(X_one)[0])
            rate_inr_eur, rate_eur_inr, ts = fetch_fx_inr_eur()
            pred_eur = pred_inr * rate_inr_eur

            st.balloons()
            st.markdown(
                f'''
                <div class="dw-card dw-fadein" style="margin-top:12px;">
                  <div class="dw-price">
                    💰 Estimated Resale Value:
                    ₹{pred_inr:,.0f} <span style="opacity:.85;">(€{pred_eur:,.0f})</span>
                  </div>
                  <div class="dw-kicker">
                    FX now: <b>1 INR = {rate_inr_eur:.6f} EUR</b> • <b>1 EUR = {rate_eur_inr:.4f} INR</b>
                    &nbsp;|&nbsp; <span style="opacity:.85;">Updated: {ts}</span>
                  </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error("Prediction failed. The model may expect different columns/formats or encodings.")
            with st.expander("Show error details"):
                st.code(repr(e))


# ============ FOOTER ============
st.markdown(
    """
    <div class="dw-fadein" style="text-align:center; margin-top:12px; opacity:.85;">
      <small style="color:var(--ink);">
        DriveWorth • Powered by your trained model. Background loaded from your GitHub image. <br/>
        All dropdowns are dataset-driven; predictions shown in ₹ with live (€) conversion.
      </small>
    </div>
    """,
    unsafe_allow_html=True,
)
