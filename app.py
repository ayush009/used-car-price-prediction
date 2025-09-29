# app.py — DriveWorth • Used Car Value Studio
# Dataset-driven UI • RF model prediction • Owners correction • Live INR→EUR

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from pathlib import Path
import base64, requests, time

# ============ CONFIG ============
st.set_page_config(page_title="DriveWorth • Used Car Value Studio", page_icon="🚗", layout="centered")

# Use your uploaded files (same folder as this script)
MODEL_PATH = Path("RF_app_best_model.pkl")
CSV_PATH   = Path("Used_Car_Price_Prediction.csv")

# Background image (unchanged per your request)
BG_URL_RAW = "https://raw.githubusercontent.com/ayush009/used-car-price-prediction/main/Images/tao-yuan-tGTwk6JBBok-unsplash.jpg"

# Theme
ACCENT      = "#10b981"
ACCENT_SOFT = "rgba(16,185,129,0.25)"
INK         = "#e5f8f1"
PANEL       = "rgba(17,24,39,0.55)"
BORDER      = "rgba(16,185,129,0.45)"


# ============ HELPERS ============

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

@st.cache_data(ttl=300, show_spinner=False)
def fetch_fx_inr_eur():
    """Live INR→EUR; fallback if offline."""
    urls = [
        "https://api.exchangerate.host/latest?base=INR&symbols=EUR",
        "https://api.frankfurter.app/latest?from=INR&to=EUR",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=8); r.raise_for_status()
            data = r.json()
            if "rates" in data and "EUR" in data["rates"]:
                rate = float(data["rates"]["EUR"])
                ts = data.get("date") or time.strftime("%Y-%m-%d %H:%M")
                return rate, 1.0 / rate, ts
        except Exception:
            continue
    fallback = 0.011
    return fallback, 1.0/fallback, "offline"

def set_background_from_url(url: str, dim: float = 0.40):
    try:
        r = requests.get(url, timeout=20); r.raise_for_status()
        b64 = base64.b64encode(r.content).decode()
    except Exception:
        b64 = ""
    css = f"""
    <style>
      :root {{
        --accent:{ACCENT}; --accent-soft:{ACCENT_SOFT}; --ink:{INK};
        --panel:{PANEL}; --border:{BORDER};
      }}
      .stApp {{
        background: linear-gradient(rgba(0,0,0,{dim}), rgba(0,0,0,{dim})),
          url("data:image/jpg;base64,{b64}") center/cover no-repeat fixed;
      }}
      .stApp [data-testid="stHeader"] {{ background: transparent; }}
      .stApp .block-container {{ background: rgba(0,0,0,0.06); border-radius:18px; }}
      .dw-title {{
        font-size:clamp(2.0rem,3.8vw,3.4rem); font-weight:800;
        background:linear-gradient(90deg,var(--ink),var(--accent),#60a5fa);
        -webkit-background-clip:text; background-clip:text; color:transparent;
        animation:hueRotate 10s linear infinite;
      }}
      @keyframes hueRotate {{ 0%{{filter:hue-rotate(0)}} 100%{{filter:hue-rotate(360deg)}} }}
      .dw-card {{ background:var(--panel); border:1px solid var(--border); border-radius:18px; padding:1rem; backdrop-filter: blur(12px); }}
      .dw-card * {{ color:var(--ink) !important; }}
      .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {{
        background: rgba(2,6,23,0.55); border:1px solid rgba(148,163,184,0.22); color:var(--ink); border-radius:12px;
      }}
      .stSelectbox:hover > div > div, .stTextInput:hover > div > div, .stNumberInput:hover > div > div {{
        border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);
      }}
      html, body, .stApp, .stApp .block-container {{ overflow:visible!important; }}
      .stApp [data-baseweb="popover"] {{ z-index: 999999 !important; }}
      .stApp [data-baseweb="menu"] {{
        background: rgba(2,6,23,0.92); border:1px solid rgba(148,163,184,0.35);
        backdrop-filter: blur(10px);
      }}
      .dw-price {{
        font-size:clamp(1.6rem,3.2vw,2.4rem); font-weight:900; text-align:center;
        border:2px dashed var(--border); border-radius:16px; padding:.6rem 1rem;
      }}
      .dw-kicker {{ text-align:center; font-size:.9rem; opacity:.88; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_df(csv_path: Path) -> pd.DataFrame:
    """Load & normalize dataset; cap owners at 4 (remove extreme outliers)."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # Ensure required columns exist
    for c in ["make","model","fuel_type","transmission","body_type",
              "registered_city","registered_state","total_owners",
              "kms_run","yr_mfr"]:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize text columns
    for c in ["make","model","fuel_type","transmission","body_type","registered_city","registered_state","city"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()

    # Owners numeric & capped
    df["total_owners"] = pd.to_numeric(df["total_owners"], errors="coerce").clip(lower=1)
    df = df[df["total_owners"] <= 4]  # drop 5+ owner outliers

    # Numerics
    df["kms_run"] = pd.to_numeric(df["kms_run"], errors="coerce")
    if "yr_mfr" in df.columns:
        df["yr_mfr"] = pd.to_numeric(df["yr_mfr"], errors="coerce")
    elif "year" in df.columns:
        df["yr_mfr"] = pd.to_numeric(df["year"], errors="coerce")

    return df.dropna(subset=["make","model"])

def options_from(df: pd.DataFrame, col: str):
    if df.empty or col not in df.columns: return []
    vals = df[col].dropna().astype(str).str.strip()
    vals = [v for v in vals.unique().tolist() if v and v.lower() != "nan"]
    return sorted(vals)

def filtered(df: pd.DataFrame, **eq):
    if df.empty: return df
    m = pd.Series(True, index=df.index)
    for k, v in eq.items():
        if v is None or v == "" or k not in df.columns: continue
        m &= (df[k] == str(v).lower().strip())
    return df[m]

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path.as_posix())


# ============ SETUP ============
set_background_from_url(BG_URL_RAW)

st.markdown(
    """
    <div class="dw-hero" style="display:flex;align-items:center;gap:.6rem;margin:.25rem 0 1rem 0;">
      <div style="font-size:2.2rem;">🚗</div>
      <h1 class="dw-title">DriveWorth — Used Car Value Studio</h1>
    </div>
    """, unsafe_allow_html=True
)
st.markdown(
    """
    <div class="dw-card">
      Get a fast, data-driven estimate of your vehicle’s resale value —
      all dropdowns are dataset-driven; predictions use your trained model.
    </div>
    """, unsafe_allow_html=True
)

# Sidebar: owner depreciation control (you can hide it later)
with st.sidebar:
    st.markdown("### Pricing Controls")
    owner_depr_pct = st.slider(
        "Owner depreciation per extra owner (%)",
        min_value=0, max_value=25, value=7, step=1,
        help="After-model rule: price × (1 - p)^(owners-1). Ensures more owners → lower price."
    )

# Load data & model
df_all = load_df(CSV_PATH)
try:
    pipe = load_model(MODEL_PATH)
except Exception as e:
    pipe = None
    st.error("⚠️ Model failed to load. Check RF_app_best_model.pkl is present.")
    with st.expander("Details"): st.code(repr(e))

# ============ FORM ============
st.markdown('<div class="dw-card">', unsafe_allow_html=True)
st.subheader("Enter Vehicle Details", anchor=False)

# Make → Model
makes = options_from(df_all, "make") or ["mercedes"]
make = st.selectbox("Car Brand (Make)", makes, index=0)
df_make = filtered(df_all, make=make)

models = options_from(df_make, "model") or ["c-class"]
model = st.selectbox("Car Model", models, index=0)
df_car = filtered(df_make, model=model)

# Fuel / Transmission / Body from the selected car subset
fuels = options_from(df_car, "fuel_type") or ["petrol"]
fuel_type = st.selectbox("Fuel Type", fuels, index=0)

transmissions = options_from(df_car, "transmission") or ["manual"]
transmission = st.selectbox("Transmission", transmissions, index=0)

bodies = options_from(df_car, "body_type") or ["sedan"]
body_type = st.selectbox("Body Type", bodies, index=0)

# Owners from the filtered rows (strictly from dataset, capped at 4)
df_car_ftb = filtered(df_car, fuel_type=fuel_type, transmission=transmission, body_type=body_type)
owner_opts = sorted(pd.Series(df_car_ftb["total_owners"].dropna().astype(int)).unique().tolist() or [1,2,3,4])
total_owners = st.selectbox("Number of Previous Owners", owner_opts, index=min(len(owner_opts)//2, len(owner_opts)-1))

# State → City
states = options_from(df_all, "registered_state") or ["berlin"]
registered_state = st.selectbox("Registered State", states, index=0)
df_state = filtered(df_all, registered_state=registered_state)
cities = options_from(df_state, "registered_city") or ["berlin"]
city = st.selectbox("Registered City", cities, index=0)

# Year & KMs (bounds from filtered car subset)
this_year = datetime.datetime.now().year
yr_series = df_car_ftb["yr_mfr"].dropna().astype(int) if not df_car_ftb.empty else df_car["yr_mfr"].dropna().astype(int)
yr_min = int(yr_series.min()) if not yr_series.empty else 1990
yr_max = int(yr_series.max()) if not yr_series.empty else this_year
year_lo, year_hi = max(1990, yr_min), max(yr_min, yr_max)
yr_mfr = st.number_input(
    "Year of Manufacture",
    min_value=int(year_lo), max_value=int(year_hi),
    value=int(clamp(2018, year_lo, year_hi)), step=1
)

kms_series = df_car_ftb["kms_run"].dropna().astype(int) if not df_car_ftb.empty else df_car["kms_run"].dropna().astype(int)
kms_min = int(kms_series.min()) if not kms_series.empty else 0
kms_max = int(kms_series.max()) if not kms_series.empty else 1_000_000
kms_lo, kms_hi = max(0, kms_min), max(kms_min + 500, kms_max)
kms_run = st.number_input(
    "Kilometers Driven",
    min_value=int(kms_lo), max_value=int(kms_hi),
    value=int(clamp(45_000, kms_lo, kms_hi)), step=500
)

st.markdown('</div>', unsafe_allow_html=True)

# Preview
car_age = max(0, this_year - int(yr_mfr))
st.markdown(
    f"""
    <div class="dw-card">
      <h3>Quick Preview</h3>
      <p><b>{make.title()} {model.upper()}</b> • {fuel_type} • {transmission} • {body_type}</p>
      <p>Year {yr_mfr} ({car_age} yrs) • {kms_run:,} kms • {int(total_owners)} owner(s)</p>
      <p>Registration: {city.title()}, {registered_state.upper()}</p>
    </div>
    """, unsafe_allow_html=True
)

# Row for model (must match training pipeline)
X_one = pd.DataFrame([{
    "yr_mfr": int(yr_mfr),
    "kms_run": float(kms_run),
    "make": make, "model": model,
    "fuel_type": fuel_type, "transmission": transmission, "body_type": body_type,
    "total_owners": int(total_owners),
    "city": city, "registered_state": registered_state,
    "car_age": car_age,
}])

# Predict with owners correction + live FX
clicked = st.button("Estimate Value", use_container_width=True)
if clicked:
    if pipe is None:
        st.error("⚠️ No model loaded — cannot predict.")
    else:
        try:
            # Base model prediction (INR)
            base_inr = float(pipe.predict(X_one)[0])

            # Monotonic correction: more owners → lower price
            per_owner = owner_depr_pct / 100.0   # from sidebar
            n_extra = max(0, int(total_owners) - 1)
            adj_factor = (1.0 - per_owner) ** n_extra
            adj_inr = max(0.0, base_inr * adj_factor)

            # Live FX
            rate_inr_eur, rate_eur_inr, ts = fetch_fx_inr_eur()
            base_eur = base_inr * rate_inr_eur
            adj_eur  = adj_inr  * rate_inr_eur

            st.balloons()
            st.markdown(
                f"""
                <div class="dw-card">
                  <div class="dw-price" style="margin-bottom:.4rem;">
                    💰 Estimated Resale Value:<br/>
                    ₹{adj_inr:,.0f} <span style="opacity:.85;">(€{adj_eur:,.0f})</span>
                  </div>
                  <div class="dw-kicker">
                    Base: ₹{base_inr:,.0f} ( €{base_eur:,.0f} ) • Owners adj: −{owner_depr_pct}% × {n_extra} → ×{adj_factor:.3f}<br/>
                    FX: 1 INR = {rate_inr_eur:.6f} EUR • 1 EUR = {rate_eur_inr:.4f} INR • Updated: {ts}
                  </div>
                </div>
                """, unsafe_allow_html=True
            )
        except Exception as e:
            st.error("Prediction failed. The model may expect different columns/encodings.")
            with st.expander("Error details"):
                st.code(repr(e))

# Footer
st.markdown('<div class="dw-kicker">DriveWorth • RF model prediction • Dataset-driven UI • Live INR→EUR</div>', unsafe_allow_html=True)
