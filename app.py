# app.py — DriveWorth • Used Car Value Studio
# Streamlit app with dependent dropdowns, INR→EUR live conversion, and corrected owner logic

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from pathlib import Path
import base64, requests, time

# ============ CONFIG ============
st.set_page_config(page_title="DriveWorth • Used Car Value Studio", page_icon="🚗", layout="centered")

MODEL_PATH = Path("results/models/RF_app_best_model.pkl")
CSV_PATH   = Path("Used_Car_Price_Prediction.csv")  # local file in repo or uploaded

BG_URL_RAW = "https://raw.githubusercontent.com/ayush009/used-car-price-prediction/main/Images/tao-yuan-tGTwk6JBBok-unsplash.jpg"

ACCENT      = "#10b981"
ACCENT_SOFT = "rgba(16,185,129,0.25)"
INK         = "#e5f8f1"
PANEL       = "rgba(17,24,39,0.55)"
BORDER      = "rgba(16,185,129,0.45)"


# ============ UTILITIES ============
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_fx_inr_eur():
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
                return rate, 1.0/rate, ts
        except Exception:
            continue
    fallback = 0.011
    return fallback, 1.0/fallback, "offline"

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


# ============ DATA LOADER ============
RAW_GITHUB_CSV = "https://raw.githubusercontent.com/ayush009/used-car-price-prediction/main/data/Used_Car_Price_Prediction.csv"
CSV_CANDIDATES = [CSV_PATH, Path(__file__).parent / CSV_PATH.name, Path("/mount/data/") / CSV_PATH.name, RAW_GITHUB_CSV]

@st.cache_data(show_spinner=False)
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    for c in ["make","model","fuel_type","transmission","body_type","registered_city","registered_state","city"]:
        if c not in df.columns: df[c] = np.nan
        df[c] = df[c].astype(str).str.lower().str.strip()
    df["total_owners"] = pd.to_numeric(df["total_owners"], errors="coerce").clip(lower=1)
    df = df[df["total_owners"] <= 4]
    df["kms_run"] = pd.to_numeric(df["kms_run"], errors="coerce")
    if "yr_mfr" not in df.columns: df["yr_mfr"] = pd.to_numeric(df["year"], errors="coerce")
    else: df["yr_mfr"] = pd.to_numeric(df["yr_mfr"], errors="coerce")
    return df.dropna(subset=["make","model"])

@st.cache_data(show_spinner=False)
def load_df(uploaded_file, candidates):
    if uploaded_file is not None:
        return _normalize_df(pd.read_csv(uploaded_file, low_memory=False))
    for c in candidates:
        try:
            if isinstance(c, Path) and c.exists():
                return _normalize_df(pd.read_csv(c.as_posix(), low_memory=False))
            elif isinstance(c, str) and c.startswith(("http://","https://")):
                return _normalize_df(pd.read_csv(c, low_memory=False))
        except Exception: continue
    raise FileNotFoundError("Dataset not found. Upload it below or place 'Used_Car_Price_Prediction.csv' next to app.py.")


# ============ MODEL LOADER ============
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path.as_posix())


# ============ UI SETUP ============
set_background_from_url(BG_URL_RAW)

st.markdown(
    """
    <div class="dw-hero" style="display:flex;align-items:center;gap:.6rem;margin:.25rem 0 1rem 0;">
      <div style="font-size:2.2rem;">🚗</div>
      <h1 class="dw-title">DriveWorth — Used Car Value Studio</h1>
    </div>
    """, unsafe_allow_html=True
)

with st.expander("📄 Dataset source (optional)"):
    st.write("Upload your dataset if the app can’t find it locally or via GitHub RAW.")
    uploaded_csv = st.file_uploader("Upload Used_Car_Price_Prediction.csv", type=["csv"])

# Load data & model
try:
    df_all = load_df(uploaded_csv, CSV_CANDIDATES)
except Exception as e:
    st.error("⚠️ Could not load dataset.")
    st.code(repr(e))
    st.stop()

try:
    pipe = load_model(MODEL_PATH)
except Exception as e:
    pipe = None
    st.error("⚠️ Model failed to load.")
    with st.expander("Details"): st.code(repr(e))


# ============ FORM ============
st.markdown('<div class="dw-card">', unsafe_allow_html=True)
st.subheader("Enter Vehicle Details", anchor=False)

makes = options_from(df_all, "make") or ["mercedes"]
make = st.selectbox("Car Brand (Make)", makes)
df_make = filtered(df_all, make=make)

models = options_from(df_make, "model") or ["c-class"]
model = st.selectbox("Car Model", models)
df_car = filtered(df_make, model=model)

fuels = options_from(df_car, "fuel_type") or ["petrol"]
fuel_type = st.selectbox("Fuel Type", fuels)

transmissions = options_from(df_car, "transmission") or ["manual"]
transmission = st.selectbox("Transmission", transmissions)

bodies = options_from(df_car, "body_type") or ["sedan"]
body_type = st.selectbox("Body Type", bodies)

df_car_ftb = filtered(df_car, fuel_type=fuel_type, transmission=transmission, body_type=body_type)
owner_opts = sorted(pd.Series(df_car_ftb["total_owners"].dropna().astype(int)).unique().tolist() or [1,2,3,4])
total_owners = st.selectbox("Number of Previous Owners", owner_opts)

states = options_from(df_all, "registered_state") or ["berlin"]
registered_state = st.selectbox("Registered State", states)
df_state = filtered(df_all, registered_state=registered_state)
cities = options_from(df_state, "registered_city") or ["berlin"]
city = st.selectbox("Registered City", cities)

this_year = datetime.datetime.now().year
yr_series = df_car_ftb["yr_mfr"].dropna().astype(int) if not df_car_ftb.empty else df_car["yr_mfr"].dropna().astype(int)
yr_min = int(yr_series.min()) if not yr_series.empty else 1990
yr_max = int(yr_series.max()) if not yr_series.empty else this_year
yr_mfr = st.number_input("Year of Manufacture", min_value=yr_min, max_value=yr_max, value=clamp(2018, yr_min, yr_max), step=1)

kms_series = df_car_ftb["kms_run"].dropna().astype(int) if not df_car_ftb.empty else df_car["kms_run"].dropna().astype(int)
kms_min = int(kms_series.min()) if not kms_series.empty else 0
kms_max = int(kms_series.max()) if not kms_series.empty else 1_000_000
kms_run = st.number_input("Kilometers Driven", min_value=kms_min, max_value=kms_max, value=clamp(45000, kms_min, kms_max), step=500)

st.markdown('</div>', unsafe_allow_html=True)

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

X_one = pd.DataFrame([{
    "yr_mfr": int(yr_mfr), "kms_run": float(kms_run), "make": make, "model": model,
    "fuel_type": fuel_type, "transmission": transmission, "body_type": body_type,
    "total_owners": int(total_owners), "city": city, "registered_state": registered_state,
    "car_age": car_age,
}])

# ============ PREDICTION ============
clicked = st.button("Estimate Value", use_container_width=True)
if clicked:
    if pipe is None:
        st.error("⚠️ No model loaded.")
    else:
        try:
            base_inr = float(pipe.predict(X_one)[0])
            depr_rate = 0.07
            n_extra = max(0, int(total_owners) - 1)
            adj_factor = (1.0 - depr_rate) ** n_extra
            adj_inr = max(0.0, base_inr * adj_factor)
            rate_inr_eur, rate_eur_inr, ts = fetch_fx_inr_eur()
            adj_eur  = adj_inr * rate_inr_eur
            st.balloons()
            st.markdown(
                f"""
                <div class="dw-card">
                  <div class="dw-price">
                    💰 Estimated Resale Value:<br/>
                    ₹{adj_inr:,.0f} <span style="opacity:.85;">(€{adj_eur:,.0f})</span>
                  </div>
                  <div class="dw-kicker">
                    Owners adj: −7% × {n_extra} → ×{adj_factor:.3f}<br/>
                    FX: 1 INR = {rate_inr_eur:.6f} EUR • Updated: {ts}
                  </div>
                </div>
                """, unsafe_allow_html=True
            )
        except Exception as e:
            st.error("Prediction failed.")
            with st.expander("Error details"): st.code(repr(e))

# ============ FOOTER ============
st.markdown('<div class="dw-kicker">DriveWorth • RF model • Dataset-driven UI • Live INR→EUR</div>', unsafe_allow_html=True)
