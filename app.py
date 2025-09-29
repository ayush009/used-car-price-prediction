# app.py — DriveWorth • Used Car Value Studio
# Supercharged Streamlit UI: glassmorphism, animated header, themed controls, dependent dropdowns, background image

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from pathlib import Path
import base64
import requests  # keep: we’re using the GitHub RAW image URL

# ============ CONFIG ============
st.set_page_config(page_title="DriveWorth • Used Car Value Studio", page_icon="🚗", layout="centered")

# Update these for your environment
MODEL_PATH = Path("results/models/RF_app_best_model.pkl")
# Use raw GitHub CSV instead of local file
CSV_PATH = "https://raw.githubusercontent.com/ayush009/used-car-price-prediction/main/data/Used_Car_Price_Prediction.csv"


# Your GitHub image (RAW) — DO NOT CHANGE
BG_URL_RAW = "https://raw.githubusercontent.com/ayush009/used-car-price-prediction/main/Images/tao-yuan-tGTwk6JBBok-unsplash.jpg"

# Accent palette
ACCENT     = "#10b981"  # emerald
ACCENT_SOFT= "rgba(16,185,129,0.25)"
INK        = "#e5f8f1"  # very light mint ink on dark bg
PANEL      = "rgba(17,24,39,0.55)"  # glass dark
BORDER     = "rgba(16,185,129,0.45)"


# ============ BACKGROUND ============
def set_background_from_url(url: str, dim: float = 0.40):
    """Embed a full-bleed background fetched from a URL, with a dark gradient overlay."""
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

      /* full-page bg with dim overlay */
      .stApp {{
        background:
          linear-gradient(rgba(0,0,0,{dim}), rgba(0,0,0,{dim})),
          url("data:image/jpg;base64,{b64}") center/cover no-repeat fixed;
      }}
      /* remove default header bg */
      .stApp [data-testid="stHeader"] {{ background: transparent; }}
      /* container glass look */
      .stApp .block-container {{
        background: rgba(0,0,0,0.06);
        border-radius: 18px;
        padding-top: 0.6rem;
        padding-bottom: 1.0rem;
      }}

      /* ===== Animated Title ===== */
      .dw-hero {{
        position: relative;
        display: flex;
        align-items: center;
        gap: .6rem;
        margin: .25rem 0 1.0rem 0;
      }}
      .dw-title {{
        font-size: clamp(2.0rem, 3.8vw, 3.4rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: .4px;
        margin: 0;
        background: linear-gradient(90deg, var(--ink), var(--accent), #60a5fa);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: hueRotate 10s linear infinite;
        text-shadow: 0 8px 30px rgba(0,0,0,.35);
      }}
      @keyframes hueRotate {{
        0% {{ filter: hue-rotate(0deg); }}
        100% {{ filter: hue-rotate(360deg); }}
      }}
      /* little car that gently floats */
      .dw-car {{
        font-size: clamp(1.8rem, 3vw, 2.4rem);
        animation: floaty 3.8s ease-in-out infinite;
        filter: drop-shadow(0 6px 12px rgba(0,0,0,.35));
      }}
      @keyframes floaty {{
        0%   {{ transform: translateY(0px) }}
        50%  {{ transform: translateY(-6px) }}
        100% {{ transform: translateY(0px) }}
      }}

      /* ===== Cards & Inputs ===== */
      .dw-card {{
        background: var(--panel);
        border: 1px solid var(--border);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 10px 26px rgba(0,0,0,.28);
      }}
      .dw-card h2, .dw-card h3, .dw-card p, .dw-card label {{
        color: var(--ink) !important;
      }}

      /* Streamlit core widgets restyle */
      .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {{
        background: rgba(2,6,23,0.55);
        border: 1px solid rgba(148,163,184,0.22);
        color: var(--ink);
        border-radius: 12px;
      }}
      .stSelectbox:hover > div > div, .stTextInput:hover > div > div, .stNumberInput:hover > div > div {{
        border-color: var(--accent);
        box-shadow: 0 0 0 3px var(--accent-soft);
      }}
      .stNumberInput button, .stSelectbox svg, .stTextInput svg {{
        color: var(--ink);
      }}

      .stButton > button {{
        background: linear-gradient(135deg, var(--accent), #34d399);
        color: #06281b;
        border: none;
        font-weight: 800;
        border-radius: 12px;
        padding: .7rem 1.05rem;
        box-shadow: 0 12px 24px rgba(16,185,129,.35);
        transition: transform .08s ease, box-shadow .15s ease, filter .15s ease;
      }}
      .stButton > button:hover {{
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 16px 32px rgba(16,185,129,.45);
      }}
      .stButton > button:active {{ transform: translateY(1px) scale(.99); }}

      /* info/success boxes tint for dark bg */
      .stAlert > div {{
        background: rgba(2,6,23,0.65);
        border: 1px solid rgba(148,163,184,0.25);
        color: var(--ink);
      }}

      /* result banner */
      .dw-price {{
        font-size: clamp(1.6rem, 3.2vw, 2.4rem);
        font-weight: 900;
        color: var(--ink);
        text-align: center;
        padding: .6rem 1rem;
        border: 2px dashed var(--border);
        border-radius: 16px;
        background: rgba(2,6,23,0.55);
      }}
      .dw-kicker {{
        text-align:center; color: var(--ink); opacity: .88; font-size: .9rem;
      }}

      /* subtle entrance animation for sections */
      .dw-fadein {{ animation: fadeInUp .6s ease both; }}
      @keyframes fadeInUp {{
        0% {{ opacity: 0; transform: translateY(10px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
      }}

      /* --- FIX selectbox dropdown menu visibility & contrast --- */
      html, body, .stApp, .stApp .block-container {{ overflow: visible !important; }}
      .stApp [data-baseweb="popover"] {{ z-index: 999999 !important; }}
      .stApp [data-baseweb="menu"] {{
        background: rgba(2,6,23,0.92);
        border: 1px solid rgba(148,163,184,0.35);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
      }}
      .stApp [data-baseweb="menu"] [role="option"] {{ color: var(--ink); }}
      .stApp [data-baseweb="menu"] [role="option"][aria-selected="true"],
      .stApp [data-baseweb="menu"] [role="option"]:hover {{
        background: rgba(16,185,129,0.18);
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Inject background using the GitHub RAW URL
set_background_from_url(BG_URL_RAW, dim=0.40)

# ============ HEADER ============
st.markdown(
    """
    <div class="dw-hero dw-fadein">
      <div class="dw-car">🚗</div>
      <h1 class="dw-title">DriveWorth — Used Car Value Studio</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dw-card dw-fadein" style="margin-top:.2rem;">
      <p style="margin:0;">
        Get a fast, data-driven estimate of your vehicle’s resale value — with smart, dependent dropdowns for make/model and state/city.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============ LOADERS ============
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH.as_posix())

@st.cache_data
def load_dropdown_maps(csv_path: Path):
    """Return dropdown sources normalized to lowercase."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return [], {}, [], {}

    df.columns = (
        df.columns.str.strip().str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )
    for col in ("make", "model", "registered_city", "registered_state"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(str).str.lower().str.strip()

    brands = sorted(x for x in df["make"].dropna().unique().tolist() if x and x != "nan")
    brand_to_models = (
        df.dropna(subset=["make", "model"])
          .groupby("make")["model"]
          .apply(lambda s: sorted(s.dropna().unique().tolist()))
          .to_dict()
    )

    states = sorted(x for x in df["registered_state"].dropna().unique().tolist() if x and x != "nan")
    state_to_cities = (
        df.dropna(subset=["registered_state", "registered_city"])
          .groupby("registered_state")["registered_city"]
          .apply(lambda s: sorted(s.dropna().unique().tolist()))
          .to_dict()
    )

    return brands, brand_to_models, states, state_to_cities

# Load model
try:
    pipe = load_model()
except Exception as _e:
    pipe = None
    st.error("Model failed to load — check MODEL_PATH.")
    with st.expander("Show error details"):
        st.code(repr(_e))

brands, brand_to_models, states, state_to_cities = load_dropdown_maps(CSV_PATH)

# ============ FORM ============
st.markdown('<div class="dw-card dw-fadein">', unsafe_allow_html=True)
st.subheader("Enter Vehicle Details", anchor=False)

col1, col2 = st.columns(2)
with col1:
    yr_mfr = st.number_input(
        "Year of Manufacture",
        min_value=1990, max_value=datetime.datetime.now().year, value=2018, step=1
    )
with col2:
    kms_run = st.number_input(
        "Kilometers Driven",
        min_value=0, max_value=1_000_000, value=45_000, step=500
    )

# Make → Model
if brands:
    default_brand_idx = brands.index("mercedes") if "mercedes" in brands else 0
    make = st.selectbox("Car Brand (Make)", options=brands, index=default_brand_idx)
    models_for_make = brand_to_models.get(make, [])
    if models_for_make:
        default_model_idx = models_for_make.index("c-class") if "c-class" in models_for_make else 0
        model = st.selectbox("Car Model", options=models_for_make, index=default_model_idx)
    else:
        model = st.text_input("Car Model", value="c-class")
else:
    make  = st.text_input("Car Brand (Make)", value="mercedes")
    model = st.text_input("Car Model", value="c-class")

fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "cng", "electric", "hybrid", "unknown"])
transmission = st.selectbox("Transmission", ["manual", "automatic", "unknown"])
body_type = st.selectbox("Body Type", ["sedan", "hatchback", "suv", "coupe", "convertible", "mpv", "unknown"])

col3, col4 = st.columns(2)
with col3:
    total_owners = st.number_input(
        "Number of Previous Owners",
        min_value=0, max_value=3, value=1, step=1
    )
with col4:
    pass  # balance

# State → City
if states:
    default_state_idx = states.index("berlin") if "berlin" in states else 0
    registered_state = st.selectbox("Registered State", options=states, index=default_state_idx)
    cities_for_state = state_to_cities.get(registered_state, [])
    if cities_for_state:
        default_city_idx = cities_for_state.index("berlin") if "berlin" in cities_for_state else 0
        city = st.selectbox("Registered City", options=cities_for_state, index=default_city_idx)
    else:
        city = st.text_input("Registered City", value="berlin")
else:
    registered_state = st.text_input("Registered State", value="berlin")
    city = st.text_input("Registered City", value="berlin")

st.markdown("</div>", unsafe_allow_html=True)  # /dw-card

# Live preview card
car_age = datetime.datetime.now().year - int(yr_mfr)
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
        <b>Owners:</b> {total_owners}
      </p>
      <p style="margin:.2rem 0;">
        <b>Registration:</b> {city.title()}, {registered_state.upper()}
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Build input row (exact 11 fields used by the APP model)
row = {
    "yr_mfr": yr_mfr,
    "kms_run": kms_run,
    "make": make.lower().strip(),
    "model": model.lower().strip(),
    "fuel_type": fuel_type,
    "transmission": transmission,
    "body_type": body_type,
    "total_owners": total_owners,
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
            pred = float(pipe.predict(X_one)[0])
            st.balloons()
            st.markdown(f'<div class="dw-card dw-fadein" style="margin-top:12px;">'
                        f'<div class="dw-price">💰 Estimated Resale Value: €{pred:,.0f}</div>'
                        f'<div class="dw-kicker">Note: Currency equals the dataset’s units; treat as an index if markets are mixed.</div>'
                        f'</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("Prediction failed. The model may expect different columns/formats.")
            with st.expander("Show error details"):
                st.code(repr(e))

# ============ FOOTER ============
st.markdown(
    """
    <div class="dw-fadein" style="text-align:center; margin-top:12px; opacity:.85;">
      <small style="color:var(--ink);">
        DriveWorth • Powered by your trained model. Background: local Unsplash image. <br/>
        UI theme crafted for high contrast over dark asphalt + green foliage.
      </small>
    </div>
    """,
    unsafe_allow_html=True,
)
