import streamlit as st
import pandas as pd
import pickle
import os
import sys
from datetime import datetime, timedelta
import requests

# ─── PATH SETUP ───────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from recommend import predict_revenue, load_historical, LOCATION_ENCODING, LOCATIONS
from feature_engineering import engineer_features

# ─── CONFIG ───────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(__file__), 'models')
RAW_DIR     = os.path.join(os.path.dirname(__file__), 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')

st.set_page_config(page_title="Plov.co Dashboard", page_icon="🚚", layout="wide")

# ─── LOAD MODEL ───────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

# ─── FETCH WEATHER ────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_weather():
    tomorrow = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.52, "longitude": 13.405,
        "daily": ["temperature_2m_max", "temperature_2m_min",
                  "precipitation_sum", "windspeed_10m_max"],
        "start_date": tomorrow, "end_date": tomorrow,
        "timezone": "Europe/Berlin"
    }
    r = requests.get(url, params=params)
    d = r.json()["daily"]
    return {
        "temp_max":      d["temperature_2m_max"][0],
        "temp_min":      d["temperature_2m_min"][0],
        "precipitation": d["precipitation_sum"][0],
        "windspeed":     d["windspeed_10m_max"][0]
    }

# ─── SIDEBAR ──────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/delivery-truck.png", width=80)
st.sidebar.title("Plov.co 🥘")
st.sidebar.markdown("**Berlin Food Truck Analytics**")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🗺️ Recommender", "📝 Log Sales"])

model   = load_model()
weather = fetch_weather()

# ══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Sales Dashboard")

    df = pd.read_csv(os.path.join(RAW_DIR, 'sales_log.csv'))
    df['revenue_eur'] = pd.to_numeric(df['revenue_eur'], errors='coerce')
    df['date']        = pd.to_datetime(df['date'])
    df_clean          = df.dropna(subset=['revenue_eur'])

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue",    f"€{df_clean['revenue_eur'].sum():,.0f}")
    col2.metric("Avg Daily Revenue",f"€{df_clean['revenue_eur'].mean():,.0f}")
    col3.metric("Best Day",         f"€{df_clean['revenue_eur'].max():,.0f}")
    col4.metric("Days Logged",      f"{len(df_clean)}")

    st.divider()

    # Revenue over time
    st.subheader("💰 Revenue Over Time")
    st.line_chart(df_clean.set_index('date')['revenue_eur'])

    # Revenue by location
    st.subheader("📍 Revenue by Location")
    loc_df = df_clean.groupby('location')['revenue_eur'].mean().sort_values(ascending=False)
    st.bar_chart(loc_df)

    # Raw data
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df_clean)

# ══════════════════════════════════════════════════════════
# PAGE 2 — RECOMMENDER
# ══════════════════════════════════════════════════════════
elif page == "🗺️ Recommender":
    st.title("🗺️ Tomorrow's Location Recommender")

    # Weather display
    st.subheader("🌤️ Tomorrow's Berlin Weather")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Max Temp",     f"{weather['temp_max']}°C")
    w2.metric("Min Temp",     f"{weather['temp_min']}°C")
    w3.metric("Rain",         f"{weather['precipitation']}mm")
    w4.metric("Wind",         f"{weather['windspeed']} km/h")

    st.divider()

    hours = st.slider("⏱️ How many hours will you be open?", 4, 12, 7)
    tomorrow = datetime.today() + timedelta(days=1)

    # Predict for all locations
    predictions = {}
    for loc in LOCATIONS:
        pred = predict_revenue(model, loc, weather, tomorrow, hours)
        predictions[loc] = round(pred, 0)

    pred_df = pd.DataFrame(list(predictions.items()),
                           columns=['Location', 'Predicted Revenue (€)'])
    pred_df = pred_df.sort_values('Predicted Revenue (€)', ascending=False).reset_index(drop=True)

    # Top 3
    st.subheader("🏆 Top 3 Locations for Tomorrow")
    medals = ["🥇", "🥈", "🥉"]
    for i, row in pred_df.head(3).iterrows():
        st.success(f"{medals[i]} **{row['Location']}** → €{row['Predicted Revenue (€)']:.0f} predicted")

    st.divider()
    st.subheader("📊 All Locations")
    st.bar_chart(pred_df.set_index('Location'))

# ══════════════════════════════════════════════════════════
# PAGE 3 — LOG SALES
# ══════════════════════════════════════════════════════════
elif page == "📝 Log Sales":
    st.title("📝 Log Today's Sales")
    st.markdown("Fill in your sales data for today and hit **Save**!")

    with st.form("sales_form"):
        col1, col2 = st.columns(2)
        with col1:
            date        = st.date_input("📅 Date", datetime.today())
            location    = st.selectbox("📍 Location", LOCATIONS)
            hours_open  = st.number_input("⏱️ Hours Open", 1, 16, 7)
        with col2:
            revenue     = st.number_input("💰 Revenue (€)", 0, 10000, 500)
            customers   = st.number_input("👥 Customers", 0, 1000, 50)
            notes       = st.text_input("📝 Notes", "")

        submitted = st.form_submit_button("💾 Save Entry")

        if submitted:
            new_row = {
                'date':        str(date),
                'temp_max':    weather['temp_max'],
                'temp_min':    weather['temp_min'],
                'precipitation': weather['precipitation'],
                'windspeed':   weather['windspeed'],
                'weathercode': 0,
                'location':    location,
                'revenue_eur': revenue,
                'customers':   customers,
                'hours_open':  hours_open,
                'notes':       notes
            }
            df = pd.read_csv(os.path.join(RAW_DIR, 'sales_log.csv'))
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(os.path.join(RAW_DIR, 'sales_log.csv'), index=False)
            st.success("✅ Entry saved successfully!")
            st.balloons()