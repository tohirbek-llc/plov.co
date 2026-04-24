import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

LOCATIONS = ['Mitte', 'Kreuzberg', 'Prenzlauer Berg', 'Neukölln', 'Friedrichshain']

# ─── LOAD MODEL ───────────────────────────────────────────
def load_model():
    path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

# ─── LOAD HISTORICAL AVERAGES ─────────────────────────────
def load_historical(location):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'features.csv'))
    loc_df = df[df['location'] == location]
    if len(loc_df) == 0:
        return 750, 750, 750  # defaults if no history
    lag1 = loc_df['revenue_eur'].iloc[-1] if len(loc_df) >= 1 else 750
    lag7 = loc_df['revenue_eur'].iloc[-7] if len(loc_df) >= 7 else 750
    roll = loc_df['revenue_eur'].tail(7).mean()
    return lag1, lag7, roll

# ─── LOCATION ENCODER ─────────────────────────────────────
LOCATION_ENCODING = {
    'Friedrichshain': 0,
    'Kreuzberg':      1,
    'Mitte':          2,
    'Neukölln':       3,
    'Prenzlauer Berg':4
}

# ─── PREDICT FOR ONE SCENARIO ─────────────────────────────
def predict_revenue(model, location, weather, date, hours_open=7):
    day_of_week  = date.weekday()
    is_weekend   = int(day_of_week in [5, 6])
    month        = date.month
    week_of_year = date.isocalendar()[1]
    temp_avg     = (weather['temp_max'] + weather['temp_min']) / 2
    temp_range   = weather['temp_max'] - weather['temp_min']
    is_rainy     = int(weather['precipitation'] > 1)
    is_good      = int(not is_rainy)
    loc_encoded  = LOCATION_ENCODING.get(location, 2)
    lag1, lag7, roll = load_historical(location)

    features = pd.DataFrame([{
        'temp_avg':        temp_avg,
        'temp_range':      temp_range,
        'precipitation':   weather['precipitation'],
        'windspeed':       weather['windspeed'],
        'is_rainy':        is_rainy,
        'is_good_weather': is_good,
        'day_of_week':     day_of_week,
        'is_weekend':      is_weekend,
        'month':           month,
        'week_of_year':    week_of_year,
        'location_encoded':loc_encoded,
        'lag_1_revenue':   lag1,
        'lag_7_revenue':   lag7,
        'rolling_7_avg':   roll,
        'hours_open':      hours_open
    }])

    return model.predict(features)[0]

# ─── GET TOMORROW'S WEATHER ───────────────────────────────
def get_tomorrow_weather():
    import requests
    from datetime import timedelta
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
        "temp_max":     d["temperature_2m_max"][0],
        "temp_min":     d["temperature_2m_min"][0],
        "precipitation":d["precipitation_sum"][0],
        "windspeed":    d["windspeed_10m_max"][0]
    }

# ─── MAIN RECOMMENDER ─────────────────────────────────────
def recommend(hours_open=7):
    print("🚚 Plov.co — Location Recommender")
    print("=" * 40)

    model   = load_model()
    weather = get_tomorrow_weather()
    tomorrow = datetime.today().replace(hour=0, minute=0, second=0) 
    
    print(f"\n🌤️  Tomorrow's Berlin Weather:")
    print(f"   Temp: {weather['temp_min']}°C — {weather['temp_max']}°C")
    print(f"   Rain: {weather['precipitation']}mm")
    print(f"   Wind: {weather['windspeed']} km/h")
    print(f"\n📍 Predicted Revenue by Location:")

    predictions = {}
    for loc in LOCATIONS:
        pred = predict_revenue(model, loc, weather, tomorrow, hours_open)
        predictions[loc] = round(pred, 2)
        print(f"   {loc:<20} → €{pred:.0f}")

    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top 3 Recommendations for Tomorrow:")
    medals = ["🥇", "🥈", "🥉"]
    for i, (loc, rev) in enumerate(ranked[:3]):
        print(f"   {medals[i]} {loc:<20} → €{rev:.0f} predicted")

    print("\n✅ Good luck tomorrow!")

# ─── RUN ──────────────────────────────────────────────────
if __name__ == "__main__":
    recommend(hours_open=7)