import requests
import pandas as pd
from datetime import datetime, timedelta
import os

#___config___
BERLIN_LAT = 52.5200
BERLIN_LON = 13.4050
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)


#___fetch weather____
def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily weather for Berlin between start_date and end_date.
    Dates format: 'YYYY-MM-DD'
    """
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": BERLIN_LAT,
        "longitude": BERLIN_LON,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "weathercode"
        ],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Europe/Berlin"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["daily"]

    df = pd.DataFrame({
        "date": data["time"],
        "temp_max": data["temperature_2m_max"],
        "temp_min": data["temperature_2m_min"],
        "precipitation": data["precipitation_sum"],
        "windspeed": data["windspeed_10m_max"],
        "weathercode": data["weathercode"]
    })

    df["date"] = pd.to_datetime(df["date"])
    return df



# ─── CREATE SALES LOG TEMPLATE ────────────────────────────
def create_sales_template(start_date: str, end_date: str):
    """
    Creates a CSV your friend fills in daily with sales info.
    """
    weather_df = fetch_weather(start_date, end_date)

    weather_df["location"] = ""          # e.g. Mitte, Kreuzberg
    weather_df["revenue_eur"] = ""       # total revenue that day
    weather_df["customers"] = ""         # number of customers
    weather_df["hours_open"] = ""        # how many hours open
    weather_df["notes"] = ""             # any notes (event nearby etc)

    output_path = os.path.join(DATA_DIR, "sales_log.csv")
    weather_df.to_csv(output_path, index=False)
    print(f"✅ Sales template created at: {output_path}")
    print(weather_df.head())


# ─── RUN ──────────────────────────────────────────────────
if __name__ == "__main__":
    today = datetime.today().strftime("%Y-%m-%d")
    thirty_days = (datetime.today() + timedelta(days=7)).strftime("%Y-%m-%d")
    create_sales_template(today, thirty_days)
       