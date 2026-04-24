import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# ─── CONFIG ───────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─── WEATHER CODE MAPPER ──────────────────────────────────
# Open-Meteo weather codes → human readable
def map_weathercode(code):
    if code == 0:
        return "clear"
    elif code in [1, 2, 3]:
        return "cloudy"
    elif code in [51, 53, 55, 61, 63, 65]:
        return "rainy"
    elif code in [71, 73, 75, 77]:
        return "snowy"
    elif code in [80, 81, 82]:
        return "showers"
    elif code in [95, 96, 99]:
        return "stormy"
    else:
        return "other"

# ─── MAIN FEATURE ENGINEERING ─────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Time features
    df["day_of_week"] = df["date"].dt.dayofweek        # 0=Monday, 6=Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # ── Weather features
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["weather_label"] = df["weathercode"].apply(map_weathercode)
    df["is_rainy"] = df["weather_label"].isin(["rainy", "showers", "stormy"]).astype(int)
    df["is_good_weather"] = df["weather_label"].isin(["clear", "cloudy"]).astype(int)

    # ── Location encoding
    le = LabelEncoder()
    df["location_encoded"] = le.fit_transform(df["location"].fillna("unknown"))

    # ── Revenue per hour (efficiency metric)
    df["revenue_per_hour"] = pd.to_numeric(df["revenue_eur"], errors="coerce") / \
                              pd.to_numeric(df["hours_open"], errors="coerce")

    # ── Lag features (previous day's revenue)
    df["revenue_eur"] = pd.to_numeric(df["revenue_eur"], errors="coerce")
    df["lag_1_revenue"] = df["revenue_eur"].shift(1)
    df["lag_7_revenue"] = df["revenue_eur"].shift(7)

    # ── Rolling average (last 7 days)
    df["rolling_7_avg"] = df["revenue_eur"].shift(1).rolling(window=7).mean()

    return df

# ─── SAVE PROCESSED DATA ──────────────────────────────────
def process_and_save():
    input_path = os.path.join(RAW_DIR, "sales_log.csv")
    df = pd.read_csv(input_path)

    print(f"📥 Loaded {len(df)} rows from sales_log.csv")

    df_featured = engineer_features(df)

    output_path = os.path.join(PROCESSED_DIR, "features.csv")
    df_featured.to_csv(output_path, index=False)

    print(f"✅ Features saved to: {output_path}")
    print(f"\n📊 Feature columns: {list(df_featured.columns)}")
    print(df_featured.head())

# ─── RUN ──────────────────────────────────────────────────
if __name__ == "__main__":
    process_and_save()