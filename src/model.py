import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
import os

# ─── CONFIG ───────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── FEATURES TO USE ──────────────────────────────────────
FEATURES = [
    'temp_avg', 'temp_range', 'precipitation', 'windspeed',
    'is_rainy', 'is_good_weather', 'day_of_week', 'is_weekend',
    'month', 'week_of_year', 'location_encoded',
    'lag_1_revenue', 'lag_7_revenue', 'rolling_7_avg',
    'hours_open'
]
TARGET = 'revenue_eur'

# ─── LOAD DATA ────────────────────────────────────────────
def load_data():
    path = os.path.join(PROCESSED_DIR, 'features.csv')
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"📥 Loaded {len(df)} rows after dropping NaN rows")
    return df

# ─── TRAIN MODELS ─────────────────────────────────────────
def train_models(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost":           XGBRegressor(n_estimators=100, learning_rate=0.1,
                                          max_depth=4, random_state=42,
                                          verbosity=0)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        results[name] = {"model": model, "mae": mae, "r2": r2, "preds": preds}
        print(f"\n📊 {name}")
        print(f"   MAE : €{mae:.2f}")
        print(f"   R²  : {r2:.4f}")

    return results, X_train, X_test, y_train, y_test

# ─── SHAP EXPLANATION ─────────────────────────────────────
def explain_model(model, X_train, X_test):
    print("\n🔍 Generating SHAP explanation for XGBoost...")
    explainer  = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plot_path = os.path.join(MODELS_DIR, 'shap_summary.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"✅ SHAP plot saved to: {plot_path}")

# ─── SAVE BEST MODEL ──────────────────────────────────────
def save_model(model):
    import pickle
    path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {path}")

# ─── RUN ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    results, X_train, X_test, y_train, y_test = train_models(df)

    # Pick best model by MAE
    best_name = min(results, key=lambda x: results[x]['mae'])
    print(f"\n🏆 Best model: {best_name}")

    best_model = results[best_name]['model']
    explain_model(results["XGBoost"]["model"], X_train, X_test)
    save_model(results["XGBoost"]["model"])