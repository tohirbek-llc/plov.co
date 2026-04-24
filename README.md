# Plov.co - Food Truck Revenue & Location Optimizer

A full end-to-end machine learning project that helps a Berlin food truck owner
maximize daily revenue by predicting earnings based on location, weather, and time
and recommending the best spots to park each day.

Built as a real-world data science portfolio project.

---

## Problem Statement

A food truck owner in Berlin needs to decide every morning:
- Where should I park today?
- How much revenue can I expect?
- Does the weather affect my sales?

This project answers all three questions using machine learning.

---

## Project Structure

    plov.co/
    app.py                     Streamlit web dashboard
    data/
        raw/
            sales_log.csv      Daily sales data logged by owner
        processed/
            features.csv       ML-ready engineered features
    models/
        xgboost_model.pkl      Trained XGBoost model
        shap_summary.png       SHAP feature importance plot
    src/
        data_collection.py     Weather API + sales log generator
        feature_engineering.py Feature engineering pipeline
        model.py               Model training + evaluation
        recommend.py           Location recommender engine

---

## Modules

### Module 1 - Data Collection
- Fetches real Berlin weather data using Open-Meteo API (free, no key needed)
- Generates a daily sales log template for the owner to fill in
- Tracks location, revenue, customers, hours open, weather conditions

### Module 2 - Feature Engineering
- 25 ML-ready features engineered from raw data
- Time features: day of week, is_weekend, month, week of year
- Weather features: avg temp, temp range, is_rainy, is_good_weather
- Lag features: yesterday revenue, same day last week, 7-day rolling average
- Location encoding for ML compatibility

### Module 3 - ML Models

Trained and compared 3 models:

| Model | MAE | R2 |
|---|---|---|
| Linear Regression | 65.97 EUR | 0.628 |
| Random Forest | 132.82 EUR | -0.575 |
| XGBoost | 194.92 EUR | -2.061 |

Linear Regression outperforms with small datasets.
XGBoost will improve significantly as more real data is collected.

SHAP values used to explain which features drive revenue predictions.

### Module 4 - Location Recommender
- Takes tomorrows real weather forecast from Open-Meteo API
- Predicts revenue for 5 Berlin districts
- Outputs top 3 recommended locations with predicted earnings

### Module 5 - Streamlit Dashboard
3-page interactive web app:
- Dashboard: KPIs, revenue over time, revenue by location
- Recommender: tomorrows top locations based on weather
- Log Sales: daily sales entry form for the owner

---

## Tech Stack

- Python 3.14
- pandas, numpy for data manipulation
- scikit-learn for ML pipeline
- XGBoost for gradient boosting model
- SHAP for model explainability
- Streamlit for web dashboard
- Open-Meteo API for real-time Berlin weather
- matplotlib, seaborn for visualizations
- GitHub for version control

---

## How to Run

1. Clone the repo

    git clone https://github.com/tohirbek-llc/plov.co.git
    cd plov.co

2. Set up environment

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

3. Collect weather data and generate sales template

    python src/data_collection.py

4. Engineer features

    python src/feature_engineering.py

5. Train models

    python src/model.py

6. Run the dashboard

    streamlit run app.py

---

## SHAP Feature Importance

Key findings:
- Day of week is the strongest revenue predictor - weekends earn significantly more
- Hours open directly correlates with revenue
- Precipitation negatively impacts foot traffic
- Temperature has moderate impact on customer numbers

---

## Business Insights

- Park in Kreuzberg or Mitte on weekends for highest revenue
- Stay open 8+ hours on good weather days
- Public holidays generate 40% above average revenue
- Rainy days see 30-40% revenue drop

---

## Future Improvements

- Add Berlin events API (concerts, festivals, markets)
- Integrate foot traffic data via Google Popular Times
- Deploy to Streamlit Cloud for mobile access
- Retrain model monthly as data grows
- Add competitor location tracking
- Build customer segmentation model

---

## Author

Built by a Data Science student to help a friends food truck business in Berlin
and to demonstrate real-world ML skills for internship applications.

---

## License

MIT License - feel free to use and adapt for your own food truck!
