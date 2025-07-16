import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
from forecasting import prepare_prophet_df, train_forecast
from classification import create_features  # reuse your feature function

# Cache hospital data load
@st.cache_data
def load_data():
    return pd.read_csv('hospital_data.csv', parse_dates=['date'])

# Cache ICU classification model load
@st.cache_resource
def load_icu_model():
    return joblib.load('icu_classifier.pkl')  # make sure you have saved this model!

# Plotly forecast plot
def plot_forecast_plotly(forecast_df):
    fig = px.line(forecast_df, x='ds', y='yhat', title="1-Year Admissions Forecast")
    st.plotly_chart(fig)

# Predict ICU admissions using classifier
def predict_icu_admissions(df, model):
    df_feat = create_features(df)
    X = df_feat[['admissions', 'day_of_week', 'month', 'adm_lag1']]
    preds = model.predict(X)
    df['icu_pred'] = preds
    return df

# Main Streamlit function
def main():
    st.title("🏥 Hospital Resource Optimization Dashboard")

    data = load_data()
    icu_model = load_icu_model()

    # Prepare data for forecasting
    df_prophet = prepare_prophet_df(data)
    train = df_prophet.iloc[:-90]
    
    # Long-term forecast: 365 days
    model, forecast = train_forecast(train, periods=365)

    st.subheader("📈 1-Year Admissions Forecast")
    plot_forecast_plotly(forecast)

    # Show latest real metrics
    latest_admissions = int(data['admissions'].iloc[-1])
    latest_icu = int(data['icu_admissions'].iloc[-1])
    latest_vent = int(data['ventilator_usage'].iloc[-1])

    st.metric("Latest Admissions", latest_admissions)
    st.metric("Latest ICU Admissions (Real)", latest_icu)
    st.metric("Latest Ventilator Usage", latest_vent)

    # ICU Prediction
    data_with_icu_pred = predict_icu_admissions(data, icu_model)
    predicted_icu_today = data_with_icu_pred['icu_pred'].iloc[-1]
    st.metric("Predicted ICU Admission (Today)", predicted_icu_today)

    # Resource limits (you can change or make these user inputs)
    ventilators_available = st.number_input("Ventilators Available", min_value=1, value=20)
    beds_available = st.number_input("ICU Beds Available", min_value=1, value=40)

    # Simple alert logic
    ventilator_demand = int(predicted_icu_today * 0.7)  # assume 70% ICU need ventilators
    if predicted_icu_today > beds_available or ventilator_demand > ventilators_available:
        st.warning(f"⚠️ Alert! Predicted ICU admissions ({predicted_icu_today}) or ventilator demand ({ventilator_demand}) may exceed available resources!")
    else:
        st.success("✅ Resources are sufficient for today's predicted ICU admissions.")

    st.info("✅ This dashboard is now extended with a 1-year forecast. You can further enhance it with live data feeds and optimization logic.")

# Run the app
if __name__ == "__main__":
    main()
