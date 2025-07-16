import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from numpy import sqrt

def prepare_prophet_df(df, target_col='admissions'):
    return df.rename(columns={'date': 'ds', target_col: 'y'})[['ds','y']]

def train_forecast(df, periods=90):
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

def evaluate_forecast(test_df, forecast_df):
    merged = pd.merge(test_df, forecast_df[['ds','yhat']], on='ds')
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = sqrt(mean_squared_error(merged['y'], merged['yhat']))

    return mae, rmse

def plot_forecast(train_df, test_df, forecast_df):
    plt.figure(figsize=(12,6))
    plt.plot(train_df['ds'], train_df['y'], label='Train Actual')
    plt.plot(test_df['ds'], test_df['y'], label='Test Actual')
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
    plt.legend()
    plt.title('Hospital Admissions Forecast')
    plt.xlabel('Date')
    plt.ylabel('Admissions')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('hospital_data.csv', parse_dates=['date'])
    df = prepare_prophet_df(data)
    train = df.iloc[:-90]
    test = df.iloc[-90:]

    # Extend forecast to 365 days
    model, forecast = train_forecast(train, periods=365)

    # Optional: Evaluate only last 90 days if you want metrics
    mae, rmse = evaluate_forecast(test, forecast[-90:])
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    plot_forecast(train, test, forecast)