from fastapi import FastAPI
from pydantic import BaseModel
from prophet import Prophet
import pandas as pd

app = FastAPI()

# Load model on startup
model = None

@app.on_event("startup")
def load_model():
    global model
    data = pd.read_csv('hospital_data.csv', parse_dates=['date'])
    df = data.rename(columns={'date': 'ds', 'admissions': 'y'})[['ds','y']]
    model = Prophet(yearly_seasonality=True)
    model.fit(df)

class ForecastRequest(BaseModel):
    days_ahead: int

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    future = model.make_future_dataframe(periods=req.days_ahead)
    forecast = model.predict(future)
    forecast_result = forecast[['ds','yhat']].tail(req.days_ahead).to_dict(orient='records')
    return {"forecast": forecast_result}