import pandas as pd
import numpy as np

def simulate_hospital_data(start_date='2022-01-01', end_date='2023-12-31'):
    np.random.seed(42)
    days = pd.date_range(start_date, end_date)
    n = len(days)

    # Simulate daily admissions (trend + seasonality + noise)
    trend = np.linspace(50, 120, n)
    seasonality = 15 * np.sin(2 * np.pi * days.dayofyear / 365.25)
    noise = np.random.normal(0, 7, n)
    admissions = np.clip(trend + seasonality + noise, 0, None).astype(int)

    # ICU admission rate ~ 20% of daily admissions + some randomness
    icu_rate = 0.20 + 0.05 * np.sin(2 * np.pi * days.dayofyear / 180)
    icu_admissions = np.array([np.random.binomial(adm, min(max(rate, 0),1)) for adm, rate in zip(admissions, icu_rate)])

    # Equipment usage (ventilators) ~ proportional to ICU admissions + noise
    ventilators = np.clip(icu_admissions * (0.7 + 0.3 * np.random.rand(n)), 0, None).astype(int)

    df = pd.DataFrame({
        'date': days,
        'admissions': admissions,
        'icu_admissions': icu_admissions,
        'ventilator_usage': ventilators
    })

    return df

def save_data(df, filename='hospital_data.csv'):
    df.to_csv(filename, index=False)

def load_data(filename='hospital_data.csv'):
    return pd.read_csv(filename, parse_dates=['date'])

if __name__ == "__main__":
    df = simulate_hospital_data()
    save_data(df)
    print(f"Simulated data saved to hospital_data.csv")