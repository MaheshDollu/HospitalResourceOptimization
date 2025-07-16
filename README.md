# Hospital Resource Optimization Project

This is a comprehensive project to simulate, forecast, classify, optimize, and visualize hospital resource usage including patient admissions, ICU needs, and ventilator usage.

---

## Setup Instructions

1. **Clone or download the project files**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Simulate and generate hospital data**

```bash
python data_pipeline.py
```

4. **Run forecasting script**

```bash
python forecasting.py
```

5. **Run ICU admission classification**

```bash
python classification.py
```

6. **Run resource optimization demo**

```bash
python optimization.py
```

7. **Start Streamlit dashboard**

```bash
streamlit run dashboard.py
```

8. **Start FastAPI server**

```bash
uvicorn api:app --reload
```

---

## Project Structure

- `data_pipeline.py` : Simulates hospital data and saves to CSV  
- `forecasting.py` : Forecasts daily admissions using Prophet  
- `classification.py` : Predicts ICU admission using Random Forest  
- `optimization.py` : Simple resource allocation optimization demo  
- `dashboard.py` : Streamlit dashboard for visualizing forecasts and stats  
- `api.py` : FastAPI endpoint for forecasting service  
- `requirements.txt` : Python dependencies  
- `README.md` : This guide  

---

## Next Steps

- Use real hospital data for training  
- Add more features for classification  
- Enhance optimization model with linear programming  
- Integrate dashboard with live API data  
- Add authentication and security  
- Automate ETL workflows  

---