
# Demand Forecasting ML API (M5 Dataset)
Production-style ML system for real-world demand forecasting with API deployment and Docker containerization.

## Overview
End-to-end machine learning system for demand forecasting using real retail time-series data (Walmart M5 dataset).

This project demonstrates:
- Time-series feature engineering
- Model training and evaluation
- API-based inference
- Dockerized deployment

---

## Dataset
- Source: Walmart M5 Forecasting Dataset
- Data: Daily product-level sales
- Approach: Single time-series modeling

Note: Dataset is not included due to GitHub file size limits. Download separately and place in data/.

---

## Feature Engineering

Lag Features:
- lag_1
- lag_7
- lag_14

Rolling Features:
- rolling_mean_7
- rolling_mean_14

Calendar Features:
- day_of_week
- month

---

## Model
- RandomForestRegressor (scikit-learn)
- Time-based train/test split (80/20)

---

## Metrics

MAE: 0.6895  
RMSE: 0.8965  
MAPE: 47.76%  
WAPE: 50.75%  

Note: MAPE/WAPE adjusted due to sparse demand.

---

## API Endpoints

Health:
GET /health

Predict:
POST /predict

Example:
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"lag_1":10,"lag_7":8,"lag_14":9,"rolling_mean_7":9,"rolling_mean_14":10,"day_of_week":2,"month":3}'

Metrics:
GET /metrics

---

## Run Locally

uvicorn app.main:app --reload

---

## Run with Docker

docker build -t demand-api-real .
docker run -p 8000:8000 demand-api-real

---

## Project Structure

app/
model/
data/
saved_models/
logs/
Dockerfile
requirements.txt

---

## Key Learnings
- Time-series forecasting with lag and rolling features
- Handling sparse demand
- FastAPI deployment
- Docker containerization

---

## Future Improvements
- Multi-series forecasting
- XGBoost
- Hyperparameter tuning
- Cloud deployment

