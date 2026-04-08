# Demand Forecasting ML API (M5 Dataset)

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
- Approach: Single time-series modeling (one product)

---

## Feature Engineering
The following features were created:

### Lag Features
- lag_1 (previous day)
- lag_7 (weekly pattern)
- lag_14

### Rolling Statistics
- rolling_mean_7
- rolling_mean_14

### Calendar Features
- day_of_week
- month

---

## Model
- RandomForestRegressor (sklearn)
- Trained on 80% of data
- Tested on 20% (time-based split)

---

## Evaluation Metrics

| Metric | Value |
|--------|------|
| MAE | 0.6895 |
| RMSE | 0.8965 |
| MAPE | 47.76% |
| WAPE | 50.75% |

Note:
MAPE/WAPE are adjusted due to sparse demand (many zero sales days).

---

## API Endpoints

### Health Check
GET /health
### Predict Demand
POST /predict
Example:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "lag_1": 10,
  "lag_7": 8,
  "lag_14": 9,
  "rolling_mean_7": 9,
  "rolling_mean_14": 10,
  "day_of_week": 2,
  "month": 3
}'

Model Metrics
GET /metrics

Run Locally

uvicorn app.main:app --reload

Run with Docker

Build image

docker build -t demand-api-real .

Run container

docker run -p 8000:8000 demand-api-real


⸻

Project Structure

.
├── app/
├── model/
├── data/
├── saved_models/
├── logs/
├── Dockerfile
├── requirements.txt
└── README.md


⸻

Key Learnings
	•	Time-series forecasting using lag and rolling features
	•	Handling sparse demand in retail datasets
	•	Building production-style ML APIs using FastAPI
	•	Containerizing ML services with Docker
	•	Exposing model metrics for monitoring
