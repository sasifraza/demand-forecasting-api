from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time
import json
import os
from datetime import datetime

app = FastAPI()

# Load model
model = joblib.load("saved_models/model.pkl")

# Logging setup
LOG_FILE = "logs/predictions.jsonl"
os.makedirs("logs", exist_ok=True)


# Input schema
class InputData(BaseModel):
    lag_1: float
    lag_7: float
    lag_14: float
    rolling_mean_7: float
    rolling_mean_14: float
    day_of_week: int
    month: int


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Metrics endpoint
@app.get("/metrics")
def metrics():
    with open("saved_models/metrics.json") as f:
        return json.load(f)


# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    start = time.time()

    features = [[
        data.lag_1,
        data.lag_7,
        data.lag_14,
        data.rolling_mean_7,
        data.rolling_mean_14,
        data.day_of_week,
        data.month
    ]]

    prediction = float(model.predict(features)[0])
    latency = time.time() - start

    # Logging
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": data.dict(),
        "prediction": prediction,
        "latency": round(latency, 4)
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log) + "\n")

    return {
        "prediction": prediction,
        "latency": round(latency, 4)
    }