import os
import json
import joblib
from sklearn.ensemble import RandomForestRegressor

from model.data_loader import load_data, prepare_single_series, create_features
from model.metrics_utils import calculate_metrics


def main():
    df = load_data()
    ts = prepare_single_series(df)
    feat = create_features(ts)

    feature_cols = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "day_of_week",
        "month",
    ]

    X = feat[feature_cols]
    y = feat["sales"].astype(float)

    split_idx = int(len(feat) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = calculate_metrics(y_test, preds)
    metrics["train_rows"] = int(len(X_train))
    metrics["test_rows"] = int(len(X_test))
    metrics["features"] = feature_cols
    metrics["model_name"] = "RandomForestRegressor"

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(model, "saved_models/model.pkl")

    with open("saved_models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model trained and saved to saved_models/model.pkl")
    print("Metrics saved to saved_models/metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()