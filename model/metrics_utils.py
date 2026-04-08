import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # avoid zero-denominator distortion
    mask = y_true > 0

    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        wape = np.sum(np.abs(y_true[mask] - y_pred[mask])) / np.sum(np.abs(y_true[mask])) * 100
    else:
        mape = np.nan
        wape = np.nan

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4) if not np.isnan(mape) else None,
        "WAPE": round(float(wape), 4) if not np.isnan(wape) else None,
    }