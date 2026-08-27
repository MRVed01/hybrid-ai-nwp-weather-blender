from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def csi(y_true, y_pred, threshold: float, event_type: str = "high") -> float:
    """Critical Success Index = hits / (hits + misses + false alarms)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if event_type == "low":
        obs = y_true <= threshold
        fcst = y_pred <= threshold
    else:
        obs = y_true >= threshold
        fcst = y_pred >= threshold

    hits = np.sum(obs & fcst)
    misses = np.sum(obs & ~fcst)
    false_alarms = np.sum(~obs & fcst)
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom else 0.0
