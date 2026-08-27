from __future__ import annotations

import numpy as np
import pandas as pd


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def normalize_weights(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return softmax(raw)


def blend_forecasts(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    return (
        weights[:, 0] * df["nwp"].to_numpy()
        + weights[:, 1] * df["ai"].to_numpy()
        + weights[:, 2] * df["ensemble"].to_numpy()
    )


def fit_quantile_mapping(source: np.ndarray, target: np.ndarray, n_quantiles: int = 401) -> dict:
    """Fit empirical quantile mapping source -> target."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source = source[np.isfinite(source)]
    target = target[np.isfinite(target)]

    q = np.linspace(0.0, 1.0, n_quantiles)
    source_q = np.quantile(source, q)
    target_q = np.quantile(target, q)
    unique_source, idx = np.unique(source_q, return_index=True)

    return {
        "q": q[idx],
        "source_q": unique_source,
        "target_q": target_q[idx],
    }


def apply_quantile_mapping(values: np.ndarray, mapper: dict) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.interp(
        values,
        mapper["source_q"],
        mapper["target_q"],
        left=mapper["target_q"][0],
        right=mapper["target_q"][-1],
    )
