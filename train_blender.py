from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.blending import blend_forecasts, fit_quantile_mapping, normalize_weights
from src.config import ARTIFACT_DIR, DATA_FILE, VARIABLES, WEIGHTS_FILE, QM_FILE
from src.features import make_features


def oracle_weights(df: pd.DataFrame) -> np.ndarray:
    """Create supervised weight targets from inverse verification error.

    This is valid for synthetic training because truth is available.
    Operationally, use historical forecast-vs-analysis verification.
    """
    errors = np.column_stack([
        np.abs(df["nwp"].to_numpy() - df["truth"].to_numpy()),
        np.abs(df["ai"].to_numpy() - df["truth"].to_numpy()),
        np.abs(df["ensemble"].to_numpy() - df["truth"].to_numpy()),
    ])
    scores = 1.0 / (errors + 0.35)
    return scores / scores.sum(axis=1, keepdims=True)


def train_variable(df: pd.DataFrame, variable: str):
    part = df[df["variable"] == variable].copy()
    cutoff = int(part["day"].max() * 0.70)
    train = part[part["day"] <= cutoff].copy()

    X = make_features(train)
    y = oracle_weights(train)
    models = {}

    for i, name in enumerate(["nwp", "ai", "ensemble"]):
        model = XGBRegressor(
            n_estimators=280,
            max_depth=5,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42 + i,
            n_jobs=4,
        )
        model.fit(X, y[:, i])
        models[name] = model

    raw = np.column_stack([models[n].predict(X) for n in ["nwp", "ai", "ensemble"]])
    weights = normalize_weights(raw)
    blended = blend_forecasts(train, weights)
    mapper = fit_quantile_mapping(blended, train["truth"].to_numpy())

    return models, mapper, cutoff


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        from src.data_generation import generate_synthetic_weather, save_dataset
        save_dataset(generate_synthetic_weather(), DATA_FILE)

    df = pd.read_csv(DATA_FILE)
    all_models, all_qm, cutoffs = {}, {}, {}

    for variable in VARIABLES:
        models, mapper, cutoff = train_variable(df, variable)
        all_models[variable] = models
        all_qm[variable] = mapper
        cutoffs[variable] = cutoff

    joblib.dump(
        {
            "models": all_models,
            "model_names": ["nwp", "ai", "ensemble"],
            "cutoffs": cutoffs,
        },
        WEIGHTS_FILE,
    )
    joblib.dump(all_qm, QM_FILE)

    print(f"Saved weight engine -> {WEIGHTS_FILE}")
    print(f"Saved quantile mappers -> {QM_FILE}")
    print(json.dumps({"train_cutoffs_by_variable": cutoffs}, indent=2))


if __name__ == "__main__":
    main()
