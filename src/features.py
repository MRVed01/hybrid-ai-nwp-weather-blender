from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build numerical context features for the meta-learner."""
    out = df.copy()
    season_map = {"winter": 0, "spring": 1, "summer": 2, "monsoon": 3}
    out["season_code"] = out["season"].map(season_map).astype(float)

    out["lat_sin"] = np.sin(np.deg2rad(out["latitude"]))
    out["lat_cos"] = np.cos(np.deg2rad(out["latitude"]))
    out["lon_sin"] = np.sin(np.deg2rad(out["longitude"]))
    out["lon_cos"] = np.cos(np.deg2rad(out["longitude"]))
    out["lead_log"] = np.log1p(out["lead_hours"])

    columns = [
        "latitude", "longitude", "lead_hours", "lead_log", "season_code",
        "pressure_regime", "humidity", "mountain_index", "coastal_index",
        "tropical_index", "monsoon_index", "cyclone_regime",
        "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    ]
    return out[columns].astype(float)
