from __future__ import annotations

import pandas as pd


def detect_extremes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["heatwave"] = (out["variable"].eq("temperature") & (out["truth"] >= 35.0)).astype(int)
    out["cloudburst"] = (out["variable"].eq("rainfall") & (out["truth"] >= 50.0)).astype(int)
    out["strong_wind"] = (out["variable"].eq("wind") & (out["truth"] >= 17.0)).astype(int)
    out["cyclone"] = (out["cyclone_regime"] == 1).astype(int)
    return out


def alert_summary(df: pd.DataFrame) -> pd.DataFrame:
    alerts = detect_extremes(df)
    return pd.DataFrame({
        "event": ["Cyclone", "Heatwave", "Cloudburst", "Strong wind"],
        "count": [
            int(alerts["cyclone"].sum()),
            int(alerts["heatwave"].sum()),
            int(alerts["cloudburst"].sum()),
            int(alerts["strong_wind"].sum()),
        ],
    })
