from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEAD_TIMES, SEASONS, SEED, VARIABLES


def _season_from_day(day: int) -> str:
    return SEASONS[(day // 12) % len(SEASONS)]


def generate_synthetic_weather(
    days: int = 48,
    lat_points: int = 15,
    lon_points: int = 20,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate a spatial/temporal synthetic weather benchmark.

    Error regimes deliberately differ:
    - NWP: coastal warm bias, modest long-lead degradation.
    - AI: mountain precipitation underestimation.
    - Ensemble: stable but conservative peak smoothing.
    """
    rng = np.random.default_rng(seed)
    lats = np.linspace(8.0, 35.0, lat_points)
    lons = np.linspace(68.0, 97.0, lon_points)
    rows = []

    for day in range(days):
        season = _season_from_day(day)
        phase = 2 * np.pi * day / 16.0

        for lat in lats:
            for lon in lons:
                mountain = np.exp(-((lat - 30.0) / 5.0) ** 2) * np.exp(-((lon - 78.0) / 8.0) ** 2)
                coastal = np.exp(-((lon - 70.0) / 3.5) ** 2)
                tropical = np.exp(-((lat - 15.0) / 9.0) ** 2)
                monsoon_corridor = np.exp(-((lon - 82.0) / 10.0) ** 2)

                base_temp = (
                    31.0 - 0.55 * (lat - 15.0) + 2.5 * np.sin(phase)
                    + {"winter": -8, "spring": -2, "summer": 4, "monsoon": 0}[season]
                )
                base_temp += 3.0 * tropical - 2.5 * mountain + 1.2 * coastal

                low_pressure = 1012.0 - 10.0 * tropical - 9.0 * np.sin(phase * 0.7)
                humidity = np.clip(
                    55 + 25 * tropical + 18 * monsoon_corridor + rng.normal(0, 5), 10, 100
                )

                rain = (
                    2.0
                    + 12.0 * monsoon_corridor * tropical
                    + 42.0 * mountain * (0.5 + 0.5 * np.sin(phase))
                    + 28.0 * np.maximum(0, np.sin(phase + lon / 9.0))
                )
                if season != "monsoon":
                    rain *= 0.35
                rain = max(0.0, rain + rng.gamma(1.4, 1.2))

                wind = (
                    5.0 + 8.0 * np.clip((1012.0 - low_pressure) / 12.0, 0, 2)
                    + 5.0 * coastal + rng.normal(0, 1.2)
                )
                if season == "monsoon":
                    wind += 2.0 * monsoon_corridor

                truth = {
                    "temperature": base_temp + rng.normal(0, 0.7),
                    "rainfall": rain,
                    "wind": max(0.2, wind),
                    "pressure": low_pressure + rng.normal(0, 1.4),
                }

                cyclone = int(
                    tropical > 0.55 and low_pressure < 1006 and wind > 12
                    and np.sin(phase + lon / 9.0) > 0.25
                )

                for lead in LEAD_TIMES:
                    lead_factor = np.sqrt(lead / 6.0)

                    nwp = {
                        "temperature": truth["temperature"] + 1.8 * coastal + 0.16 * lead_factor
                            + rng.normal(0, 0.8 + 0.15 * lead_factor),
                        "rainfall": truth["rainfall"] * (1 - 0.035 * lead_factor) + 2.5
                            + rng.normal(0, 2.8 + lead_factor),
                        "wind": truth["wind"] * (1 - 0.025 * lead_factor)
                            + rng.normal(0, 1.3 + 0.2 * lead_factor),
                        "pressure": truth["pressure"] + 1.8 * lead_factor
                            + rng.normal(0, 2.0),
                    }

                    ai = {
                        "temperature": truth["temperature"] + 0.08 * lead_factor
                            + rng.normal(0, 0.65 + 0.10 * lead_factor),
                        "rainfall": truth["rainfall"] * (
                            1 - 0.35 * mountain * min(1.4, lead_factor / 2.0)
                        ) + rng.normal(0, 2.0 + 0.9 * lead_factor),
                        "wind": truth["wind"] * (1 - 0.02 * lead_factor)
                            + rng.normal(0, 1.0 + 0.15 * lead_factor),
                        "pressure": truth["pressure"]
                            + rng.normal(0, 1.5 + 0.35 * lead_factor),
                    }

                    ensemble = {
                        "temperature": 0.75 * truth["temperature"] + 0.25 * base_temp
                            + rng.normal(0, 0.55 + 0.12 * lead_factor),
                        "rainfall": 0.72 * truth["rainfall"]
                            + rng.normal(0, 3.0 + 0.7 * lead_factor),
                        "wind": 0.78 * truth["wind"]
                            + rng.normal(0, 1.1 + 0.15 * lead_factor),
                        "pressure": 0.80 * truth["pressure"] + 0.20 * 1012
                            + rng.normal(0, 1.2 + 0.2 * lead_factor),
                    }

                    if cyclone:
                        nwp["rainfall"] *= 0.90
                        ai["rainfall"] *= 0.75
                        ensemble["rainfall"] *= 0.65
                        nwp["wind"] *= 0.92
                        ai["wind"] *= 0.90

                    context = {
                        "day": day,
                        "latitude": lat,
                        "longitude": lon,
                        "lead_hours": lead,
                        "season": season,
                        "pressure_regime": low_pressure,
                        "humidity": humidity,
                        "mountain_index": mountain,
                        "coastal_index": coastal,
                        "tropical_index": tropical,
                        "monsoon_index": monsoon_corridor,
                        "cyclone_regime": cyclone,
                    }

                    for variable in VARIABLES:
                        rows.append({
                            **context,
                            "variable": variable,
                            "truth": truth[variable],
                            "nwp": nwp[variable],
                            "ai": ai[variable],
                            "ensemble": ensemble[variable],
                        })

    df = pd.DataFrame(rows)
    for col in ["nwp", "ai", "ensemble"]:
        if col != "pressure":
            df[col] = df[col].clip(lower=0.0)
    return df


def save_dataset(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
