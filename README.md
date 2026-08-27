# Hybrid AI–NWP Weather Blending Framework

A runnable hackathon reference implementation for dynamically blending NWP, AI-weather-model,
and ensemble forecasts using contextual machine-learning weights.

## Architecture

```text
ERA5 / Synthetic Truth ─────┐
NWP forecasts ──────────────┤
AI forecasts ───────────────┼─> Data ingestion / validation
Ensemble forecasts ─────────┘
                                  │
                                  ▼
                         Context extraction
             lat/lon • lead time • season • pressure
             humidity • terrain • coast • weather regime
                                  │
                                  ▼
                         Meta-learner weight engine
                     ┌───────────┼───────────┐
                     ▼           ▼           ▼
                  w_nwp       w_ai      w_ensemble
                     └───────────┼───────────┘
                                 ▼
                    Weighted forecast blending
                                 │
                                 ▼
                     Tail-aware Quantile Mapping
                                 │
                                 ▼
                     Metrics + Extreme Detection
                                 │
                                 ▼
                       Streamlit dashboard
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py
python train_blender.py
python evaluate.py
streamlit run app.py
```

`app.py` automatically generates data and trains artifacts if they are missing.

## Project structure

```text
hybrid_ai_nwp_weather_blender/
├── app.py
├── generate_data.py
├── train_blender.py
├── evaluate.py
├── requirements.txt
├── README.md
├── data/
├── artifacts/
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_generation.py
    ├── features.py
    ├── blending.py
    ├── metrics.py
    └── extremes.py
```

## Production transition

The interfaces are intentionally separated so the synthetic generator can later be replaced by
ERA5/reanalysis and operational GFS/ECMWF/AI forecast ingestion without changing the dashboard
or verification layers. In a real deployment, meta-learning targets must come from archived
forecast-vs-analysis verification data; future truth is used here only to create supervised
synthetic training labels.
