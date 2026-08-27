from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

DATA_FILE = DATA_DIR / "synthetic_weather.csv"
WEIGHTS_FILE = ARTIFACT_DIR / "weight_engine.joblib"
QM_FILE = ARTIFACT_DIR / "quantile_mapper.joblib"
EVAL_FILE = ARTIFACT_DIR / "evaluation.csv"

VARIABLES = ["temperature", "rainfall", "wind", "pressure"]
LEAD_TIMES = [6, 24, 72, 120]
SEASONS = ["winter", "spring", "summer", "monsoon"]

CSI_THRESHOLDS = {
    "temperature": 35.0,
    "rainfall": 25.0,
    "wind": 17.0,
    "pressure": 995.0,
}

SEED = 42
