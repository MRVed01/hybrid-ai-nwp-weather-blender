from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.blending import apply_quantile_mapping, blend_forecasts, normalize_weights
from src.config import DATA_FILE, EVAL_FILE, VARIABLES, WEIGHTS_FILE, QM_FILE, CSI_THRESHOLDS
from src.features import make_features
from src.metrics import csi, mae, rmse


def main():
    df = pd.read_csv(DATA_FILE)
    bundle = joblib.load(WEIGHTS_FILE)
    qms = joblib.load(QM_FILE)
    rows = []

    for variable in VARIABLES:
        part = df[df["variable"] == variable].copy()
        cutoff = bundle["cutoffs"][variable]
        test = part[part["day"] > cutoff].copy()

        models = bundle["models"][variable]
        raw = np.column_stack([
            models[n].predict(make_features(test))
            for n in ["nwp", "ai", "ensemble"]
        ])
        weights = normalize_weights(raw)

        test["blended_raw"] = blend_forecasts(test, weights)
        test["blended"] = apply_quantile_mapping(
            test["blended_raw"].to_numpy(), qms[variable]
        )

        threshold = CSI_THRESHOLDS[variable]
        event_type = "low" if variable == "pressure" else "high"
        scores = {
            "NWP": test["nwp"].to_numpy(),
            "AI": test["ai"].to_numpy(),
            "Ensemble": test["ensemble"].to_numpy(),
            "Blended": test["blended"].to_numpy(),
        }

        baseline_rmse = rmse(test["truth"], scores["NWP"])
        baseline_mae = mae(test["truth"], scores["NWP"])

        for model_name, pred in scores.items():
            model_rmse = rmse(test["truth"], pred)
            model_mae = mae(test["truth"], pred)
            rows.append({
                "variable": variable,
                "model": model_name,
                "RMSE": model_rmse,
                "MAE": model_mae,
                "CSI": csi(test["truth"], pred, threshold, event_type),
                "RMSE_improvement_vs_NWP_pct":
                    100 * (baseline_rmse - model_rmse) / baseline_rmse,
                "MAE_improvement_vs_NWP_pct":
                    100 * (baseline_mae - model_mae) / baseline_mae,
            })

    result = pd.DataFrame(rows)
    EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(EVAL_FILE, index=False)

    print("\n=== Test-set verification ===")
    print(result.round(3).to_string(index=False))
    print("\n=== Blended improvement vs NWP ===")
    print(
        result[result["model"] == "Blended"][
            ["variable", "RMSE_improvement_vs_NWP_pct",
             "MAE_improvement_vs_NWP_pct", "CSI"]
        ].round(2).to_string(index=False)
    )


if __name__ == "__main__":
    main()
