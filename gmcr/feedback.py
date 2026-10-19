"""Log real restores, check each against the model straight away, and track drift."""
from datetime import datetime

import numpy as np
import pandas as pd

from gmcr.config import PLATFORMS, TARGET, feedback_path, features
from gmcr.predict import load_model, predict, validate
from gmcr.train import Z_FLAG

MIN_FOR_DRIFT = 5   # one unusual restore is not drift


def load(platform: str) -> pd.DataFrame:
    path = feedback_path(platform)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def score(platform: str, workload: dict, actual_min: float) -> dict:
    """How unusual this restore is for its settings: expected minutes, ratio, robust z, flagged."""
    b = load_model(platform)
    expected = predict(platform, workload)["p50_min"]
    resid = float(np.log(actual_min) - np.log(expected))
    loc = PLATFORMS[platform]["location"]
    center, scale = b["peers"].get(workload[loc[0]], (b["residual_center"], b["residual_scale"])) if loc else \
        (b["residual_center"], b["residual_scale"])
    z = (resid - center) / scale
    return {"expected_min": expected, "actual_min": actual_min, "ratio": float(np.exp(resid)), "z": float(z),
            "flagged": bool(abs(z) > Z_FLAG)}


def save(platform: str, workload: dict, actual_min: float) -> dict:
    """Append one real restore. It joins training the next time the model is refitted (train.refit)."""
    if not actual_min > 0:
        raise ValueError("Actual restore time must be greater than 0 minutes.")
    validate(load_model(platform), pd.DataFrame([workload]))
    result = score(platform, workload, actual_min)
    now = datetime.now().isoformat(timespec="seconds")
    row = {"Timestamp": now, PLATFORMS[platform]["id"]: f"feedback-{now}",
           **{c: workload[c] for c in features(platform)}, TARGET: actual_min}
    path = feedback_path(platform)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return result


def trend(platform: str) -> dict:
    """Logged restores vs the model over time, and whether recent error has drifted past 1.25 × CV MAE."""
    fb = load(platform)
    if fb.empty:
        return {"entries": pd.DataFrame(), "drift": None, "message": "No restores logged yet."}
    expected = [predict(platform, r[features(platform)].to_dict())["p50_min"] for _, r in fb.iterrows()]
    entries = fb.assign(Timestamp=pd.to_datetime(fb.Timestamp), expected_min=expected).sort_values("Timestamp")
    if len(entries) < MIN_FOR_DRIFT:
        return {"entries": entries, "drift": None,
                "message": f"{len(entries)} restore(s) logged. Drift is checked from {MIN_FOR_DRIFT} onwards."}
    recent = float(np.abs(entries[TARGET] - entries.expected_min).tail(20).mean())
    limit = 1.25 * load_model(platform)["cv_scores"]["mae_min"]
    drift = recent > limit
    return {"entries": entries, "drift": drift, "recent_mae_min": recent, "limit_min": limit,
            "message": f"Recent error {recent:.1f} min vs {limit:.1f} min drift limit."
                       + (" Real restores are drifting from the model: refit or check the environment." if drift else "")}
