"""Forecast restore time with the saved best model for a platform."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from gmcr.config import MODEL_DIR, PLATFORMS, physics_baseline


@lru_cache(maxsize=None)
def load_model(platform: str) -> dict:
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform {platform!r}; expected one of {list(PLATFORMS)}")
    path = MODEL_DIR / f"{platform}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m gmcr.train")
    return joblib.load(path)


def validate(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Reject inputs the model cannot handle: missing fields, unseen categories, non-positive numbers."""
    missing = [c for c in bundle["features"] if c not in rows.columns]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    for col, allowed in bundle["categories"].items():
        bad = sorted(set(rows[col]) - set(allowed))
        if bad:
            raise ValueError(f"{col}={bad[0]!r} was never seen in training; use one of {allowed}")
    for col in bundle["numeric_range"]:
        if not (pd.to_numeric(rows[col], errors="coerce") > 0).all():
            raise ValueError(f"{col} must be greater than 0")
    return rows[bundle["features"]]


def forecast(platform: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Batch forecast. Columns: p50_min, p90_min, physics_min, friction_min, beats_physics, extrapolated."""
    bundle = load_model(platform)
    X = validate(bundle, rows)
    lo, hi = bundle["numeric_range"]["VM_Size_GB"]
    # Trees cannot extrapolate: beyond the trained sizes the forecast would go flat. Predict at the nearest
    # trained size and scale by size, since minutes per GB are steady within a configuration in this data.
    size = X.VM_Size_GB.to_numpy(float)
    inside = np.clip(size, lo, hi)
    p50 = bundle["model"].predict(X.assign(VM_Size_GB=inside)) * size / inside
    physics = physics_baseline(platform, X)
    return pd.DataFrame({
        "p50_min": p50, "p90_min": p50 * bundle["p90_factor"], "physics_min": physics,
        "friction_min": np.maximum(p50 - physics, 0.0),  # physics is a floor: a restore faster than it has no friction
        "beats_physics": p50 < physics,
        "extrapolated": ~X.VM_Size_GB.between(lo, hi).to_numpy(),
    }, index=rows.index)


def predict(platform: str, workload: dict) -> dict:
    """One workload, e.g. {"VM_Size_GB": 500, "EBS_Volume_Type": "gp3", ...}. Minutes: p50, p90, physics, friction."""
    r = forecast(platform, pd.DataFrame([workload])).iloc[0]
    return {"algorithm": load_model(platform)["algorithm"], "p50_min": float(r.p50_min), "p90_min": float(r.p90_min),
            "physics_min": float(r.physics_min), "friction_min": float(r.friction_min),
            "beats_physics": bool(r.beats_physics), "extrapolated": bool(r.extrapolated)}
