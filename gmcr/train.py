"""Compare Gradient Boosting, LightGBM and CatBoost on each platform and save the best model per platform.

Run from the Codethon folder:  .venv/Scripts/python -m gmcr.train

Writes to models/:
  aws.joblib, azure.joblib, vmware.joblib   best model per platform, refitted on all rows
  comparison.csv                            every algorithm and setting, 5-fold CV scores
  comparison.png                            best setting of each algorithm per platform
  model_card.json                           winner per platform with its scores
"""
import json
import time
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from gmcr.config import MODEL_DIR, PLATFORMS, TARGET, features, load_data, physics_baseline
from gmcr.models import ALGOS, GRIDS, build

KFOLD = KFold(5, shuffle=True, random_state=0)


def scores(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {"mae_min": float(mean_absolute_error(actual, pred)),
            "rmse_min": float(np.sqrt(mean_squared_error(actual, pred))),
            "r2": float(r2_score(actual, pred)),
            "median_error": float(np.median(np.abs(pred - actual) / actual))}


def compare(platform: str) -> tuple[pd.DataFrame, dict]:
    """Score every algorithm and setting with the same 5 folds. Returns all rows and the CV predictions."""
    d = load_data(platform)
    X, y = d[features(platform)], d[TARGET].to_numpy()
    rows, preds = [], {}
    rows.append({"platform": PLATFORMS[platform]["name"], "algorithm": "Physics formula", "params": "-",
                 **scores(y, np.maximum(physics_baseline(platform, d), 0.1)), "fit_seconds": 0.0})
    for algo in ALGOS:
        for params in GRIDS[algo]:
            t = time.time()
            pred = cross_val_predict(build(algo, PLATFORMS[platform]["cat"], params), X, y, cv=KFOLD)
            key = (algo, json.dumps(params))
            preds[key] = pred
            rows.append({"platform": PLATFORMS[platform]["name"], "algorithm": algo, "params": key[1],
                         **scores(y, pred), "fit_seconds": (time.time() - t) / KFOLD.n_splits})
    return pd.DataFrame(rows), preds


Z_FLAG = 3.5   # robust z-score beyond which a past restore is flagged as unusual


def _robust(resid: np.ndarray) -> tuple[float, float]:
    center = float(np.median(resid))
    return center, float(1.4826 * np.median(np.abs(resid - center))) or 1e-9


def _fit_and_save(platform: str, algo: str, params: dict, d: pd.DataFrame, cv_pred: np.ndarray) -> dict:
    cfg = PLATFORMS[platform]
    X, y = d[features(platform)], d[TARGET].to_numpy()
    model = clone(build(algo, cfg["cat"], params)).fit(X, y)
    resid = np.log(y) - np.log(cv_pred)
    # P90 = P50 × the 90th percentile of cross-validated log errors, so the band is measured on unseen rows.
    p90_factor = float(max(1.0, np.exp(np.quantile(resid, 0.9))))
    center, scale = _robust(resid)
    # Unusual past restores: compare each with restores in the same location, so one slow region does not
    # hide individual bad restores inside it.
    loc = cfg["location"][0] if cfg["location"] else None
    peers = {v: _robust(resid[d[loc].to_numpy() == v]) for v in d[loc].unique()} if loc else {}
    cs = np.array([peers.get(v, (center, scale)) for v in d[loc]]) if loc else np.tile([center, scale], (len(d), 1))
    z = (resid - cs[:, 0]) / cs[:, 1]
    scored = d[[cfg["id"], *features(platform), TARGET]].assign(expected=cv_pred, ratio=np.exp(resid), resid=resid, z=z,
                                                                flagged=np.abs(z) > Z_FLAG)
    bundle = {
        "platform": platform, "algorithm": algo, "params": params, "model": model,
        "features": features(platform), "p90_factor": p90_factor,
        "categories": {c: sorted(d[c].unique().tolist()) for c in cfg["cat"]},
        "numeric_range": {c: [float(d[c].min()), float(d[c].max())] for c in cfg["num"]},
        "cv_scores": scores(y, cv_pred), "residual_center": center, "residual_scale": scale, "peers": peers,
        "scored": scored, "rows": len(d), "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_DIR / f"{platform}.joblib")
    return {k: v for k, v in bundle.items() if k not in ("model", "scored", "peers")}


def save_best(platform: str, results: pd.DataFrame, preds: dict) -> dict:
    best = results[results.algorithm != "Physics formula"].sort_values("mae_min").iloc[0]
    return _fit_and_save(platform, best.algorithm, json.loads(best.params), load_data(platform),
                         preds[(best.algorithm, best.params)])


def refit(platform: str) -> dict:
    """Retrain the saved winner on current data (including logged feedback) without rerunning the comparison."""
    from gmcr import predict
    old = predict.load_model(platform)
    d = load_data(platform)
    cv_pred = cross_val_predict(build(old["algorithm"], PLATFORMS[platform]["cat"], old["params"]),
                                d[features(platform)], d[TARGET].to_numpy(), cv=KFOLD)
    card = _fit_and_save(platform, old["algorithm"], old["params"], d, cv_pred)
    predict.load_model.cache_clear()
    return card


def chart(results: pd.DataFrame):
    best = results.loc[results.groupby(["platform", "algorithm"]).mae_min.idxmin()]
    fig, axes = plt.subplots(1, len(PLATFORMS), figsize=(13, 4))
    for ax, cfg in zip(axes, PLATFORMS.values()):
        b = best[(best.platform == cfg["name"]) & (best.algorithm != "Physics formula")].set_index("algorithm").loc[list(ALGOS)]
        physics = best[(best.platform == cfg["name"]) & (best.algorithm == "Physics formula")].mae_min.iloc[0]
        colors = ["#2a9d8f" if v == b.mae_min.min() else "#b0bec5" for v in b.mae_min]
        bars = ax.bar(b.index, b.mae_min, color=colors)
        ax.bar_label(bars, fmt="%.1f")
        ax.margins(y=0.15)
        ax.set_title(f"{cfg['name']}  (physics formula: {physics:.0f} min)")
        ax.set_ylabel("CV mean absolute error (min)")
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("Restore-time model comparison, 5-fold cross-validation (lower is better)")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "comparison.png", dpi=150)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    all_results, card = [], {}
    for platform in PLATFORMS:
        print(f"Comparing models on {PLATFORMS[platform]['name']}…", flush=True)
        results, preds = compare(platform)
        card[platform] = save_best(platform, results, preds)
        all_results.append(results)
    results = pd.concat(all_results, ignore_index=True)
    results.round(4).to_csv(MODEL_DIR / "comparison.csv", index=False)
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    chart(results)

    best = results.loc[results.groupby(["platform", "algorithm"]).mae_min.idxmin()]
    table = best.pivot(index="algorithm", columns="platform", values="mae_min").loc[["Physics formula", *ALGOS]]
    print("\nCV mean absolute error (minutes), best setting per algorithm:")
    print(table[[c["name"] for c in PLATFORMS.values()]].round(1).to_string())
    print("\nSaved best model per platform:")
    for platform, c in card.items():
        print(f"  {PLATFORMS[platform]['name']:7s} {c['algorithm']:17s} MAE {c['cv_scores']['mae_min']:.1f} min, "
              f"R² {c['cv_scores']['r2']:.3f}, P90 = P50 × {c['p90_factor']:.2f}")


if __name__ == "__main__":
    main()
