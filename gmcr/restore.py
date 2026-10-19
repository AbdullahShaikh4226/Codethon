"""Restore analysis for the demo: why a restore is slow, what would fix it, and which past restores were unusual."""
import numpy as np
import pandas as pd
import shap
from catboost import Pool

from gmcr.config import PLATFORMS, TARGET, features
from gmcr.models import CatBoostModel
from gmcr.predict import forecast, load_model, validate

RED, AMBER = 1.5, 1.2      # cause multipliers that turn a controllable input red / amber
MIN_SAVING = 0.05          # hide what-ifs that save under 5%: within model noise
MAX_WHATIF_VALUES = 12     # only try inputs with a short list of real values
HOT_EXCESS = 1.3           # a region is hot when its restores run 30%+ slower than expected


def _py(v):
    """numpy scalar → plain Python value (strings pass through)."""
    return v.item() if hasattr(v, "item") else v


def _source(name: str, cats) -> str:
    """Map a one-hot column such as EBS_Volume_Type_st1 back to EBS_Volume_Type."""
    return next((c for c in cats if name.startswith(c + "_")), name)


def _shap(platform: str, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """SHAP values in log-minutes per input column, plus the base value (log of a typical restore)."""
    reg = load_model(platform)["model"].regressor_
    if isinstance(reg, CatBoostModel):
        sv = reg.model_.get_feature_importance(Pool(X, cat_features=list(reg.cat)), type="ShapValues")
        return pd.DataFrame(sv[:, :-1], columns=X.columns, index=X.index), sv[:, -1]
    # Other winners are pipelines: explain the final tree model on the transformed input, then sum one-hot
    # columns back into their source input.
    Xt = reg[:-1].transform(X)
    names = list(Xt.columns) if hasattr(Xt, "columns") else list(reg[:-1].get_feature_names_out())
    explainer = shap.TreeExplainer(reg[-1])
    sv = pd.DataFrame(np.asarray(explainer.shap_values(Xt)), columns=names, index=X.index)
    folded = sv.T.groupby([_source(n, PLATFORMS[platform]["cat"]) for n in names]).sum().T
    return folded[list(X.columns)], np.full(len(X), float(np.ravel(explainer.expected_value)[0]))


def factors(platform: str, workload: dict) -> pd.DataFrame:
    """How much each input multiplies this restore's time versus a typical one. Index = input name.

    Columns: value, multiplier, controllable, flag ("red", "amber" or "")."""
    X = validate(load_model(platform), pd.DataFrame([workload]))
    sv, _ = _shap(platform, X)
    out = pd.DataFrame({"value": X.iloc[0], "multiplier": np.exp(sv.iloc[0])})
    out["controllable"] = out.index.isin(PLATFORMS[platform]["controllable"])
    out["flag"] = np.select([out.controllable & (out.multiplier >= RED), out.controllable & (out.multiplier >= AMBER)],
                            ["red", "amber"], "")
    return out.sort_values("multiplier", ascending=False)


def typical_minutes(platform: str) -> float:
    """The restore time SHAP multipliers are measured against."""
    X = validate(load_model(platform), pd.DataFrame([typical_config(platform)]))
    return float(np.exp(_shap(platform, X)[1][0]))


def what_if(platform: str, workload: dict) -> pd.DataFrame:
    """For each controllable input, the best value seen in training, if it saves at least 5%.

    Columns: input, current, change_to, p50_now, p50_after, minutes_saved. Best saving first."""
    bundle = load_model(platform)
    row = validate(bundle, pd.DataFrame([workload]))
    now = float(forecast(platform, row).p50_min.iloc[0])
    data = bundle["scored"]
    out = []
    for col in PLATFORMS[platform]["controllable"]:
        values = np.sort(data[col].unique())
        if len(values) > MAX_WHATIF_VALUES:
            continue
        preds = forecast(platform, pd.concat([row.assign(**{col: v}) for v in values], ignore_index=True)).p50_min.to_numpy()
        best = int(np.argmin(preds))
        if values[best] != row[col].iloc[0] and preds[best] < now * (1 - MIN_SAVING):
            out.append({"input": col, "current": _py(row[col].iloc[0]), "change_to": _py(values[best]),
                        "p50_now": now, "p50_after": float(preds[best]), "minutes_saved": now - float(preds[best])})
    cols = ["input", "current", "change_to", "p50_now", "p50_after", "minutes_saved"]
    return pd.DataFrame(out, columns=cols).sort_values("minutes_saved", ascending=False, ignore_index=True)


def global_importance(platform: str) -> pd.Series:
    """Mean |SHAP| per input over the training data: what drives restore time on this platform."""
    data = load_model(platform)["scored"]
    sv, _ = _shap(platform, data[features(platform)])
    return sv.abs().mean().sort_values(ascending=False)


def scalability(platform: str, workload: dict, sizes) -> pd.DataFrame:
    """Forecast the same workload at many VM sizes. Adds a VM_Size_GB column."""
    rows = pd.DataFrame([workload] * len(sizes)).assign(VM_Size_GB=np.asarray(sizes, float))
    return forecast(platform, rows).assign(VM_Size_GB=rows.VM_Size_GB.to_numpy())


def flagged_restores(platform: str) -> pd.DataFrame:
    """Past restores that took far longer or shorter than the model expects for their settings, worst first."""
    s = load_model(platform)["scored"]
    return s[s.flagged].sort_values("z", key=np.abs, ascending=False, ignore_index=True)


def location_heat(platform: str) -> pd.DataFrame | None:
    """Per region: restores, flagged count, and how much slower than expected they ran. None if no location data.

    The restore model knows the region, so a region that is slow every time is learned as normal for it;
    this finds regions whose restores are erratic or slower than the model expects."""
    cfg = PLATFORMS[platform]
    if not cfg["location"]:
        return None
    b = load_model(platform)
    s, col = b["scored"], cfg["location"][0]
    overall = s.flagged.mean()
    g = s.groupby(col).agg(restores=("z", "size"), flagged=("flagged", "sum"), flag_rate=("flagged", "mean"),
                           median_resid=("resid", "median"))
    g["slower_than_expected"] = np.exp(g.pop("median_resid") - b["residual_center"])
    g["hot"] = (g.restores >= 10) & ((g.slower_than_expected >= HOT_EXCESS) | ((g.flag_rate >= 2 * overall) & (g.flagged >= 5)))
    return g.reset_index().rename(columns={col: "location"})


def typical_config(platform: str) -> dict:
    """The most common value of every input in the training data."""
    s = load_model(platform)["scored"]
    return {c: _py(s[c].mode().iloc[0]) for c in features(platform)}


def compare_platforms(size_gb: float) -> pd.DataFrame:
    """The same VM size on each platform's most common configuration."""
    rows = []
    for p, cfg in PLATFORMS.items():
        w = {**typical_config(p), "VM_Size_GB": size_gb}
        f = forecast(p, pd.DataFrame([w])).iloc[0]
        rows.append({"platform": cfg["name"], "setup": ", ".join(f"{c} {w[c]}" for c in features(p) if c != "VM_Size_GB"),
                     "physics_min": f.physics_min, "p50_min": f.p50_min, "p90_min": f.p90_min})
    return pd.DataFrame(rows)
