"""A. Probability that a VM fails within the next 15 minutes."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gmcr.explain import evidence
from gmcr.features import HORIZON, vm_minute_features
from gmcr.telemetry import Telemetry

BANDS = [(0.5, "High"), (0.2, "Elevated"), (0.0, "Low")]
NON_FEATURES = {"vm_id", "minute", "is_down", "label", "minutes_to_failure"}


def split_vms(fleet: pd.DataFrame, test_share: float = 0.3, seed: int = 0) -> tuple[set, set]:
    """Hold out whole VMs, so evaluation never sees a VM the model trained on."""
    ids = np.array(sorted(fleet.vm_id))
    test = set(np.random.default_rng(seed).choice(ids, int(len(ids) * test_share), replace=False))
    return set(ids) - test, test


def band(p: float) -> str:
    return next(name for cut, name in BANDS if p >= cut)


def _alert_runs(minutes: np.ndarray) -> int:
    """Number of separate alert episodes (consecutive alert minutes count once)."""
    return int(len(minutes) and 1 + np.sum(np.diff(np.sort(minutes)) > 1))


def evaluate(df: pd.DataFrame, p: np.ndarray, threshold: float) -> dict:
    df = df.assign(alert=p >= threshold)
    caught, leads, n_fail = 0, [], 0
    for _, g in df.groupby("vm_id"):
        for f in sorted(set((g.minute + g.minutes_to_failure)[g.label == 1].astype(int))):
            n_fail += 1
            w = g[(g.minute >= f - HORIZON) & (g.minute < f) & g.alert]
            if len(w):
                caught += 1
                leads.append(f - w.minute.min())
    # False alerts: alert episodes more than an hour away from any failure.
    far = df[(df.minutes_to_failure > 60) & df.alert]
    false_runs = sum(_alert_runs(g.minute.to_numpy()) for _, g in far.groupby("vm_id"))
    vm_days = df.groupby("vm_id").minute.size().sum() / 1440
    return {"threshold": threshold, "failures_caught": f"{caught}/{n_fail}", "recall_incidents": caught / max(n_fail, 1),
            "median_lead_min": float(np.median(leads)) if leads else None,
            "false_alert_episodes_per_vm_day": false_runs / vm_days}


def caught_by_cause(data: pd.DataFrame, p: np.ndarray, incidents: pd.DataFrame, threshold: float = 0.2) -> dict:
    """Share of failures warned about (risk ≥ threshold) in the HORIZON minutes before, per cause."""
    d = data.assign(p=p)
    out = {}
    for cause, g in incidents.groupby("cause"):
        hits = [bool((d[(d.vm_id == i.vm_id) & d.minute.between(i.fail_minute - HORIZON, i.fail_minute - 1)].p >= threshold).any())
                for i in g.itertuples()]
        out[cause] = float(np.mean(hits))
    return out


def train(data: pd.DataFrame, train_vms: set, test_vms: set, incidents: pd.DataFrame | None = None) -> tuple[dict, dict]:
    feats = [c for c in data.columns if c not in NON_FEATURES]
    tr, te = data[data.vm_id.isin(train_vms)], data[data.vm_id.isin(test_vms)]
    model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              tree_method="hist", eval_metric="logloss", random_state=0, n_jobs=-1)
    model.fit(tr[feats], tr.label)
    p = model.predict_proba(te[feats])[:, 1]
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(tr[feats].fillna(0), tr.label)
    pb = baseline.predict_proba(te[feats].fillna(0))[:, 1]
    metrics = {
        "test_vms": len(test_vms), "test_minutes": len(te), "positive_rate": float(te.label.mean()),
        "xgboost": {"roc_auc": roc_auc_score(te.label, p), "pr_auc": average_precision_score(te.label, p)},
        "logistic_baseline": {"roc_auc": roc_auc_score(te.label, pb), "pr_auc": average_precision_score(te.label, pb)},
        "at_elevated": evaluate(te, p, 0.2), "at_high": evaluate(te, p, 0.5),
    }
    if incidents is not None:
        metrics["caught_by_cause_at_elevated"] = caught_by_cause(te, p, incidents[incidents.vm_id.isin(test_vms)])
    return {"model": model, "features": feats}, metrics


def assess(bundle: dict, tel: Telemetry, vm_id: str, minute: int) -> dict:
    """Risk for one VM at one minute, using only telemetry up to that minute."""
    vm = tel.fleet.set_index("vm_id").loc[vm_id]
    m = tel.metrics[(tel.metrics.vm_id == vm_id) & (tel.metrics.minute <= minute)]
    if m.empty:
        raise ValueError(f"No telemetry for {vm_id} at minute {minute}")
    if bool(m.is_down.iloc[-1]):
        return {"vm_id": vm_id, "minute": minute, "down": True, "probability": 1.0, "band": "Down",
                "signals": [], "message": f"{vm_id} is down."}
    past = tel.incidents[(tel.incidents.vm_id == vm_id) & (tel.incidents.fail_minute <= minute)].fail_minute
    f = vm_minute_features(m, tel.events[tel.events.minute <= minute], vm_id, vm.hist_failures, list(past)).iloc[[-1]]
    X = f[bundle["features"]]
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    contrib = bundle["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0, :-1]
    signals = evidence(bundle["features"], X.iloc[0].to_numpy(float), contrib)
    b = band(prob)
    msg = f"{b} risk of failure within the next {HORIZON} minutes ({prob:.0%})."
    if b != "Low" and signals:
        msg += " Signals: " + "; ".join(signals) + "."
    return {"vm_id": vm_id, "minute": minute, "down": False, "probability": prob, "band": b,
            "signals": signals, "message": msg}
