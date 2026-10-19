"""B. Rank the likely causes of a VM outage from the hour of telemetry before it."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold

from gmcr.explain import evidence
from gmcr.features import window_features
from gmcr.telemetry import CAUSE_LABEL, CAUSES, Telemetry

NON_FEATURES = {"incident_id", "vm_id", "cause"}


def _model():
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                             objective="multi:softprob", tree_method="hist", random_state=0, n_jobs=-1)


def _scores(y: np.ndarray, proba: np.ndarray) -> dict:
    top2 = np.argsort(proba, axis=1)[:, -2:]
    pred = proba.argmax(axis=1)
    return {"accuracy": accuracy_score(y, pred), "top2_accuracy": float(np.mean([y[i] in top2[i] for i in range(len(y))])),
            "macro_f1": f1_score(y, pred, average="macro"),
            "recall_by_cause": dict(zip(CAUSES, recall_score(y, pred, average=None, labels=range(len(CAUSES)))))}


def train(data: pd.DataFrame, train_vms: set, test_vms: set) -> tuple[dict, dict, pd.DataFrame]:
    feats = [c for c in data.columns if c not in NON_FEATURES]
    y = data.cause.map({c: i for i, c in enumerate(CAUSES)}).to_numpy()
    tr = data.vm_id.isin(train_vms).to_numpy()
    cv_proba = np.zeros((tr.sum(), len(CAUSES)))
    for a, b in GroupKFold(5).split(data[tr], groups=data.vm_id[tr]):
        cv_proba[b] = _model().fit(data[tr].iloc[a][feats], y[tr][a]).predict_proba(data[tr].iloc[b][feats])
    model = _model().fit(data[tr][feats], y[tr])
    test_proba = model.predict_proba(data[~tr][feats])
    metrics = {"train_incidents": int(tr.sum()), "test_incidents": int((~tr).sum()),
               "cv_on_train_vms": _scores(y[tr], cv_proba), "held_out_vms": _scores(y[~tr], test_proba),
               "majority_class_accuracy": float(np.bincount(y[~tr]).max() / (~tr).sum())}
    cm = pd.DataFrame(confusion_matrix(y[~tr], test_proba.argmax(1), labels=range(len(CAUSES))),
                      index=[f"true {c}" for c in CAUSES], columns=[f"pred {c}" for c in CAUSES])
    return {"model": model, "features": feats}, metrics, cm


def diagnose(bundle: dict, tel: Telemetry, vm_id: str, minute: int) -> dict:
    """Ranked causes for an outage at `minute`, with the evidence behind the top one."""
    row = pd.DataFrame([window_features(tel.metrics, tel.events, vm_id, minute)])[bundle["features"]]
    proba = bundle["model"].predict_proba(row)[0]
    order = np.argsort(proba)[::-1]
    contrib = bundle["model"].get_booster().predict(xgb.DMatrix(row), pred_contribs=True)[0]  # (classes, features+1)
    top = order[0]
    ev = evidence(bundle["features"], row.iloc[0].to_numpy(float), contrib[top, :-1])
    ranked = [{"cause": CAUSES[i], "label": CAUSE_LABEL[CAUSES[i]], "probability": float(proba[i])} for i in order]
    msg = f"Likely cause: {CAUSE_LABEL[CAUSES[top]]} ({proba[top]:.0%})"
    msg += f"; supporting evidence: {', '.join(ev)}." if ev else "; no strong single signal (low-warning failure)."
    if proba[top] < 0.5:
        msg += f" Also possible: {CAUSE_LABEL[CAUSES[order[1]]]} ({proba[order[1]]:.0%})."
    return {"vm_id": vm_id, "minute": minute, "ranked": ranked, "evidence": ev, "message": msg}
