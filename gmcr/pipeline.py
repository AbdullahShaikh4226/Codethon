"""End-to-end incident pipeline: telemetry → A failure risk → B root cause → C recovery plan and triage.

Run from the Codethon folder:
  .venv/Scripts/python -m gmcr.pipeline              train everything (reuses data and restore models if present)
  .venv/Scripts/python -m gmcr.pipeline --regenerate also regenerate the synthetic telemetry

Writes models/failure_risk.joblib, models/root_cause.joblib, models/ops_card.json,
models/root_cause_confusion.csv and reports/demo_report.md.
"""
import argparse
import json

import joblib
import pandas as pd

from gmcr import failure, rootcause, telemetry, train as restore_train
from gmcr.config import MODEL_DIR, PLATFORMS, ROOT
from gmcr.features import failure_dataset, incident_dataset
from gmcr.recovery import plan
from gmcr.telemetry import CAUSES, Telemetry

REPORT_DIR = ROOT / "reports"


def load_models() -> tuple[dict, dict]:
    paths = [MODEL_DIR / "failure_risk.joblib", MODEL_DIR / "root_cause.joblib"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}. Run: python -m gmcr.pipeline")
    return joblib.load(paths[0]), joblib.load(paths[1])


def analyze(tel: Telemetry, risk_bundle: dict, cause_bundle: dict, vm_id: str, minute: int, sla_min: float = 240) -> dict:
    """A + B + C for one VM at one minute. B runs only when the VM is down."""
    vm = tel.fleet.set_index("vm_id").loc[vm_id].copy()
    vm["vm_id"] = vm_id
    risk = failure.assess(risk_bundle, tel, vm_id, minute)
    diagnosis = None
    if risk["down"]:
        # Diagnose from telemetry up to the minute the VM went down.
        m = tel.metrics[(tel.metrics.vm_id == vm_id) & (tel.metrics.minute <= minute)]
        up = m[~m.is_down]
        down_since = int(up.minute.max()) + 1 if len(up) else minute
        diagnosis = rootcause.diagnose(cause_bundle, tel, vm_id, down_since)
    return {"risk": risk, "diagnosis": diagnosis, "plan": plan(vm, risk, diagnosis, sla_min)}


def triage(tel: Telemetry, risk_bundle: dict, cause_bundle: dict, minute: int, vm_ids=None, sla_min: float = 240) -> pd.DataFrame:
    """Every VM at one minute, highest priority first. Low-risk VMs that are up are left out."""
    rows = []
    for vm_id in vm_ids if vm_ids is not None else tel.fleet.vm_id:
        r = analyze(tel, risk_bundle, cause_bundle, vm_id, minute, sla_min)
        if r["risk"]["down"] or r["risk"]["band"] != "Low":
            p = r["plan"]
            rows.append({"vm_id": vm_id, "severity": p["severity"], "priority": p["priority_score"], "tier": p["tier"],
                         "status": "down" if r["risk"]["down"] else f"{r['risk']['band']} risk {r['risk']['probability']:.0%}",
                         "likely_cause": p["likely_cause"] or "-", "eta_p50_min": round(p["eta_p50_min"]),
                         "next_step": p["steps"][0]})
    cols = ["vm_id", "severity", "priority", "tier", "status", "likely_cause", "eta_p50_min", "next_step"]
    out = pd.DataFrame(rows, columns=cols)
    # Severity first (down beats at-risk), then priority within a severity.
    return out.assign(_s=out.severity.str[-1].astype(int) if len(out) else []).sort_values(["_s", "priority"], ascending=[True, False], ignore_index=True).drop(columns="_s")


def demo_report(tel: Telemetry, risk_bundle: dict, cause_bundle: dict, test_vms: set) -> str:
    """One held-out incident per cause: risk 10 minutes before, diagnosis and plan after it goes down."""
    lines = ["# GMCR incident pipeline: demo on held-out VMs", "",
             "Synthetic telemetry. Each case: failure risk as the failure approaches (A), then root cause (B) and "
             "recovery plan (C) one minute after the VM went down.", ""]
    held = tel.incidents[tel.incidents.vm_id.isin(test_vms)]
    for cause in CAUSES:
        rows = held[held.cause == cause]
        if rows.empty:
            continue
        inc = rows.iloc[0]
        f = int(inc.fail_minute)
        path = {k: analyze(tel, risk_bundle, cause_bundle, inc.vm_id, f - k)["risk"] for k in (15, 10, 5, 1)}
        first = next((path[k] for k in (15, 10, 5, 1) if path[k]["band"] != "Low"), path[1])
        after = analyze(tel, risk_bundle, cause_bundle, inc.vm_id, f + 1)
        trend = " → ".join(f"{path[k]['probability']:.0%}" for k in (15, 10, 5, 1))
        lines += [f"## {inc.incident_id}: {inc.vm_id} ({inc.platform}), true cause: {cause}", "",
                  f"- **A, risk at 15/10/5/1 min before:** {trend}",
                  f"- **A, first warning:** {first['message']}",
                  f"- **B, after it went down:** {after['diagnosis']['message']}",
                  f"- **C:** {after['plan']['severity']}. {after['plan']['message']}", ""]
    minute = int(held.fail_minute.iloc[len(held) // 2]) + 1
    t = triage(tel, risk_bundle, cause_bundle, minute, sorted(test_vms))
    lines += [f"## Fleet triage at minute {minute} (held-out VMs)", "", t.to_markdown(index=False), ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="regenerate the synthetic telemetry")
    args = ap.parse_args()

    print("1/5 Telemetry…", flush=True)
    tel = telemetry.load(regenerate=args.regenerate)
    if not all((MODEL_DIR / f"{p}.joblib").exists() for p in PLATFORMS):
        print("2/5 Restore-time models (C)…", flush=True)
        restore_train.main()
    else:
        print("2/5 Restore-time models found; retrain with python -m gmcr.train", flush=True)
    train_vms, test_vms = failure.split_vms(tel.fleet)

    print("3/5 Failure-risk model (A)…", flush=True)
    risk_bundle, risk_metrics = failure.train(failure_dataset(tel), train_vms, test_vms, tel.incidents)
    print("4/5 Root-cause model (B)…", flush=True)
    cause_bundle, cause_metrics, confusion = rootcause.train(incident_dataset(tel), train_vms, test_vms)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(risk_bundle, MODEL_DIR / "failure_risk.joblib")
    joblib.dump(cause_bundle, MODEL_DIR / "root_cause.joblib")
    confusion.to_csv(MODEL_DIR / "root_cause_confusion.csv")
    card = {"failure_risk": risk_metrics, "root_cause": cause_metrics, "test_vms": sorted(test_vms)}
    (MODEL_DIR / "ops_card.json").write_text(json.dumps(card, indent=2, default=float))

    print("5/5 Demo report…", flush=True)
    REPORT_DIR.mkdir(exist_ok=True)
    report = demo_report(tel, risk_bundle, cause_bundle, test_vms)
    (REPORT_DIR / "demo_report.md").write_text(report, encoding="utf-8")

    a, b = risk_metrics, cause_metrics["held_out_vms"]
    print(f"\nA failure risk (held-out VMs): PR-AUC {a['xgboost']['pr_auc']:.2f} (logistic baseline {a['logistic_baseline']['pr_auc']:.2f}); "
          f"Elevated threshold catches {a['at_elevated']['failures_caught']} failures, median {a['at_elevated']['median_lead_min']:.0f} min ahead, "
          f"{a['at_elevated']['false_alert_episodes_per_vm_day']:.2f} false alerts per VM-day")
    print("  warned in time, by cause: " + ", ".join(f"{c} {v:.0%}" for c, v in a["caught_by_cause_at_elevated"].items()))
    print(f"B root cause (held-out VMs): accuracy {b['accuracy']:.1%}, top-2 {b['top2_accuracy']:.1%}, macro F1 {b['macro_f1']:.2f}")
    print(f"Report: {REPORT_DIR / 'demo_report.md'}")


if __name__ == "__main__":
    main()
