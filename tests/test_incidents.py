"""Tests for the incident pipeline (A failure risk, B root cause, C recovery).
Tests marked 'trained' need `python -m gmcr.pipeline` to have been run first."""
import json

import numpy as np
import pytest

from gmcr import telemetry
from gmcr.config import MODEL_DIR
from gmcr.explain import describe, evidence
from gmcr.features import HORIZON, failure_dataset, vm_minute_features
from gmcr.telemetry import CODES, METRICS

trained = pytest.mark.skipif(not (MODEL_DIR / "ops_card.json").exists() or not (telemetry.SYNTH_DIR / "metrics.parquet").exists(),
                             reason="run python -m gmcr.pipeline first")


@pytest.fixture(scope="module")
def small():
    return telemetry.generate(n_vms=12, days=1, failures_per_day=2.0, seed=3)


def test_generator_is_deterministic_and_consistent(small):
    again = telemetry.generate(n_vms=12, days=1, failures_per_day=2.0, seed=3)
    assert small.incidents.equals(again.incidents)
    assert len(small.metrics) == 12 * 1440 and len(small.incidents) > 0
    assert set(small.events.code) <= set(CODES)
    down = small.metrics[small.metrics.is_down]
    assert down[METRICS].isna().all().all() and (down.health_ok == 0).all()
    for i in small.incidents.itertuples():  # every failure is marked by a VM_STOPPED event
        assert ((small.events.vm_id == i.vm_id) & (small.events.minute == i.fail_minute) & (small.events.code == "VM_STOPPED")).any()


def test_features_never_use_the_future(small):
    vm = small.fleet.iloc[0]
    fails = sorted(small.incidents[small.incidents.vm_id == vm.vm_id].fail_minute)
    full = vm_minute_features(small.metrics, small.events, vm.vm_id, vm.hist_failures, fails).set_index("minute")
    for x in (300, 700, 1200):
        cut = vm_minute_features(small.metrics[small.metrics.minute <= x], small.events[small.events.minute <= x],
                                 vm.vm_id, vm.hist_failures, fails).set_index("minute")
        cols = [c for c in full.columns if c != "vm_id"]
        assert np.allclose(full.loc[x, cols].astype(float), cut.loc[x, cols].astype(float), equal_nan=True)


def test_labels_mark_the_15_minutes_before_each_failure(small):
    d = failure_dataset(small)
    assert d.label.sum() == HORIZON * len(small.incidents)
    assert d[d.label == 1].minutes_to_failure.between(1, HORIZON).all()
    assert not d.is_down.any()


def test_evidence_skips_normal_values_and_repeats():
    assert describe("evt60_AUTH_FAIL", 2) is None            # background noise level
    assert describe("evt60_AUTH_FAIL", 11) == "11 failed logins in the last hour"
    assert describe("evt60_KERNEL_PANIC", 1) == "1 kernel panic in the last hour"
    assert describe("disk_free_pct_min", 67) is None and describe("disk_free_pct_min", 4) == "free disk fell to 4%"
    assert describe("disk_latency_ms_vs_normal", 1.1) is None
    names = ["disk_latency_ms_peak_vs_normal", "disk_latency_ms_vs_normal", "evt60_DISK_IO_ERR"]
    ev = evidence(names, np.array([8.3, 1.8, 5.0]), np.array([3.0, 2.0, 1.0]))
    assert ev == ["disk latency peaked at 8.3× normal", "5 disk I/O errors in the last hour"]


@pytest.fixture(scope="module")
def live():
    from gmcr.pipeline import load_models
    tel = telemetry.load()
    risk, cause = load_models()
    card = json.loads((MODEL_DIR / "ops_card.json").read_text())
    held = tel.incidents[tel.incidents.vm_id.isin(card["test_vms"])]
    return tel, risk, cause, card, held


@trained
def test_a_failure_risk_quality(live):
    _, _, _, card, _ = live
    a = card["failure_risk"]
    assert a["xgboost"]["pr_auc"] > a["logistic_baseline"]["pr_auc"]
    assert a["at_elevated"]["recall_incidents"] >= 0.75
    assert a["at_elevated"]["false_alert_episodes_per_vm_day"] < 1.0
    assert a["caught_by_cause_at_elevated"]["host"] < 0.2  # host failures give no warning: documented limit


@trained
def test_a_warns_before_a_held_out_network_failure(live):
    from gmcr.failure import assess
    tel, risk, _, _, held = live
    inc = held[held.cause == "network"].iloc[0]
    early, late = assess(risk, tel, inc.vm_id, inc.fail_minute - 60), assess(risk, tel, inc.vm_id, inc.fail_minute - 1)
    assert late["probability"] > early["probability"] and late["band"] != "Low"
    assert "within the next 15 minutes" in late["message"]
    assert assess(risk, tel, inc.vm_id, inc.fail_minute + 1)["down"]


@trained
def test_b_root_cause_quality_and_evidence(live):
    from gmcr.rootcause import diagnose
    tel, _, cause, card, held = live
    assert card["root_cause"]["held_out_vms"]["accuracy"] >= 0.9
    inc = held[held.cause == "storage_io"].iloc[0]
    d = diagnose(cause, tel, inc.vm_id, inc.fail_minute)
    assert d["ranked"][0]["cause"] == "storage_io"
    assert abs(sum(r["probability"] for r in d["ranked"]) - 1) < 1e-6
    assert any("disk" in e for e in d["evidence"]) and d["message"].startswith("Likely cause: storage I/O failure")


@trained
def test_c_eta_uses_restore_model_and_playbook(live):
    from gmcr.predict import predict
    from gmcr.recovery import PLAYBOOK, eta
    tel, *_ = live
    vm = tel.fleet.iloc[0]
    cfg = json.loads(vm.restore_config)
    restore = predict(vm.platform, cfg)
    assert eta("storage_io", vm.platform, cfg)["p50_min"] == pytest.approx(PLAYBOOK["storage_io"]["fix_min"] + restore["p50_min"])
    assert eta("app", vm.platform, cfg)["p50_min"] == PLAYBOOK["app"]["fix_min"]  # no restore needed
    sec = eta("security", vm.platform, cfg)
    assert sec["p50_min"] > restore["p50_min"] + PLAYBOOK["security"]["fix_min"] - 1e-9


@trained
def test_c_plan_and_triage(live):
    from gmcr.pipeline import analyze, triage
    tel, risk, cause, _, held = live
    inc = held[held.cause == "security"].iloc[0]
    r = analyze(tel, risk, cause, inc.vm_id, inc.fail_minute + 1)
    assert r["plan"]["severity"] == "SEV1" and r["plan"]["message"].startswith("Restore ETA:")
    assert "prepare an alternative recovery target" in r["plan"]["message"]
    up = analyze(tel, risk, cause, inc.vm_id, inc.fail_minute - 120)
    assert up["diagnosis"] is None and not up["risk"]["down"]
    t = triage(tel, risk, cause, int(inc.fail_minute) + 1, sorted(held.vm_id.unique()))
    assert inc.vm_id in set(t.vm_id)
    assert list(t.severity) == sorted(t.severity)  # SEV1 before SEV2 before SEV3
    for _, g in t.groupby("severity"):
        assert list(g.priority) == sorted(g.priority, reverse=True)
