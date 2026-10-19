"""Tests for restore analysis, business impact, templates, report and feedback.
Needs `python -m gmcr.train` to have been run first."""
import io
import shutil

import numpy as np
import pandas as pd
import pytest

from gmcr import config, ops, predict as P, restore
from gmcr.config import DEMO_CASE, MODEL_DIR, PLATFORMS, features

pytestmark = pytest.mark.skipif(not all((MODEL_DIR / f"{p}.joblib").exists() for p in PLATFORMS),
                                reason="run python -m gmcr.train first")
P_DEMO, W_DEMO = DEMO_CASE["platform"], DEMO_CASE["workload"]


@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_shap_adds_up_to_the_forecast(platform):
    X = P.load_model(platform)["scored"][features(platform)].sample(20, random_state=0)
    sv, base = restore._shap(platform, X)
    assert np.allclose(np.exp(base + sv.sum(axis=1)), P.forecast(platform, X).p50_min, rtol=1e-4)


def test_shap_fallback_for_pipeline_winners(monkeypatch):
    from gmcr.models import build
    d = config.load_data("vmware")
    model = build("Gradient Boosting", PLATFORMS["vmware"]["cat"], {"n_estimators": 50}).fit(d[features("vmware")], d.Restore_Time_Min)
    monkeypatch.setattr(restore, "load_model", lambda p: {"model": model})
    X = d[features("vmware")].head(5)
    sv, base = restore._shap("vmware", X)
    assert list(sv.columns) == features("vmware")  # one-hot columns folded back
    assert np.allclose(np.exp(base + sv.sum(axis=1)), model.predict(X), rtol=1e-4)


def test_demo_case_cause_and_fix():
    f = restore.factors(P_DEMO, W_DEMO)
    red = f[f.flag == "red"]
    assert list(red.index) == ["EBS_Volume_Type"] and red.multiplier.iloc[0] > 3.5
    assert 0.95 < f.loc["Provisioned_IOPS", "multiplier"] < 1.05
    assert f.loc["VM_Size_GB", "flag"] == ""  # workload size is never flagged
    w = restore.what_if(P_DEMO, W_DEMO)
    assert w.input.iloc[0] == "EBS_Volume_Type" and w.change_to.iloc[0] == "io2" and w.minutes_saved.iloc[0] > 700
    assert (w.minutes_saved >= restore.MIN_SAVING * w.p50_now - 1e-9).all()


def test_scalability_keeps_growing_beyond_training_range():
    s = restore.scalability(P_DEMO, W_DEMO, [1000, 1971, 4000])
    assert list(s.extrapolated) == [False, False, True]
    assert s.p50_min.iloc[2] == pytest.approx(s.p50_min.iloc[1] * 4000 / 1996, rel=0.02)  # 1996 GB = largest trained


@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_validation_helpers(platform):
    flagged = restore.flagged_restores(platform)
    assert len(flagged) and (flagged.z.abs() > 3.5).all()
    assert list(flagged.z.abs()) == sorted(flagged.z.abs(), reverse=True)
    imp = restore.global_importance(platform)
    assert set(imp.index) == set(features(platform)) and imp.iloc[0] > 0
    heat = restore.location_heat(platform)
    assert (heat is None) == (not PLATFORMS[platform]["location"])
    assert P.predict(platform, restore.typical_config(platform))["p50_min"] > 0


def test_compare_platforms():
    c = restore.compare_platforms(1000)
    assert list(c.platform) == ["AWS", "Azure", "VMware"] and (c.p90_min >= c.p50_min).all()


def test_impact_and_incidents():
    assert ops.impact(70.6167, 2.3, 60, 25000)["breach_cost"] == pytest.approx(4423.6, abs=0.1)  # reference site
    ok = ops.impact(50, 10, 60, 25000)
    assert not ok["breach"] and ok["breach_cost"] == 0 and ok["friction_cost"] < ok["downtime_cost"]
    assert ops.incident_overhead("Ransomware", 500) == 130
    with pytest.raises(ValueError):
        ops.incident_overhead("Alien invasion", 1)


def test_critical_path_and_chaos():
    tiers = pd.DataFrame({"Tier": ["DB", "MW", "FE", "Cache"], "Depends_On": ["", "DB", "MW", ""]})
    finish, chain = ops.critical_path(tiers, {"DB": 70, "MW": 28, "FE": 14, "Cache": 90})
    assert finish["FE"] == 112 and chain == ["DB", "MW", "FE"]
    with pytest.raises(ValueError, match="cycle"):
        ops.critical_path(pd.DataFrame({"Tier": ["A", "B"], "Depends_On": ["B", "A"]}), {"A": 1, "B": 1})
    assert np.allclose(ops.chaos(100, 0), 100) and np.median(ops.chaos(100, 20)) > 100


def test_templates():
    hcl2 = pytest.importorskip("hcl2")
    yaml = pytest.importorskip("yaml")
    from gmcr.iac import kubernetes, terraform
    st1 = hcl2.load(io.StringIO(terraform("aws", W_DEMO, 1, 2)))["resource"][0]['"aws_ebs_volume"']['"gmcr_restore"']
    gp3 = hcl2.load(io.StringIO(terraform("aws", {**W_DEMO, "EBS_Volume_Type": "gp3"}, 1, 2)))["resource"][0]['"aws_ebs_volume"']['"gmcr_restore"']
    assert "iops" not in st1 and gp3["iops"] == 16000
    assert terraform("vmware", restore.typical_config("vmware"), 1, 2) is None
    docs = list(yaml.safe_load_all(kubernetes("azure", {"VM_Size_GB": 500}, 1, 2)))
    ss = docs[1]
    assert ss["spec"]["selector"]["matchLabels"] == ss["spec"]["template"]["metadata"]["labels"]
    assert ss["metadata"]["name"] == ss["metadata"]["name"].lower()


def test_pdf_report_contents():
    pypdf = pytest.importorskip("pypdf")
    from gmcr.report import pdf
    fc = P.predict(P_DEMO, W_DEMO)
    ov = ops.incident_overhead("Ransomware", W_DEMO["VM_Size_GB"])
    imp = ops.impact(fc["p90_min"] + ov, fc["friction_min"], 240, 25000)
    flags = [{"input": i, **r} for i, r in restore.factors(P_DEMO, W_DEMO).query("flag != ''").iterrows()]
    data = pdf(P_DEMO, W_DEMO, fc, imp, 240, 25000, "Ransomware", ov, flags,
               restore.what_if(P_DEMO, W_DEMO).to_dict("records"), P.load_model(P_DEMO)["cv_scores"])
    text = pypdf.PdfReader(io.BytesIO(data)).pages[0].extract_text()
    assert data[:5] == b"%PDF-" and "SLA BREACH" in text and "st1 → io2" in text and "×4.0" in text


def test_feedback_round_trip(tmp_path, monkeypatch):
    from gmcr import feedback, train
    (tmp_path / "models").mkdir()
    for f in MODEL_DIR.glob("*.joblib"):
        shutil.copy(f, tmp_path / "models")
    monkeypatch.setattr(config, "FEEDBACK_DIR", tmp_path / "fb")
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(P, "MODEL_DIR", tmp_path / "models")
    P.load_model.cache_clear()
    try:
        w = {**W_DEMO, "EBS_Volume_Type": "gp3", "VM_Size_GB": 500}
        expected = P.predict(P_DEMO, w)["p50_min"]
        assert not feedback.save(P_DEMO, w, expected * 1.01)["flagged"]
        assert feedback.save(P_DEMO, w, expected * 3)["flagged"]
        assert feedback.trend(P_DEMO)["drift"] is None  # fewer than 5 entries
        for _ in range(3):
            feedback.save(P_DEMO, w, expected * 3)
        assert feedback.trend(P_DEMO)["drift"] is True
        assert train.refit(P_DEMO)["rows"] == 1005
        with pytest.raises(ValueError):
            feedback.save(P_DEMO, w, 0)
    finally:
        P.load_model.cache_clear()  # later tests must load the real models again
