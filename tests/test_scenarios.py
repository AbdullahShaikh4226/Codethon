"""Tests for the critical path, chaos simulation and multi-cloud scenarios. Needs `python -m gmcr.train` first."""
import numpy as np
import pandas as pd
import pytest

from gmcr import ops, scenarios as S
from gmcr.config import DEMO_CASE, MODEL_DIR, PLATFORMS
from gmcr.predict import predict

pytestmark = pytest.mark.skipif(not all((MODEL_DIR / f"{p}.joblib").exists() for p in PLATFORMS),
                                reason="run python -m gmcr.train first")
P, W = DEMO_CASE["platform"], DEMO_CASE["workload"]


def test_critical_path_default_stack_runs_in_sequence():
    r = S.critical_path(P, W)
    t = r["table"].set_index("tier")
    assert t.loc["Database", "size_gb"] == W["VM_Size_GB"]  # empty size = the workload's size
    assert t.loc["Database", "restore_p90_min"] == pytest.approx(predict(P, W)["p90_min"])
    assert t.loc["App server", "start_p90_min"] == pytest.approx(t.loc["Database", "finish_p90_min"])
    assert r["total_p90_min"] == pytest.approx(t.restore_p90_min.sum())
    assert r["chain"] == ["Database", "App server", "Web frontend"] and r["breach"]
    assert r["message"].startswith("Application back in")


def test_critical_path_parallel_tiers_and_incident():
    tiers = pd.DataFrame({"Tier": ["DB", "Cache", "API"], "Size_GB": [500, 50, 100], "Depends_On": ["", "", "DB"]})
    r = S.critical_path(P, {**W, "EBS_Volume_Type": "gp3"}, tiers)
    t = r["table"].set_index("tier")
    assert t.loc["Cache", "start_p90_min"] == 0 and not t.loc["Cache", "critical"]  # runs alongside DB
    assert r["total_p90_min"] == pytest.approx(t.loc["DB", "restore_p90_min"] + t.loc["API", "restore_p90_min"])
    ransom = S.critical_path(P, {**W, "EBS_Volume_Type": "gp3"}, tiers, incident="Ransomware")
    extra = 120 + 0.02 * (500 + 100)  # isolation once, forensics per GB on the critical chain
    assert ransom["total_p90_min"] == pytest.approx(r["total_p90_min"] + extra)


@pytest.mark.parametrize("tiers, match", [
    (pd.DataFrame({"Tier": ["A", "B"], "Size_GB": [1, 1], "Depends_On": ["B", "A"]}), "cycle"),
    (pd.DataFrame({"Tier": ["A", "A"], "Size_GB": [1, 1], "Depends_On": ["", ""]}), "unique"),
    (pd.DataFrame({"Tier": ["A"], "Size_GB": [0], "Depends_On": [""]}), "greater than 0"),
    (pd.DataFrame({"Tier": ["A"], "Size_GB": [10], "Depends_On": ["Z"]}), "unknown"),
])
def test_critical_path_rejects_bad_stacks(tiers, match):
    with pytest.raises(ValueError, match=match):
        S.critical_path(P, W, tiers)


def test_chaos_simulation():
    calm = S.chaos_simulation(P, W, 0)
    assert np.allclose(calm["finish_min"], predict(P, W)["p50_min"])
    rough = S.chaos_simulation(P, W, 40, incident="Ransomware", sla_min=1000)
    assert rough["median_min"] > calm["median_min"] + ops.incident_overhead("Ransomware", W["VM_Size_GB"]) - 1
    assert 0 <= rough["breach_probability"] <= 1 and rough["histogram"].runs.sum() == 500
    assert S.chaos_simulation(P, W, 40)["finish_min"].tolist() == S.chaos_simulation(P, W, 40)["finish_min"].tolist()
    with pytest.raises(ValueError):
        S.chaos_simulation(P, W, 150)


def test_multi_cloud():
    r = S.multi_cloud(1000)
    t = r["table"]
    assert sorted(t.platform) == ["AWS", "Azure", "VMware"] and list(t.p90_min) == sorted(t.p90_min)
    assert r["fastest"] == t.platform.iloc[0] and (t.downtime_cost_p90 > 0).all()
    slow_aws = S.multi_cloud(1000, workloads={"aws": {"EBS_Volume_Type": "st1"}})["table"].set_index("platform")
    assert slow_aws.loc["AWS", "p50_min"] > t.set_index("platform").loc["AWS", "p50_min"] * 3
    ransom = S.multi_cloud(1000, incident="Ransomware")["table"].set_index("platform")
    assert ransom.loc["AWS", "p50_min"] == pytest.approx(t.set_index("platform").loc["AWS", "p50_min"] + 140)
    with pytest.raises(ValueError):
        S.multi_cloud(0)
