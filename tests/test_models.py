"""Model tests. Run from the Codethon folder: .venv/Scripts/python -m pytest -q
Tests marked 'trained' need `python -m gmcr.train` to have been run first."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict

from gmcr import config
from gmcr.config import MODEL_DIR, PLATFORMS, TARGET, features, load_data, physics_baseline
from gmcr.models import ALGOS, GRIDS, build

trained = pytest.mark.skipif(not (MODEL_DIR / "comparison.csv").exists(), reason="run python -m gmcr.train first")


def test_physics_baselines():
    azure = pd.DataFrame({"VM_Size_GB": [500, 500], "Network_Bandwidth_Mbps": [1000, 1000],
                          "Restore_Method": ["Vault", "Instant"]})
    assert physics_baseline("azure", azure) == pytest.approx([68.27, 0.0], abs=0.01)
    assert physics_baseline("aws", pd.DataFrame({"VM_Size_GB": [500], "Provisioned_IOPS": [3000]}))[0] == pytest.approx(11.38, abs=0.01)
    assert physics_baseline("vmware", pd.DataFrame({"VM_Size_GB": [500], "Network_Gbps": [1]}))[0] == pytest.approx(68.27, abs=0.01)


def test_load_data_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_data("aws")
    pd.DataFrame({"VM_Size_GB": [1]}).to_csv(tmp_path / PLATFORMS["aws"]["csv"], index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_data("aws")


@pytest.mark.parametrize("algo", ALGOS)
def test_every_model_clones_fits_and_predicts_one_row(algo):
    # clone() is what cross-validation does; CatBoost broke it before the wrapper.
    d = load_data("vmware")
    X, y = d[features("vmware")], d[TARGET]
    model = clone(build(algo, PLATFORMS["vmware"]["cat"], GRIDS[algo][0])).fit(X.iloc[:800], y.iloc[:800])
    batch = model.predict(X.iloc[800:])
    single = [model.predict(X.iloc[[i]])[0] for i in range(800, 805)]
    assert (batch > 0).all()
    # LightGBM must map a lone category to the training categories, not to code 0.
    assert np.allclose(single, batch[:5])


def test_lightgbm_small_data_settings_keep_aws_accurate():
    d = load_data("aws")
    pred = cross_val_predict(build("LightGBM", PLATFORMS["aws"]["cat"], GRIDS["LightGBM"][0]),
                             d[features("aws")], d[TARGET], cv=KFold(5, shuffle=True, random_state=0))
    assert np.abs(pred - d[TARGET]).mean() < 4  # 17.8 min with LightGBM's default categorical settings


@trained
def test_saved_winner_is_lowest_error_and_beats_physics():
    results = pd.read_csv(MODEL_DIR / "comparison.csv")
    card = json.loads((MODEL_DIR / "model_card.json").read_text())
    for platform, cfg in PLATFORMS.items():
        r = results[results.platform == cfg["name"]]
        models = r[r.algorithm != "Physics formula"]
        assert card[platform]["algorithm"] == models.sort_values("mae_min").algorithm.iloc[0]
        assert models.mae_min.min() * 10 < r[r.algorithm == "Physics formula"].mae_min.iloc[0]
        assert card[platform]["cv_scores"]["r2"] > 0.95


@trained
@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_predict_slowest_restore(platform):
    from gmcr.predict import predict
    d = load_data(platform)
    i = d[TARGET].idxmax()
    r = predict(platform, d.loc[i, features(platform)].to_dict())
    assert r["p90_min"] >= r["p50_min"] > 0 and r["friction_min"] >= 0
    assert r["p50_min"] == pytest.approx(d.loc[i, TARGET], rel=0.05)


@trained
def test_predict_rejects_bad_input():
    from gmcr.predict import predict
    w = {"VM_Size_GB": 500, "EBS_Volume_Type": "gp3", "Provisioned_IOPS": 3000, "Snapshot_Age_Days": 7, "Region": "us-east-1"}
    assert predict("aws", w)["p50_min"] > 0
    with pytest.raises(ValueError, match="never seen"):
        predict("aws", {**w, "Region": "us-west-1"})
    with pytest.raises(ValueError, match="Missing"):
        predict("aws", {k: v for k, v in w.items() if k != "Region"})
    with pytest.raises(ValueError, match="greater than 0"):
        predict("aws", {**w, "VM_Size_GB": 0})
    assert predict("aws", {**w, "VM_Size_GB": 9000})["extrapolated"]
