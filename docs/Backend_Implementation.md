# GMCR Backend (hackathon demo)

**Owner:** Person B
**Status (2026-10-19): complete for the demo.** 44 tests pass. Person A can build the frontend on the
functions in section 2.
**Partner:** Person A owns the Streamlit app ([Frontend_Implementation.md](Frontend_Implementation.md)).
**Design of A/B/C:** [Incident_Pipeline.md](Incident_Pipeline.md)

---

## 1. Run

```
.venv/Scripts/python -m gmcr.train      # restore-time models: 4 libraries compared, best saved (about 90 s)
.venv/Scripts/python -m gmcr.pipeline   # failure risk (A), root cause (B), demo report (about 60 s)
.venv/Scripts/python -m pytest -q       # 44 tests (about 16 s)
```

Setup on a new machine: `python -m venv .venv`, then `.venv/Scripts/python -m pip install -r requirements.txt`.

## 2. Functions for the frontend

All forecasts are in minutes. `platform` is `"aws"`, `"azure"` or `"vmware"`. A workload is a dict with one
value per input of that platform (`config.features(platform)`).

### Inputs and display

| Call | Returns |
|---|---|
| `config.PLATFORMS[p]` | `name`, `num`, `cat`, `controllable`, `location` lists |
| `config.features(p)` | Input names, in model order |
| `config.HELP[col]`, `config.label(col)` | Tooltip text; display name (`VM_Size_GB` → `VM Size (GB)`) |
| `config.DEMO_CASE` | `{"platform": "aws", "workload": {...}}`: the demo story's restore |
| `predict.load_model(p)["categories"]` | Allowed values per categorical input (use for dropdowns) |
| `predict.load_model(p)["scored"][col].unique()` | Values seen in training for numeric inputs |
| `restore.typical_config(p)` | Most common value of every input (good defaults) |

### Main screen: forecast, cause, fix, cost

| Call | Returns |
|---|---|
| `predict.predict(p, workload)` | `algorithm, p50_min, p90_min, physics_min, friction_min, beats_physics, extrapolated` |
| `predict.forecast(p, df)` | Same, for many rows |
| `restore.factors(p, workload)` | DataFrame, index = input: `value, multiplier, controllable, flag` (`"red"`, `"amber"`, `""`) |
| `restore.typical_minutes(p)` | The typical restore the multipliers compare against |
| `restore.what_if(p, workload)` | DataFrame `input, current, change_to, p50_now, p50_after, minutes_saved` (≥ 5% savings only) |
| `restore.scalability(p, workload, sizes)` | Forecast per size, with `extrapolated` beyond the trained sizes |
| `ops.INCIDENTS`, `ops.incident_overhead(kind, size_gb)` | Assumed extra minutes for ransomware, wiper etc. |
| `ops.impact(rto_p90, friction, sla, cost_per_hour)` | `breach, buffer_min, downtime_cost, breach_cost, friction_cost` |
| `predict.load_model(p)["cv_scores"]` | `mae_min, rmse_min, r2, median_error` for the Model trust panel |

Raises `ValueError` with a readable message for a missing input, an unseen category or a non-positive number.
Show the message; never show a forecast instead.

### Validation tab

| Call | Returns |
|---|---|
| `models/comparison.csv`, `models/comparison.png` | All 4 libraries per platform (CatBoost wins all three) |
| `restore.global_importance(p)` | What drives restore time (mean \|SHAP\|) |
| `restore.flagged_restores(p)` | Unusual past restores: `expected, ratio, z`, worst first |
| `restore.location_heat(p)` | Per region: `restores, flagged, slower_than_expected, hot`; `None` for VMware |

### Other tabs

| Call | Returns |
|---|---|
| `scenarios.critical_path(p, workload, tiers=None, incident, sla_min)` | Application restored tier by tier. `table` (tier, size, depends_on, restore P50/P90, start, finish, critical), `total_p50_min`, `total_p90_min`, `chain`, `breach`, `message`. `tiers` is a DataFrame `Tier, Size_GB, Depends_On` (empty size = workload size); default is Database → App server → Web frontend. `ValueError` on a cycle, unknown or duplicate tier, or 0 GB |
| `scenarios.chaos_simulation(p, workload, noise_pct, incident, sla_min)` | 500 runs under unstable throughput: `finish_min`, `median_min`, `p90_min`, `breach_probability`, `histogram`, `message` |
| `scenarios.multi_cloud(size_gb, sla_min, cost_per_hour, incident, workloads=None)` | Same VM size on AWS, Azure, VMware (typical setup, or `workloads={"aws": {...}}`): `table` sorted by P90 with physics, P50, P90, friction, downtime cost, SLA; `fastest`; `message` |
| `iac.terraform(p, workload, p50, p90)` / `iac.kubernetes(...)` | Template text; Terraform is `None` for VMware |
| `report.pdf(p, workload, forecast, impact, sla, cost, incident, overhead, flags, what_ifs, cv_scores, incident_analysis=None)` | PDF bytes |
| `feedback.save(p, workload, actual_min)` | Logs a real restore; returns `expected_min, ratio, z, flagged` |
| `feedback.trend(p)` | Logged restores vs expected; drift verdict from 5 entries |
| `train.refit(p)` | Retrains the saved winner including feedback (a few seconds) |

### Incident pipeline (A, B, C)

| Call | Returns |
|---|---|
| `telemetry.load()` | Synthetic fleet: `fleet, metrics, events, incidents` |
| `pipeline.load_models()` | `(risk_model, cause_model)` |
| `pipeline.analyze(tel, risk, cause, vm_id, minute)` | `risk` (A), `diagnosis` (B, only when down), `plan` (C), each with a `message` |
| `pipeline.triage(tel, risk, cause, minute)` | Down or at-risk VMs, highest priority first |

## 3. Demo-case numbers (CatBoost, 2026-10-19)

AWS, 1971 GB, st1, 64,000 IOPS, us-east-1, 340 days:

| | Value |
|---|---|
| Physics formula | 2.1 min |
| Forecast P50 / P90 | 877 / 895 min (actual 875) |
| Cause flags | st1 volume type ×4.08 (red); IOPS ×1.01 |
| What-if | st1 → io2 saves 775 min (877 → 102) |
| With Ransomware | +159 min overhead (assumed); P90 1,055 min |
| At SLA 240 min, $25,000/h | Breach; P90 downtime cost $439,410 |

## 4. Model results

Restore time, CV MAE in minutes:

| | AWS | Azure | VMware |
|---|---|---|---|
| Physics formula | 131.2 | 156.7 | 660.3 |
| Gradient Boosting | 2.4 | 7.0 | 34.3 |
| LightGBM | 2.6 | 7.8 | 33.3 |
| XGBoost | 2.6 | 7.1 | 37.2 |
| **CatBoost (saved)** | **1.8** | **5.6** | **20.2** |

A, B and C results are in [Incident_Pipeline.md](Incident_Pipeline.md).

## 5. Decisions and fixes worth knowing

| Item | Why |
|---|---|
| Log target for every model | Restore times are heavily skewed; error halved or better on Azure and VMware |
| LightGBM categorical settings relaxed | Defaults gave AWS MAE 17.8 instead of 2.6 on 1,000 rows |
| CatBoost `one_hot_max_size=10` | 40× faster and more accurate; categories have at most 4 values |
| CatBoost wrapped for `clone()` | CatBoost rewrites `cat_features`, which scikit-learn 1.8 rejects |
| Sizes beyond training scaled by size | Trees go flat beyond the largest trained VM |
| Friction never negative | A restore faster than physics has no friction; `beats_physics` says why |
| Unusual restores compared with peers in the same region | One slow region would otherwise hide individual bad restores |

## 6. Known limits

- Restore data and the incident telemetry are both synthetic (R² 0.99+). Say so if asked.
- Incident overhead, playbook minutes and severity weights are assumptions.
- Region has no effect in the restore data, so no region shows as hot.
- Host failures cannot be predicted (no warning signals).
