# GMCR: Project Review and Improvement Plan

Review date: 2026-10-19

## 0. What was reviewed, and how to read this report

| Input | Status | How it was reviewed |
|---|---|---|
| Project idea and PPT | **PPT file not received.** The NotebookLM summary of the 7-slide deck (pasted earlier) was used as the project idea. | Read in full |
| `data_test/aws_restore_data.csv` | Received | Profiled, then 5-fold cross-validated models trained on it |
| `data_test/azure_restore_data.csv` | Received | Same |
| `data_test/vmware_restore_dataset.csv` | Received | Same |
| https://gmcr-tool-v2.streamlit.app/ | Public, reachable | Driven in headless Chrome: all 8 tabs, all 3 platforms, all 4 algorithms, every help tooltip, both code generators, the Chaos Simulator, and 8 input probes |

Every claim carries one of these tags:

- **[SITE]**: seen on the live website during this review.
- **[DATA]**: measured on the three CSV files.
- **[PROPOSED]**: our recommendation. Not on the site today.
- **[NOT VERIFIED]**: could not be checked. Reason given.

---

## 1. The project as we understand it

- **Problem.** Teams estimate cloud recovery time (RTO) as `data size ÷ bandwidth`. This assumes a perfect
  vacuum and ignores friction: API latency, TCP handshakes, disk hydration, throttling.
- **Solution.** A Gradient Boosting model learns real restore behaviour from historical restores. It predicts
  RTO, isolates **cloud friction** (forecast − physics baseline), projects RTO across VM sizes, and explains
  each prediction with feature importance and SHAP.
- **Workflow.** Offline: clean → scale → train → save model + scaler. Online: workload vector → physics
  baseline → model → forecast → friction, scalability curve, explanation.
- **Expected outcome.** A trustworthy, explainable RTO forecast for failover decisions.
- **Theme gap (from the earlier review).** The deck has no threat model. The pipeline design in
  [GMCR_Pipeline.md](GMCR_Pipeline.md) adds attack scenarios and a breach cost layer.

---

## 2. The website: what it does (verified)

The site is titled **"GMCR Orchestrator"** and its header reads **"DR Orchestrator & IaC Generator"**. It is
an implementation of the same GMCR idea.

### 2.1 Sidebar [SITE]

| Control | Values seen |
|---|---|
| Cloud Platform | Azure, AWS, VMware |
| Data Refinery: Outlier Exclusion Filter | Range slider 0.01 to 0.95. Tooltip: left handle drops "Too Good To Be True" restores, right handle drops "Catastrophic Failures" |
| Prediction Algorithm | Gradient Boosting, Random Forest, Linear Regression, Theoretical Physics (Formula) |
| Status message | "⚠️ No data found in DB. Using default/empty set." on **all three** platforms |

### 2.2 Inputs per platform [SITE]

| Platform | Inputs on the site |
|---|---|
| Azure | VM Size (GB) 500, Restore Method (Instant, Vault), Disk Tier (Standard_HDD, Standard_SSD, Premium_SSD, Ultra), Region (East US, West Europe, SE Asia, West US), Bandwidth (Mbps) 1000 |
| AWS | VM Size (GB) 500, Volume Type (st1, gp2, gp3, io2), Provisioned IOPS 3000, Snapshot Age (Days) 7, Region (us-east-1, us-west-1, eu-central-1) |
| VMware | VM Size (GB) 500, Transport Mode (NBD, HotAdd, SAN), Network Speed (Gbps) (1, 10, 25), Target Storage (HDD, SSD, NVMe) |
| All | Target SLA (Minutes) 60, Cost of Downtime ($/Hour) 25000 |

### 2.3 Tabs [SITE]

| Tab | What it does (observed) |
|---|---|
| **AI Orchestrator** | Final Prediction, Cloud Friction, FinOps panel (SLA status, safety buffer or projected loss), "AI Mitigation Recommendation", Model Trust (R² score and AI Correction %), Scalability Curve, "Global Recovery Telemetry" map with an arc from vault to target region |
| **Architecture & IaC** | Generated DR topology graph (System Vault → Zero-Trust FW (Port 443) → Clean Room → Recovery Node → Storage), "SRE Critical Path Sequencer" (Database → Middleware → Frontend/WAF), buttons that generate Terraform and a Kubernetes manifest, each with a download button |
| **Chaos Simulator** | Base Bandwidth slider (10 to 2000 Mbps) and Instability (Noise) slider (0 to 100). START runs a live countdown. At 500 Mbps and noise 20 it showed "Remaining 136.3 min, Speed 62.5 MB/s", then fluctuated (134.2, 134.6 min) |
| **Validation & Explainability** | Placeholders only: "Train a model to see metrics" and "Train a model … to see SHAP explainability analysis". With Physics selected: "Manual Formula: No validation data available." |
| **Historical Trends** | "Not enough historical data for Azure/AWS/VMware. Submit feedback entries or add training data to see trends." |
| **Multi-Cloud & Cost Analysis** | Side-by-side table (Physics RTO, compute, storage, egress, total cost), capability radar, cost breakdown bar chart, "Historical Performance from Database" (empty) |
| **Executive Report** | Lists PDF contents (cover, executive summary, prediction details, configuration, cost, AI recommendations, disclaimer) and a "Generate & Download PDF Report" button |
| **Feedback Loop** | Form for a real restore: platform inputs + "Actual Time Taken (min)" + "Submit Feedback" |

### 2.4 Formulas worked out from the site's outputs [SITE]

These were inferred from the numbers the site produced, and each one reproduces the site's output exactly.

| Item | Formula | Site output that confirms it |
|---|---|---|
| Azure physics baseline | `size_GB × 8192 / bandwidth_Mbps / 60` | 500 GB at 1000 Mbps → **68.3 min** |
| AWS physics baseline | `size_GB × 1024 / (IOPS × 0.25 MB) / 60` | 500 GB at 3000 IOPS → **11.4 min** |
| VMware physics baseline | `size_GB × 8192 / (Gbps × 1000) / 60` | 500 GB at 1 Gbps → 68.3 min |
| "Theoretical Physics" prediction | baseline + about 2.3 min | 68.3 → **70.6 min** (friction shown as +2.3) |
| Projected loss on SLA breach | `(RTO − SLA) / 60 × cost_per_hour` | (70.6 − 60) / 60 × 25,000 = **$4,423.61** |
| Critical path sequencer | Middleware = 0.4 × DB, Frontend = 0.2 × DB | 70.6 / **28.2** / **14.1**, total **113.0 min** |

### 2.5 Problems found on the site [SITE]

| # | Severity | Problem | Evidence |
|---|---|---|---|
| 1 | **Critical** | The AI models have no data. Gradient Boosting, Random Forest and Linear Regression all return **0.0 min** on every platform. | "No data found in DB" on Azure, AWS and VMware; Final Prediction 0.0 min; Model Trust "Low Confidence: 0.0%" |
| 2 | **Critical** | Because the prediction is 0.0 min, the site shows **"✅ SLA SECURE, Safety Buffer 60.0 mins"**. A broken model reports a safe status. | Same screens as #1 |
| 3 | High | Friction is shown as negative: **−68.3 min** (Azure), **−11.4 min** (AWS). | AI Orchestrator tab |
| 4 | High | The bottleneck flag is one fixed message per platform (see section 3). | 8 input probes |
| 5 | High | The SHAP and validation tab never renders, so explainability, a core claim in the deck, is not demonstrable today. | Validation & Explainability tab |
| 6 | Medium | Physics mode is labelled **"Using Linear Regression Formula"** under Model Trust. | All platforms, physics selected |
| 7 | Medium | Dropdown values do not match the datasets (section 5.4), so a model trained on these CSVs would never see the site's "SE Asia", "West US" or "us-west-1". | Compare 2.2 with section 5 |
| 8 | Medium | Multi-Cloud tab: with Available Bandwidth 1000 Mbps, Azure physics RTO is 68.3 min but VMware is **6.8 min**. 6.8 matches 10 Gbps, so the VMware column seems to ignore the bandwidth input. | Multi-Cloud table |
| 9 | Medium | Generated Terraform would not pass `terraform validate`. It uses `azurerm_*` resources with AWS-style arguments (`region`, `instance_type`, `ingress { from_port … cidr_blocks }`) and omits required ones such as `resource_group_name`. | Generated `main.tf` (read, not executed) |
| 10 | Medium | Generated Kubernetes manifest has invalid names. `System-recovery-pvc` and image `System/agent:latest` contain uppercase letters, which Kubernetes and image references reject. The StatefulSet has no `selector` or `serviceName`. | Generated `manifest.yaml` (read, not applied) |
| 11 | Low | The Multi-Cloud footnote is garbled: the `$` signs in "$/hr … $/GB" are rendered as LaTeX math. | Multi-Cloud tab |
| 12 | Low | The critical path uses fixed ratios (0.4 and 0.2), not real dependencies. | Section 2.4 |
| 13 | Low | The topology graph renders very small at 1440 px wide and is hard to read. | Screenshot of Architecture tab |

### 2.6 Not verified

| Item | Reason |
|---|---|
| PDF report content | The button was not clicked; downloads do not complete in a headless browser |
| Feedback submission | **Deliberately not submitted.** It would write to the live database |
| SHAP charts, validation metrics, historical trends | Never render because the database is empty |
| Outlier filter effect | No data to filter |
| Why the database is empty | Unknown. One plausible cause is Streamlit Community Cloud's non-persistent file system, if the app uses a local SQLite file. Not confirmed. |

---

## 3. Does the site identify where an issue came from, and flag it?

**Answer: no, not in any verified, data-driven way.** It has a generic warning, not source identification.

### What exists [SITE]

When the predicted RTO exceeds the SLA, the AI Orchestrator tab shows:

> 🚨 SLA BREACH DETECTED, Projected Loss: $4,423.61
> ⚠️ AI Mitigation Recommendation
> 👉 Bottleneck: Network Bandwidth is saturating. Upgrade base bandwidth.

### How it behaves (8 probes plus 3 platforms)

| Observation | Evidence |
|---|---|
| Shown **only on SLA breach**. With the SLA met, no flag appears. | Azure 10000 Mbps → "SLA SECURE", no recommendation |
| Text depends on the **platform**, not the inputs. | Azure and VMware: "Network Bandwidth is saturating. Upgrade base bandwidth." AWS: "Throttled Provisioned IOPS. Upgrade to io2 Block Express." |
| The same Azure message appeared at 500 GB/1000 Mbps, 4000 GB/1000 Mbps and 4000 GB/100 Mbps. | Probes |
| It appears in physics mode, where no model runs. So it is not produced by the AI. | Physics-mode probes |
| It never names a specific VM, region, component or time. | All screens |
| The map draws an arc from vault to target region, coloured green to red. Its tooltip says "Live 3D routing from the Primary Vault to the target edge region." The arc did not change with any input we tried. It is a visual, not an issue flag. | AI Orchestrator map |

Caveat: in physics mode, Disk Tier, Restore Method and Region do not change the prediction. So we could not
test whether the message changes in AI mode, which needs data the site does not have.

### The data contradicts these messages [DATA]

| Platform | Site's flag | What the data says |
|---|---|---|
| AWS | "Throttled Provisioned IOPS" | Within gp3, restore time per GB is the same at 3,000, 16,000 and 64,000 IOPS (0.0855, 0.0836, 0.0865 min/GB). IOPS importance ≈ 0. The real driver is **volume type**: st1 median 500.2 min vs io2 51.4 min. |
| Azure | "Network Bandwidth is saturating" | Bandwidth has no correlation with restore time (r = 0.006). In Vault restores, 10000 Mbps is no faster than 1000 Mbps. The real drivers are **Restore Method** (Instant median 13.8 min vs Vault 234.6 min) and **Disk Tier**. |
| VMware | "Network Bandwidth is saturating" | Partly true: 1 Gbps is about 2× slower per GB than 10 Gbps. But the larger drivers are **Concurrency** (0.117 → 0.753 min/GB from level 1 to 5), **Transport Mode** (NBD median 1213 min vs SAN 282) and **Target Storage** (HDD 976 vs NVMe 231). |

So the current flag would send a team to fix the wrong thing on all three platforms.

---

## 4. Proposed: issue-source identification and flagging [PROPOSED]

The aim is to keep the site's layout and workflow, and replace the fixed message with three levels of
"source", each driven by the model and the data.

| Level | Question it answers | Method | Where it appears (existing tab) |
|---|---|---|---|
| **L1 Factor** | Which input makes *this* forecast slow? | Per-prediction SHAP on a log-target model, folded back to the original inputs | AI Orchestrator: replaces "AI Mitigation Recommendation" |
| **L2 Record** | Which past restores were abnormal? | Residual anomaly: actual vs cross-validated prediction, robust z-score | Validation & Explainability: new "Flagged restores" table |
| **L3 Location** | Where do problems concentrate? | Flag rate and mean friction per region, tier, transport mode | Global Recovery Telemetry map + Historical Trends |

### L1: factor-level source (why is this forecast slow?)

Train on `log(Restore_Time_Min)`. SHAP values on a log scale become **multipliers**, which read naturally:
"st1 volume makes this restore 4.0× slower than a typical one."

Tested on the slowest restore in each dataset [DATA]:

| Dataset | Restore | Typical → this one | Top factors (multiplier) |
|---|---|---|---|
| AWS | i-5d58953418c5ca73, 1971 GB, st1, 64000 IOPS | 99 → 874 min (actual 875) | **EBS_Volume_Type ×4.03**, VM_Size_GB ×2.00, Snapshot_Age ×1.09, IOPS ×1.00 |
| Azure | VM-764, 3824 GB, Standard_HDD, Vault, GRS | 71 → 2289 min (actual 2286) | **Disk_Tier ×3.51**, **Restore_Method ×3.17**, VM_Size ×2.32, Redundancy ×1.25, Bandwidth ×1.00 |
| VMware | VM 843, 2259 GB, NBD, HDD, 1 Gbps, concurrency 5 | 351 → 11883 min (actual 11930) | **Target_Storage ×2.53**, **Transport_Mode ×2.45**, VM_Size ×2.39, Concurrency ×1.99, Network ×1.15 |

For AWS, the site's flag would say "Throttled IOPS". L1 says the cause is the st1 volume type, with IOPS at ×1.00.

**Flag rule:** any controllable factor ≥ ×1.5 is flagged red, from ×1.2 to ×1.5 amber. VM size is shown but
never flagged, because it is the workload, not a fault.

**Recommendation with proof (what-if):** for each flagged factor, re-run the model with the better option and
show the minutes saved. Tested on the slowest AWS restore [DATA]: st1 874 min → gp2 302, **gp3 168**,
**io2 103 min** (model estimates). The site's recommendation becomes a model result instead of fixed text.

| Factor | Better option to test |
|---|---|
| AWS EBS_Volume_Type | st1 → gp3 or io2 |
| Azure Restore_Method | Vault → Instant |
| Azure Disk_Tier | Standard_HDD → Premium_SSD or Ultra |
| VMware Transport_Mode | NBD → HotAdd or SAN |
| VMware Target_Storage | HDD → SSD or NVMe |
| VMware Concurrency_Level | Lower it (stagger restores) |
| VMware Network_Gbps | 1 → 10 |

```python
import numpy as np, pandas as pd, shap

def factor_flags(pipe, cat_cols, row: pd.DataFrame):
    """Return {input column: multiplier} for one workload, largest first."""
    pre, model = pipe.named_steps["pre"], pipe.named_steps["m"]
    Xt = pre.transform(row)
    sv = np.asarray(shap.TreeExplainer(model).shap_values(Xt)).reshape(-1)
    names = pre.get_feature_names_out()
    src = [next((c for c in cat_cols if n.startswith(c + "_")), n) for n in names]  # fold one-hot back
    mult = np.exp(pd.Series(sv, index=names).groupby(src).sum())
    return mult.sort_values(ascending=False)

def what_if(pipe, row, column, better_value):
    alt = row.assign(**{column: better_value})
    return float(np.exp(pipe.predict(row))[0]), float(np.exp(pipe.predict(alt))[0])
```

### L2: record-level source (which restore went wrong?)

1. Get a **cross-validated** prediction for every historical restore, so no row is scored by a model that saw it.
2. Compute `r = log(actual) − log(predicted)`.
3. Compute a robust z-score: `z = (r − median) / (1.4826 × MAD)`.
4. Flag `|z| > 3.5`. A positive z means slower than expected, a negative z means suspiciously fast.
5. List flagged rows with their ID (`Instance_ID`, `VM_Resource_ID`, `VM_ID`) and their L1 factors.

Tested [DATA]: AWS flags **50** of 1000 rows, Azure **43**, VMware **23**. Example: AWS
`i-ed98e0bb8ba4a9e5` (326 GB st1) took 146.0 min where the model expected 128.7 (z = +4.6).

Caveat: the datasets have **no root-cause labels**, so we cannot tell whether the flagged rows are injected
faults or noise. Section 7 explains how to test the detector with injected faults.

**Feedback Loop link:** when a user submits a real restore on the existing Feedback tab, score it at once and
show "Flagged: 2.3× slower than expected. Top factor: Disk_Tier."

### L3: location-level source (where do problems concentrate?)

- Group flagged rows and friction by Region, Volume Type / Disk Tier / Transport Mode.
- Flag a group when its flag rate is more than 2× the overall rate and it has at least 10 rows.
- On the existing map, colour each region by flag rate and friction instead of a fixed arc.

Caveat [DATA]: in these datasets **Region has no measurable effect** (importance ≈ 0 on AWS and Azure), so
L3 will show no hot region with the current data. To demo it, inject a regional fault (section 7) or use
real logs. Azure resource IDs cannot localise below region either: all 1000 rows have a different
subscription and the same resource group, "Prod".

---

## 5. The three datasets

### 5.1 Profile [DATA]

| | AWS | Azure | VMware |
|---|---|---|---|
| Rows × columns | 1000 × 7 | 1000 × 8 | 1000 × 8 |
| Nulls, duplicate rows, duplicate IDs | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| VM size range | 51 to 1,996 GB | 65 to 4,093 GB | 41 to 2,495 GB |
| Restore time range (median) | 12.6 to 875 min (95.6) | 2.1 to 2,286 min (90.6) | 6.1 to 11,930 min (370.7) |
| Target skew | 2.82 | 2.95 | 5.00 |
| Restores faster than physics | 0% | **49.1%** | 0% |
| Timestamps | None | None | None |
| Incident or attack type | None | None | None |

### 5.2 How well each approach fits (5-fold cross-validation) [DATA]

| Model | AWS MAE / R² | Azure MAE / R² | VMware MAE / R² |
|---|---|---|---|
| Physics formula | 131.2 / −0.78 | 190.7 / −0.16 | 660.3 / −0.33 |
| Linear regression | 33.6 / 0.84 | 117.3 / 0.62 | 383.1 / 0.59 |
| Random forest | 3.9 / 0.998 | 9.1 / 0.989 | 90.7 / 0.930 |
| Gradient Boosting | **3.0** / 0.999 | 21.1 / 0.977 | 121.7 / 0.918 |
| **Gradient Boosting, log target** | 3.3 / 0.998 | **7.9** / **0.994** | **50.3** / **0.985** |

MAE is in minutes. Takeaways:

- The deck's core claim holds on all three datasets: ML beats the physics formula many times over.
- Plain Gradient Boosting loses to Random Forest on Azure and VMware. With a **log target** it is best or
  near-best everywhere. Use the log target.
- R² of 0.99+ means the data follows a formula with little noise, which suggests synthetic data. Say so to judges.

### 5.3 What drives restore time [DATA]

| Dataset | Strong drivers | No measurable effect |
|---|---|---|
| AWS | VM size, EBS volume type (st1 ≫ gp2 > gp3 > io2), a small effect from snapshot age | Provisioned IOPS, Region |
| Azure | Restore method (Vault ≫ Instant), VM size, disk tier | Network bandwidth, Region |
| VMware | VM size, concurrency, transport mode (NBD worst), target storage (HDD worst), network at 1 Gbps | Disk provisioning |

### 5.4 Data quality issues [DATA]

| Issue | Detail | Fix |
|---|---|---|
| Category names differ from the site | Azure data "Southeast Asia" vs site "SE Asia"; site "West US" not in data. AWS data "us-west-2" vs site "us-west-1". | Build dropdowns from the training data's unique values |
| Inputs missing on the site | VMware `Concurrency_Level` (2nd-strongest driver) and `Disk_Provisioning`; Azure `Vault_Redundancy` | Add them |
| Unrealistic AWS combinations | 64,000 IOPS appears on gp3, gp2 and st1. From AWS documentation (outside knowledge, not checked against the data): gp3 tops out at 16,000 IOPS, and st1 and gp2 have no provisioned IOPS. | Cap IOPS by volume type, or present it as "synthetic" |
| Physics fails on Azure Instant restores | 49% of Azure restores beat physics, so friction goes negative | Make the baseline method-aware: Instant restore mounts the snapshot instead of copying it |
| No timestamps | The Historical Trends tab cannot work with these files | Add a `Restore_Date` column (real or synthetic) |
| No incident type | The security angle cannot be trained | Add `Incident_Type` (hardware, misconfig, ransomware, wiper) |
| Size range | All below 5000 GB; AWS stops at 1996 GB | Mark the scalability curve beyond the training range as "extrapolated" |

### 5.5 Best dataset per function

| Function | Best dataset | Why |
|---|---|---|
| RTO forecast | Each platform its own | Features differ per platform; train one model each |
| Cloud friction (forecast − physics) | **AWS, VMware** | Actual ≥ physics in 100% of rows. On Azure, friction goes negative for Instant restores. |
| Scalability curve | **Azure** | Widest size range, 65 to 4,093 GB |
| L1 factor flagging (SHAP) | **VMware** | Most operational causes with clear, separate effects: transport, storage, concurrency, network |
| L2 record flagging (anomalies) | **AWS** | Tightest fit (R² 0.999), so deviations stand out; has instance IDs |
| L3 location flagging | None as-is | Region has no effect in any dataset. Needs injected faults or real logs. |
| What-if recommendation | **Azure** | Restore Method gives the clearest win (median Vault 234.6 → Instant 13.8 min) |
| Chaos simulator, critical path | **VMware** | Concurrency is a real, measured contention effect |
| Historical trends | None | No timestamps |
| Multi-cloud comparison | All three | Compare only on shared inputs (size, physics) plus each platform's model |

---

## 6. Final recommendation

### Retain [SITE]

- Sidebar layout: platform, outlier filter, algorithm choice.
- Physics baseline vs AI forecast, and the friction metric.
- FinOps panel: SLA status, safety buffer, projected loss formula.
- Model Trust panel, with a correct label.
- Scalability curve.
- The 8-tab workflow, including the Feedback Loop for retraining.
- Multi-cloud comparison, radar and cost breakdown.
- Executive PDF report.
- Plain-language tooltips on every input; they are good.

### Improve

| What | Change |
|---|---|
| Data loading | Ship the 3 CSVs with the app as seed data and load them when the DB is empty. If loading fails, **show an error, never 0.0 min and "SLA SECURE"**. |
| Model | Log target; keep Gradient Boosting; one pipeline per platform; show CV MAE in the Model Trust panel |
| Friction | Never show negative friction without an explanation; method-aware Azure baseline |
| Bottleneck message | Replace with L1 SHAP factors and what-if savings |
| Inputs | Generate options from the data; add the missing VMware and Azure inputs |
| Multi-Cloud | Use the bandwidth input for VMware; escape `$` as `\$` in markdown |
| IaC generators | Use valid `azurerm`/`aws` arguments; lowercase Kubernetes names; add `selector` and `serviceName`; run `terraform validate` in CI |
| Critical path | Take tier dependencies and sizes as inputs instead of fixed 0.4 and 0.2 ratios |
| Topology graph | Larger, or rendered top-to-bottom |

### Add [PROPOSED]

1. L1 factor flags with what-if savings (AI Orchestrator tab).
2. L2 flagged-restores table (Validation tab) and live scoring of feedback entries.
3. L3 region heat map (map and Historical Trends).
4. Prediction interval: P50 and P90 models, with the SLA compared against P90.
5. Security layer: `Incident_Type` input, attack-recovery overhead, breach cost (see [GMCR_Pipeline.md](GMCR_Pipeline.md)).
6. `Restore_Date` so Historical Trends and drift detection work.

---

## 7. Step-by-step implementation plan

| Step | Task | Done when |
|---|---|---|
| 1 | Seed the DB from `data_test/*.csv` on start-up; error banner if empty | All 4 algorithms return non-zero forecasts on all 3 platforms |
| 2 | Build dropdowns from training data; add VMware Concurrency, Disk_Provisioning, Azure Vault_Redundancy | No input value the model has never seen |
| 3 | Retrain with log target, one pipeline per platform; fix the Model Trust label | CV MAE near section 5.2: about 3, 8 and 50 min |
| 4 | Method-aware Azure baseline; clip or explain negative friction | No negative friction on default inputs |
| 5 | L1: `factor_flags()` + flag rule + what-if, replacing the fixed message | AWS st1 case flags volume type, not IOPS |
| 6 | L2: cross-validated residual z-scores; flagged table; score feedback entries | Flagged table lists IDs with top factor |
| 7 | **Detector test with injected faults**: copy a dataset, multiply restore time by 2 to 3 for 20 random rows and for all rows in one region, rerun L2 and L3 | Measure precision and recall; the injected region shows as hot on the map |
| 8 | L3: group flag rates; colour the map | Injected region turns red |
| 9 | Fix IaC output and the Multi-Cloud bugs | `terraform validate` and `kubectl apply --dry-run=client` pass |
| 10 | Add P90 model and the security layer | Ransomware scenario shows higher RTO and a dollar cost |
| 11 | Tests: physics formulas, loss formula ($4,423.61 case), flag rule, what-if | `pytest` green |
| 12 | Rehearse the demo: broken flag before, data-driven flag after | 3-minute run-through |

### Challenges

| Challenge | Mitigation |
|---|---|
| Synthetic data: SHAP will recover the generator's rules, not real cloud behaviour | Say it openly; the method is the product; show the injected-fault test |
| No root-cause labels to score L1 against | Injected-fault test (step 7) gives measurable precision and recall |
| SHAP explains the model, not causality | Phrase flags as "the model attributes …"; back them with the what-if result |
| Region has no effect in the data | L3 demo needs injected faults or real logs |
| Database persistence on Streamlit Cloud (not verified as the cause) | Seed from CSV on start-up; use a hosted DB for feedback data |
| One-hot names (`EBS_Volume_Type_st1`) are unreadable | Fold SHAP back to source columns (code in section 4) |
| Heavy skew (VMware max 11,930 min) | Log target; report median percentage error alongside MAE |

---

## Appendix: method

- Website: headless Chrome driven through the DevTools protocol, 2026-10-19. Text, tooltips and screenshots
  were captured per tab. Inputs were changed with real mouse and keyboard events. Nothing was submitted.
- Data: pandas 2.3.3, scikit-learn 1.8.0, shap 0.52.0. 5-fold `KFold(shuffle=True, random_state=0)`.
  Gradient Boosting `n_estimators=400, max_depth=3, learning_rate=0.05`. One-hot encoding for categories.
