"""HTTP API for the React frontend.

Run from the Codethon folder:  .venv/Scripts/python -m uvicorn gmcr.api:app --port 8000
"""
import json
import math
from contextlib import asynccontextmanager
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from gmcr import ops, restore, scenarios
from gmcr.config import DEMO_CASE, HELP, MODEL_DIR, PLATFORMS, features, label
from gmcr.fleet import Fleet
from gmcr.predict import load_model, predict
from gmcr.report import pdf
from gmcr.telemetry import CAUSES, CAUSE_LABEL

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the fleet replay and incident diagnoses (about 10 s) so no page waits on first open.
    incident_list()
    yield


app = FastAPI(title="GMCR API", lifespan=lifespan)


def clean(x):
    """numpy / pandas / NaN → plain JSON values."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return clean(x.to_dict("records"))
    if isinstance(x, pd.Series):
        return clean(x.to_dict())
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


@app.exception_handler(ValueError)
async def bad_input(_: Request, err: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(err)})


@app.exception_handler(FileNotFoundError)
async def not_trained(_: Request, err: FileNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(err)})


@lru_cache(maxsize=1)
def fleet() -> Fleet:
    return Fleet()


@lru_cache(maxsize=1)
def incident_list() -> list:
    return clean(fleet().incident_list())


# ---------- shared request bodies ----------

class Scenario(BaseModel):
    platform: str
    workload: dict
    incident: str = "Routine restore"
    sla_min: float = Field(240, gt=0)
    cost_per_hour: float = Field(25000, ge=0)


def _check_platform(p: str):
    if p not in PLATFORMS:
        raise ValueError(f"Unknown platform {p!r}")


# ---------- metadata ----------

@app.get("/api/meta")
def meta():
    plats = {}
    for p, cfg in PLATFORMS.items():
        b = load_model(p)
        typical = restore.typical_config(p)
        inputs = []
        for c in features(p):
            item = {"name": c, "label": label(c), "help": HELP.get(c, ""), "default": typical[c]}
            if c in b["categories"]:
                item |= {"kind": "select", "options": b["categories"][c]}
            else:
                values = sorted(b["scored"][c].unique().tolist())
                if len(values) <= restore.MAX_WHATIF_VALUES:
                    item |= {"kind": "select", "options": values}
                else:
                    lo, hi = b["numeric_range"][c]
                    item |= {"kind": "number", "min": 1, "trained_min": lo, "trained_max": hi}
            inputs.append(item)
        plats[p] = {"name": cfg["name"], "inputs": inputs, "algorithm": b["algorithm"], "cv_scores": b["cv_scores"]}
    return clean({"platforms": plats, "incidents": list(ops.INCIDENTS), "demo_case": DEMO_CASE,
                  "defaults": {"sla_min": 240, "cost_per_hour": 25000}})


# ---------- restore forecast ----------

def _forecast(s: Scenario) -> dict:
    _check_platform(s.platform)
    fc = predict(s.platform, s.workload)
    overhead = ops.incident_overhead(s.incident, s.workload["VM_Size_GB"])
    p50, p90 = fc["p50_min"] + overhead, fc["p90_min"] + overhead
    imp = ops.impact(p90, fc["friction_min"], s.sla_min, s.cost_per_hour)
    factors = restore.factors(s.platform, s.workload)
    what_if = restore.what_if(s.platform, s.workload)
    b = load_model(s.platform)
    lo, hi = b["numeric_range"]["VM_Size_GB"]
    size = float(s.workload["VM_Size_GB"])
    sizes = np.unique(np.concatenate([np.linspace(10, max(5000.0, size * 1.5), 60), [size]]))
    scal = restore.scalability(s.platform, s.workload, sizes)
    extra = np.array([ops.incident_overhead(s.incident, x) for x in sizes])
    return {
        "forecast": fc, "overhead_min": overhead, "total_p50_min": p50, "total_p90_min": p90, "impact": imp,
        "sla_min": s.sla_min,
        "breakdown": {"physics_min": min(fc["physics_min"], fc["p50_min"]), "friction_min": fc["friction_min"],
                      "overhead_min": overhead},
        "factors": [{"input": k, "label": label(k), **r} for k, r in factors.iterrows()],
        "typical_min": restore.typical_minutes(s.platform),
        "what_if": [{**r, "label": label(r["input"])} for r in what_if.to_dict("records")],
        "scalability": [{"size_gb": r.VM_Size_GB, "p50_min": r.p50_min + e, "p90_min": r.p90_min + e,
                         "physics_min": r.physics_min, "extrapolated": r.extrapolated}
                        for r, e in zip(scal.itertuples(), extra)],
        "trained_size_gb": [lo, hi], "algorithm": b["algorithm"], "cv_scores": b["cv_scores"],
    }


@app.post("/api/forecast")
def forecast_endpoint(s: Scenario):
    return clean(_forecast(s))


@app.post("/api/report")
def report_endpoint(s: Scenario):
    r = _forecast(s)
    flags = [f for f in r["factors"] if f["flag"]]
    data = pdf(s.platform, s.workload, r["forecast"], r["impact"], s.sla_min, s.cost_per_hour, s.incident,
               r["overhead_min"], flags, r["what_if"], r["cv_scores"])
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="gmcr-{s.platform}-forecast.pdf"'})


# ---------- scenarios ----------

class Tier(BaseModel):
    tier: str
    size_gb: float | None = None
    depends_on: str = ""


class CriticalPath(Scenario):
    tiers: list[Tier] | None = None


class Chaos(Scenario):
    noise_pct: float = 20


class MultiCloud(BaseModel):
    size_gb: float
    incident: str = "Routine restore"
    sla_min: float = Field(240, gt=0)
    cost_per_hour: float = Field(25000, ge=0)


@app.post("/api/scenarios/critical-path")
def critical_path_endpoint(s: CriticalPath):
    _check_platform(s.platform)
    tiers = None if not s.tiers else pd.DataFrame(
        {"Tier": [t.tier.strip() for t in s.tiers], "Size_GB": [t.size_gb for t in s.tiers],
         "Depends_On": [t.depends_on.strip() for t in s.tiers]})
    return clean(scenarios.critical_path(s.platform, s.workload, tiers, s.incident, s.sla_min))


@app.post("/api/scenarios/chaos")
def chaos_endpoint(s: Chaos):
    _check_platform(s.platform)
    r = scenarios.chaos_simulation(s.platform, s.workload, s.noise_pct, s.incident, s.sla_min)
    r.pop("finish_min")
    return clean(r)


@app.post("/api/scenarios/multi-cloud")
def multi_cloud_endpoint(s: MultiCloud):
    return clean(scenarios.multi_cloud(s.size_gb, s.sla_min, s.cost_per_hour, s.incident))


# ---------- model evidence ----------

@app.get("/api/models")
def models_endpoint():
    comparison = pd.read_csv(MODEL_DIR / "comparison.csv")
    best = comparison.loc[comparison.groupby(["platform", "algorithm"]).mae_min.idxmin()]
    ops_card = json.loads((MODEL_DIR / "ops_card.json").read_text())
    confusion = pd.read_csv(MODEL_DIR / "root_cause_confusion.csv", index_col=0)
    restore_platforms = {}
    for p, cfg in PLATFORMS.items():
        s = load_model(p)["scored"]
        restore_platforms[p] = {
            "name": cfg["name"],
            "importance": [{"input": k, "label": label(k), "value": v} for k, v in restore.global_importance(p).items()],
            "points": [{"id": str(r[0]), "actual": r[1], "expected": r[2], "flagged": r[3]}
                       for r in s[[cfg["id"], "Restore_Time_Min", "expected", "flagged"]].itertuples(index=False)],
            "flagged_count": int(s.flagged.sum()),
        }
    return clean({
        "comparison": best[["platform", "algorithm", "mae_min", "r2", "median_error"]],
        "restore": restore_platforms,
        "failure_risk": ops_card["failure_risk"],
        "root_cause": ops_card["root_cause"],
        "causes": [{"cause": c, "label": CAUSE_LABEL[c]} for c in CAUSES],
        "confusion": confusion.to_numpy().tolist(),
    })


# ---------- fleet & incidents ----------

@app.get("/api/fleet/overview")
def fleet_overview():
    f = fleet()
    return clean({"minutes": f.minutes, "vms": len(f.vms), "timeline": f.overview(),
                  "incidents": [{"incident_id": i.incident_id, "vm_id": i.vm_id, "fail_minute": i.fail_minute}
                                for i in f.incidents.itertuples()]})


@app.get("/api/fleet/triage")
def fleet_triage(minute: int, sla_min: float = 240):
    return clean({"minute": minute, "rows": fleet().triage(minute, sla_min)})


@app.get("/api/incidents")
def incidents():
    return incident_list()


@app.get("/api/incidents/{incident_id}")
def incident(incident_id: str):
    try:
        return clean(fleet().incident_detail(incident_id))
    except KeyError as err:
        raise HTTPException(404, err.args[0])
