"""Complete what-if scenarios for the frontend: application critical path, chaos simulation, multi-cloud comparison.

Each takes a workload and returns everything a screen needs, including a one-line message."""
import numpy as np
import pandas as pd

from gmcr import ops
from gmcr.config import PLATFORMS, features, label
from gmcr.predict import forecast
from gmcr.restore import typical_config

# Default application stack: the database restores first, the app tier after it, the web tier last.
DEFAULT_TIERS = pd.DataFrame({"Tier": ["Database", "App server", "Web frontend"],
                              "Size_GB": [None, 100, 50],          # None = the workload's own size
                              "Depends_On": ["", "Database", "App server"]})


def critical_path(platform: str, workload: dict, tiers: pd.DataFrame | None = None,
                  incident: str = "Routine restore", sla_min: float = 240) -> dict:
    """Restore an application tier by tier. Each tier uses the workload's settings with its own size and starts
    when the tier it depends on is back; independent tiers restore in parallel.

    The incident's fixed minutes (isolation, finding a clean restore point) happen once, before any tier starts;
    its per-GB minutes apply to each tier. Returns table, totals, chain, breach and message."""
    tiers = (DEFAULT_TIERS if tiers is None else tiers).copy()
    tiers["Size_GB"] = tiers.Size_GB.fillna(workload["VM_Size_GB"]).astype(float)
    tiers["Depends_On"] = tiers.Depends_On.fillna("").astype(str)
    if tiers.Tier.duplicated().any():
        raise ValueError("Tier names must be unique.")
    if not (tiers.Size_GB > 0).all():
        raise ValueError("Every tier needs a size greater than 0 GB.")
    f = forecast(platform, pd.DataFrame([workload] * len(tiers)).assign(VM_Size_GB=tiers.Size_GB.to_numpy()))
    fixed = ops.incident_overhead(incident, 0)
    per_gb = ops.incident_overhead(incident, tiers.Size_GB.to_numpy()) - fixed
    totals = {}
    for q in ("p50", "p90"):
        minutes = dict(zip(tiers.Tier, f[f"{q}_min"].to_numpy() + per_gb))
        finish, chain = ops.critical_path(tiers, minutes)
        totals[q] = {"minutes": minutes, "finish": {t: fixed + v for t, v in finish.items()}, "chain": chain}
    p90 = totals["p90"]
    table = pd.DataFrame({
        "tier": tiers.Tier, "size_gb": tiers.Size_GB, "depends_on": tiers.Depends_On,
        "restore_p50_min": [totals["p50"]["minutes"][t] for t in tiers.Tier],
        "restore_p90_min": [p90["minutes"][t] for t in tiers.Tier],
        "start_p90_min": [p90["finish"][t] - p90["minutes"][t] for t in tiers.Tier],
        "finish_p90_min": [p90["finish"][t] for t in tiers.Tier],
        "critical": tiers.Tier.isin(p90["chain"]).to_numpy(),
    })
    total_p50 = max(totals["p50"]["finish"].values())
    total_p90 = max(p90["finish"].values())
    breach = total_p90 > sla_min
    msg = (f"Application back in {total_p50:.0f} min (P90 {total_p90:.0f}); critical path: {' → '.join(p90['chain'])}. "
           + (f"P90 breaks the {sla_min:.0f} min SLA by {total_p90 - sla_min:.0f} min." if breach
              else f"Within the {sla_min:.0f} min SLA."))
    return {"table": table, "total_p50_min": total_p50, "total_p90_min": total_p90, "chain": p90["chain"],
            "incident_fixed_min": fixed, "breach": breach, "message": msg}


def chaos_simulation(platform: str, workload: dict, noise_pct: float, incident: str = "Routine restore",
                     sla_min: float = 240, runs: int = 500, seed: int = 0) -> dict:
    """Recovery under unstable throughput: the restore part varies, the incident overhead does not."""
    if not 0 <= noise_pct <= 100:
        raise ValueError("Instability must be between 0 and 100%.")
    p50 = float(forecast(platform, pd.DataFrame([workload])).p50_min.iloc[0])
    overhead = ops.incident_overhead(incident, workload["VM_Size_GB"])
    finish = ops.chaos(p50, noise_pct, runs, seed) + overhead
    breach_p = float(np.mean(finish > sla_min))
    counts, edges = np.histogram(finish, bins=30)
    msg = (f"With {noise_pct:.0f}% instability: median {np.median(finish):.0f} min, 90th percentile "
           f"{np.quantile(finish, 0.9):.0f} min, {breach_p:.0%} chance of breaking the {sla_min:.0f} min SLA.")
    return {"finish_min": finish, "nominal_min": p50 + overhead, "median_min": float(np.median(finish)),
            "p90_min": float(np.quantile(finish, 0.9)), "worst_min": float(finish.max()), "breach_probability": breach_p,
            "histogram": pd.DataFrame({"from_min": edges[:-1], "to_min": edges[1:], "runs": counts}), "message": msg}


def multi_cloud(size_gb: float, sla_min: float = 240, cost_per_hour: float = 25000,
                incident: str = "Routine restore", workloads: dict | None = None) -> dict:
    """The same VM size on AWS, Azure and VMware. Each platform uses its most common setup unless `workloads`
    gives one ({"aws": {...}}). Costs are downtime costs; cloud list prices are not included."""
    if not size_gb > 0:
        raise ValueError("VM size must be greater than 0 GB.")
    rows = []
    for p, cfg in PLATFORMS.items():
        w = {**typical_config(p), **(workloads or {}).get(p, {}), "VM_Size_GB": size_gb}
        f = forecast(p, pd.DataFrame([w])).iloc[0]
        overhead = ops.incident_overhead(incident, size_gb)
        p50, p90 = float(f.p50_min) + overhead, float(f.p90_min) + overhead
        imp = ops.impact(p90, float(f.friction_min), sla_min, cost_per_hour)
        rows.append({"platform": cfg["name"], "setup": ", ".join(f"{label(c)} {w[c]}" for c in features(p) if c != "VM_Size_GB"),
                     "physics_min": float(f.physics_min), "p50_min": p50, "p90_min": p90,
                     "friction_min": float(f.friction_min), "downtime_cost_p90": imp["downtime_cost"],
                     "sla": "breach" if imp["breach"] else "met", "extrapolated": bool(f.extrapolated)})
    table = pd.DataFrame(rows).sort_values("p90_min", ignore_index=True)
    best, worst = table.iloc[0], table.iloc[-1]
    # Ranked on P90, the number the SLA is judged on.
    msg = (f"Fastest worst case for {size_gb:,.0f} GB: {best.platform} (P90 {best.p90_min:.0f} min); "
           f"slowest: {worst.platform} (P90 {worst.p90_min:.0f} min). "
           f"{(table.sla == 'met').sum()} of 3 meet the {sla_min:.0f} min SLA.")
    return {"table": table, "fastest": best.platform, "message": msg}
