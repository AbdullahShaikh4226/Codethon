"""Business impact, incident scenarios, critical path and the chaos simulator."""
import numpy as np
import pandas as pd

# Assumption, not learned: the restore data carries no incident labels. Extra minutes on top of the restore
# forecast for isolation, forensics and finding a clean restore point: (fixed minutes, minutes per GB).
INCIDENTS = {
    "Routine restore": (0, 0.0),
    "Hardware failure": (5, 0.0),
    "Misconfiguration": (15, 0.0),
    "Ransomware": (120, 0.02),
    "Wiper malware": (180, 0.03),
}


def incident_overhead(kind: str, size_gb: float) -> float:
    if kind not in INCIDENTS:
        raise ValueError(f"Unknown incident {kind!r}; expected one of {list(INCIDENTS)}")
    fixed, per_gb = INCIDENTS[kind]
    return fixed + per_gb * size_gb


def impact(rto_min: float, friction_min: float, sla_min: float, cost_per_hour: float) -> dict:
    """Pass the P90 recovery time to judge the SLA on the worst realistic case."""
    overrun = max(rto_min - sla_min, 0.0)
    return {
        "breach": rto_min > sla_min,
        "buffer_min": sla_min - rto_min,
        "downtime_cost": rto_min / 60 * cost_per_hour,
        "breach_cost": overrun / 60 * cost_per_hour,                    # minutes beyond the SLA
        "friction_cost": max(friction_min, 0.0) / 60 * cost_per_hour,  # part of downtime_cost, not extra
    }


def critical_path(tiers: pd.DataFrame, minutes: dict) -> tuple[dict, list]:
    """Finish time per tier when each tier starts after the tier it depends on.

    tiers: columns Tier, Depends_On ("" for none). minutes: Tier -> restore minutes.
    Returns (finish times, the chain of tiers that sets the total)."""
    deps = {r.Tier: (r.Depends_On or "").strip() for r in tiers.itertuples()}
    unknown = {d for d in deps.values() if d and d not in deps}
    if unknown:
        raise ValueError(f"Depends_On names unknown tiers: {sorted(unknown)}")
    finish, visiting = {}, set()

    def end(t):
        if t in finish:
            return finish[t]
        if t in visiting:
            raise ValueError(f"Dependency cycle through {t}")
        visiting.add(t)
        finish[t] = (end(deps[t]) if deps[t] else 0.0) + minutes[t]
        visiting.discard(t)
        return finish[t]

    for t in deps:
        end(t)
    chain = [max(finish, key=finish.get)]
    while deps[chain[-1]]:
        chain.append(deps[chain[-1]])
    return finish, chain[::-1]


def chaos(minutes: float, noise_pct: float, runs: int = 500, seed: int = 0) -> np.ndarray:
    """Monte Carlo of unstable throughput: speed each step is nominal × (1 + noise × N(0,1)), floored at 5%.

    Returns the finish time of each run in minutes."""
    if minutes <= 0:
        return np.zeros(runs)
    steps = 400                      # each step moves 1/400 of the data at nominal speed
    rng = np.random.default_rng(seed)
    speed = np.maximum(1 + noise_pct / 100 * rng.standard_normal((runs, steps)), 0.05)
    return (minutes / steps / speed).sum(axis=1)
