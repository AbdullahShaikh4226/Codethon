"""C. Recovery ETA and actions: restore-time model (physics + best ML model) + cause playbook + priority."""
import json

import pandas as pd

from gmcr.config import PLATFORMS
from gmcr.predict import predict as restore_forecast
from gmcr.telemetry import CAUSE_LABEL

# Assumed playbook per cause. fix_min = hands-on minutes before or instead of a restore (not learned: the data
# has no recovery logs). restore = whether a full restore is needed, timed by the restore model.
PLAYBOOK = {
    "disk_exhaustion": {"investigate": "disk usage (logs, temp files)", "action": "extend the volume or free space, then restart services",
                        "restore": False, "fix_min": 20, "alt_target": False},
    "storage_io": {"investigate": "storage", "action": "restore the disk onto healthy storage",
                   "restore": True, "fix_min": 15, "alt_target": True},
    "network": {"investigate": "the network path (NIC, security rules, routes)", "action": "fail traffic over and reattach or replace the network interface",
                "restore": False, "fix_min": 25, "alt_target": False},
    "host": {"investigate": "the host", "action": "redeploy the VM onto a healthy host",
             "restore": True, "fix_min": 10, "alt_target": True},
    "os_crash": {"investigate": "the OS (memory use, kernel logs)", "action": "reboot; restore from the last good snapshot if it recurs",
                 "restore": False, "fix_min": 15, "alt_target": False},
    "app": {"investigate": "the application (recent deploys, error logs)", "action": "roll back the last deploy and restart the service",
            "restore": False, "fix_min": 20, "alt_target": False},
    "security": {"investigate": "the security incident", "action": "isolate the VM, preserve forensic evidence, rebuild from a clean snapshot",
                 "restore": True, "fix_min": 120, "alt_target": True},
}
TIER_WEIGHT = {"critical": 3, "standard": 2, "low": 1}


def eta(cause: str, platform: str, config: dict) -> dict:
    play = PLAYBOOK[cause]
    r = restore_forecast(platform, config) if play["restore"] else None
    extra = 0.02 * config["VM_Size_GB"] if cause == "security" else 0.0   # forensic imaging, assumed 0.02 min/GB
    p50 = play["fix_min"] + extra + (r["p50_min"] if r else 0.0)
    p90 = play["fix_min"] + extra + (r["p90_min"] if r else 0.0)
    return {"p50_min": p50, "p90_min": p90, "restore": r}


def plan(vm: pd.Series, risk: dict, diagnosis: dict | None, sla_min: float = 240) -> dict:
    """Recovery plan for one VM. diagnosis is None when the VM is up (no outage to explain yet)."""
    config = json.loads(vm.restore_config)
    platform = vm.platform
    if diagnosis is None:
        # Up but at risk: plan for a full restore so the team knows the worst case before it happens.
        r = restore_forecast(platform, config)
        p50, p90, top, second, conf = r["p50_min"], r["p90_min"], None, None, None
        steps = ["no outage yet: act on the warning signals and pre-stage a restore point"]
        alt = r["p90_min"] > sla_min
    else:
        ranked = diagnosis["ranked"]
        top, second, conf = ranked[0]["cause"], ranked[1]["cause"], ranked[0]["probability"]
        e = eta(top, platform, config)
        p50, p90 = e["p50_min"], e["p90_min"]
        play = PLAYBOOK[top]
        steps = [f"investigate {play['investigate']} first", play["action"]]
        alt = play["alt_target"] or conf < 0.5 or p90 > sla_min
        if conf < 0.5:
            steps.append(f"rule out {CAUSE_LABEL[second]}")
    if alt:
        steps.append("prepare an alternative recovery target")

    down = risk["down"]
    state = 1.0 if down else risk["probability"]
    score = TIER_WEIGHT[vm.tier] * state * (1 + min(p90 / sla_min, 3)) * (1.5 if top == "security" else 1.0)
    if down and (vm.tier == "critical" or top == "security"):
        sev = "SEV1"
    elif down:
        sev = "SEV2"
    elif risk["band"] != "Low":
        sev = "SEV3"
    else:
        sev = "SEV4"
    label = "Restore ETA" if down else "Restore ETA if it fails"
    spread = f" (P90 {p90:.0f})" if round(p90) > round(p50) else ""
    msg = f"{label}: {p50:.0f} minutes{spread}; " + "; ".join(steps) + "."
    return {"vm_id": vm.vm_id, "platform": PLATFORMS[platform]["name"], "tier": vm.tier, "severity": sev,
            "priority_score": round(score, 2), "eta_p50_min": p50, "eta_p90_min": p90, "sla_breach": p90 > sla_min,
            "likely_cause": top, "steps": steps, "message": msg}
