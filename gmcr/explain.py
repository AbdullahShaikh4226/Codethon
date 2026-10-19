"""Turn model features and their contributions into plain-language evidence."""
import numpy as np

METRIC_NAMES = {"cpu_pct": ("CPU", "%"), "mem_pct": ("memory", "%"), "disk_latency_ms": ("disk latency", " ms"),
                "disk_free_pct": ("free disk", "%"), "net_err_per_min": ("network errors", "/min")}
# A raw level only counts as evidence when it is clearly abnormal; so does a 15-minute change.
ABNORMAL_LEVEL = {"cpu_pct": lambda v: v >= 80, "mem_pct": lambda v: v >= 85, "disk_latency_ms": lambda v: v >= 30,
                  "disk_free_pct": lambda v: v <= 15, "net_err_per_min": lambda v: v >= 5}
ABNORMAL_CHANGE = {"cpu_pct": 20, "mem_pct": 15, "disk_latency_ms": 10, "disk_free_pct": 5, "net_err_per_min": 5}
CODE_NAMES = {"DISK_IO_ERR": "disk I/O error", "FS_READONLY": "read-only filesystem event",
              "ENOSPC": "disk-full error", "NET_TIMEOUT": "network timeout", "LINK_DOWN": "network link-down event",
              "HOST_DEGRADED": "host-degraded platform event", "OOM_KILL": "out-of-memory kill",
              "KERNEL_PANIC": "kernel panic", "APP_5XX": "HTTP 5xx error", "APP_EXIT": "application crash",
              "AUTH_FAIL": "failed login", "SEC_ALERT": "security alert"}
# These codes also appear as background noise on healthy VMs, so a handful proves nothing.
MIN_COUNT = {"APP_5XX": 3, "AUTH_FAIL": 5, "NET_TIMEOUT": 3}


def _n(count: float, noun: str) -> str:
    return f"{int(count)} {noun}{'' if int(count) == 1 else 's'}"


def subject(feature: str) -> str:
    """What a feature is about (a metric or an event code), so evidence never repeats the same thing."""
    for prefix in ("evt15_", "evt60_"):
        if feature.startswith(prefix):
            return feature[len(prefix):]
    return next((m for m in METRIC_NAMES if feature.startswith(m)), feature)


def describe(feature: str, v: float) -> str | None:
    """Readable evidence, or None when the value is normal and would not convince anyone."""
    if v is None or np.isnan(v):
        return None
    for prefix, when in (("evt15_", "in the last 15 minutes"), ("evt60_", "in the last hour")):
        if feature.startswith(prefix):
            code = feature[len(prefix):]
            return f"{_n(v, CODE_NAMES[code])} {when}" if v >= MIN_COUNT.get(code, 1) else None
    if feature == "disk_minutes_to_full":
        return f"disk full in about {v:.0f} min at the current rate" if v < 300 else None
    if feature in ("health_fail15", "health_fail"):
        return _n(v, "failed health check") if v > 0 else None
    if feature == "failures_so_far":
        return f"{_n(v, 'earlier failure')} on this VM" if v > 0 else None
    metric = subject(feature)
    if metric not in METRIC_NAMES:
        return None
    name, unit = METRIC_NAMES[metric]
    suffix = feature[len(metric):]
    if suffix in ("_ratio60", "_vs_normal", "_peak_vs_normal"):
        if abs(v - 1) < 0.25:
            return None
        return f"{name} {'peaked at' if 'peak' in suffix else 'at'} {v:.1f}× normal"
    if suffix == "_delta15":
        return f"{name} changed by {v:+.0f}{unit} in 15 min" if abs(v) >= ABNORMAL_CHANGE[metric] else None
    if not ABNORMAL_LEVEL[metric](v):
        return None
    return {"": f"{name} at {v:.0f}{unit}", "_mean15": f"{name} averaging {v:.0f}{unit} over 15 min",
            "_max15": f"{name} peaked at {v:.0f}{unit}", "_max": f"{name} peaked at {v:.0f}{unit}",
            "_min": f"{name} fell to {v:.0f}{unit}", "_mean": f"{name} averaged {v:.0f}{unit}"}.get(suffix)


def evidence(names, values, contributions, k: int = 3) -> list[str]:
    """Top-k features pushing toward the prediction that read as real evidence, one per metric or event."""
    out, seen = [], set()
    for i in np.argsort(contributions)[::-1]:
        if contributions[i] <= 0 or len(out) == k:
            break
        text = describe(names[i], float(values[i]))
        if text and subject(names[i]) not in seen:
            out.append(text)
            seen.add(subject(names[i]))
    return out
