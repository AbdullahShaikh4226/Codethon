"""Synthetic VM-fleet telemetry: per-minute metrics, log/platform events and labelled failures.

No real monitoring data is available, so failures are simulated from documented mechanisms. Each cause
leaves its own precursor pattern (strength varies per incident), host failures give almost no warning,
and benign spikes (batch jobs, scanners, transient network errors) look like trouble but lead nowhere.
Every number here is an assumption for the demo, not a measurement.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gmcr.config import PLATFORMS, ROOT, features, load_data

SYNTH_DIR = ROOT / "data_synthetic"
METRICS = ["cpu_pct", "mem_pct", "disk_latency_ms", "disk_free_pct", "net_err_per_min"]
CAUSES = ["disk_exhaustion", "storage_io", "network", "host", "os_crash", "app", "security"]
CAUSE_LABEL = {"disk_exhaustion": "disk exhaustion", "storage_io": "storage I/O failure",
               "network": "network failure", "host": "host issue", "os_crash": "OS crash",
               "app": "application failure", "security": "suspected security incident"}
CAUSE_P = [0.15, 0.17, 0.15, 0.12, 0.15, 0.15, 0.11]
PRECURSOR_MIN = {"disk_exhaustion": (30, 120), "storage_io": (15, 60), "network": (10, 40), "host": (0, 8),
                 "os_crash": (20, 90), "app": (10, 30), "security": (20, 60)}
DOWNTIME_MIN = (20, 120)
CODES = {  # code: (source, message)
    "DISK_IO_ERR": ("os", "blk_update_request: I/O error, dev sda"),
    "FS_READONLY": ("os", "EXT4-fs error: remounting filesystem read-only"),
    "ENOSPC": ("os", "write failed: No space left on device"),
    "NET_TIMEOUT": ("os", "connection timed out to upstream"),
    "LINK_DOWN": ("os", "eth0: link is down"),
    "HOST_DEGRADED": ("platform", "host hardware degraded; VM may be affected"),
    "OOM_KILL": ("os", "Out of memory: Killed process"),
    "KERNEL_PANIC": ("os", "Kernel panic - not syncing"),
    "APP_5XX": ("app", "HTTP 500 Internal Server Error"),
    "APP_EXIT": ("app", "main service exited with status 1"),
    "AUTH_FAIL": ("auth", "Failed password for invalid user"),
    "SEC_ALERT": ("security", "suspicious process opening outbound connections"),
    "VM_STOPPED": ("platform", "VM stopped unexpectedly"),
}
BACKGROUND = {"NET_TIMEOUT": 0.01, "DISK_IO_ERR": 0.002, "AUTH_FAIL": 0.02, "APP_5XX": 0.02,
              "OOM_KILL": 0.0005, "HOST_DEGRADED": 0.0003}
TERMINAL = {  # events written at the failure minute: (code, probability, count)
    "disk_exhaustion": [("ENOSPC", 1.0, 3), ("APP_EXIT", 0.5, 1)],
    "storage_io": [("DISK_IO_ERR", 1.0, 3), ("FS_READONLY", 0.8, 1)],
    "network": [("LINK_DOWN", 0.9, 1), ("NET_TIMEOUT", 1.0, 3)],
    "host": [("HOST_DEGRADED", 0.6, 1)],
    "os_crash": [("KERNEL_PANIC", 0.8, 1), ("OOM_KILL", 0.6, 1)],
    "app": [("APP_EXIT", 1.0, 1), ("APP_5XX", 1.0, 5)],
    "security": [("SEC_ALERT", 0.7, 1), ("AUTH_FAIL", 0.6, 5)],
}


@dataclass
class Telemetry:
    fleet: pd.DataFrame       # vm_id, platform, tier, hist_failures, restore_config (JSON)
    metrics: pd.DataFrame     # vm_id, minute, METRICS..., health_ok, is_down
    events: pd.DataFrame      # vm_id, minute, source, code, message
    incidents: pd.DataFrame   # incident_id, vm_id, platform, cause, precursor_start, fail_minute, recover_minute, strength


def _vm(rng, vm_id, T, hist_failures, failures_per_day):
    base = {"cpu_pct": rng.uniform(10, 50), "mem_pct": rng.uniform(30, 65), "disk_latency_ms": rng.uniform(2, 10),
            "disk_free_pct": rng.uniform(30, 80), "net_err_per_min": rng.uniform(0, 0.3)}
    t = np.arange(T)
    m = {
        "cpu_pct": base["cpu_pct"] + 10 * np.sin(2 * np.pi * t / 1440) + rng.normal(0, 4, T),
        "mem_pct": base["mem_pct"] + rng.normal(0, 2, T),
        "disk_latency_ms": base["disk_latency_ms"] * rng.lognormal(0, 0.15, T),
        "disk_free_pct": base["disk_free_pct"] - 0.002 * t + rng.normal(0, 0.2, T),
        "net_err_per_min": rng.poisson(base["net_err_per_min"], T).astype(float),
    }
    health = (rng.random(T) > 0.001).astype(float)
    rates = {c: np.full(T, BACKGROUND.get(c, 0.0)) for c in CODES}
    extra_events = []

    # Benign episodes: look like trouble, lead nowhere.
    for _ in range(rng.poisson(1.5 * T / 1440)):
        s, L = rng.integers(0, T - 20), rng.integers(5, 20)
        kind = rng.integers(5)
        w = slice(s, s + L)
        if kind == 0:
            m["cpu_pct"][w] += 35
        elif kind == 1:
            m["disk_latency_ms"][w] *= 4
        elif kind == 2:
            m["net_err_per_min"][w] += 15
            rates["NET_TIMEOUT"][w] += 0.5
        elif kind == 3:
            m["mem_pct"][w] += 20
        else:
            rates["AUTH_FAIL"][w] += 1.5  # internet scanner

    # Failures, at least 240 minutes apart, never in the first 180 minutes.
    n = rng.poisson(failures_per_day * (T / 1440) * (1 + 0.5 * hist_failures))
    times = []
    for f in np.sort(rng.integers(180, T - 30, n)):
        if not times or f - times[-1] >= 240:
            times.append(int(f))
    is_down = np.zeros(T, bool)
    incidents = []
    for f in times:
        cause = CAUSES[rng.choice(len(CAUSES), p=CAUSE_P)]
        L = int(rng.integers(*PRECURSOR_MIN[cause])) if PRECURSOR_MIN[cause][1] else 0
        s = rng.uniform(0.4, 1.0)                      # how clearly this incident shows its cause
        p0 = f - L
        r = (np.arange(L) / max(L, 1)) ** 2 if L else np.zeros(0)
        w = slice(p0, f)
        if cause == "disk_exhaustion":
            m["disk_free_pct"][w] = m["disk_free_pct"][p0] * (1 - 0.97 * r)
            m["disk_latency_ms"][w] *= 1 + 1.5 * r * s
            rates["ENOSPC"][w] += 0.3 * r * s
        elif cause == "storage_io":
            m["disk_latency_ms"][w] *= 1 + 15 * r * s
            rates["DISK_IO_ERR"][w] += 0.6 * r * s
        elif cause == "network":
            m["net_err_per_min"][w] += rng.poisson(30 * r * s)
            rates["NET_TIMEOUT"][w] += 0.8 * r * s
            health[w] = np.where(rng.random(L) < 0.3 * r, 0, health[w])
        elif cause == "host":
            if L and rng.random() < 0.5:
                extra_events.append((p0, "HOST_DEGRADED"))
        elif cause == "os_crash":
            m["mem_pct"][w] += (98 - m["mem_pct"][w]) * r * s
            rates["OOM_KILL"][w] += 0.2 * r * s
        elif cause == "app":
            m["cpu_pct"][w] += rng.choice([-1, 1]) * 20 * r * s
            rates["APP_5XX"][w] += 1.0 * r * s
            health[w] = np.where(rng.random(L) < 0.2 * r, 0, health[w])
        elif cause == "security":
            m["cpu_pct"][w] += (95 - m["cpu_pct"][w]) * r * s
            m["net_err_per_min"][w] += rng.poisson(5 * r * s)
            rates["AUTH_FAIL"][w] += 2.0 * r * s
            rates["SEC_ALERT"][w] += 0.1 * r * s
        for code, prob, count in TERMINAL[cause]:
            if rng.random() < prob:
                extra_events += [(f, code)] * count
        extra_events.append((f, "VM_STOPPED"))
        D = int(rng.integers(*DOWNTIME_MIN))
        is_down[f:f + D] = True
        if cause == "disk_exhaustion":  # space is freed during recovery
            m["disk_free_pct"][f + D:] = base["disk_free_pct"] - 0.002 * t[f + D:]
        incidents.append({"vm_id": vm_id, "cause": cause, "precursor_start": p0, "fail_minute": f,
                          "recover_minute": min(f + D, T), "strength": round(s, 2)})

    m["cpu_pct"] = np.clip(m["cpu_pct"], 0, 100)
    m["mem_pct"] = np.clip(m["mem_pct"], 0, 100)
    m["disk_free_pct"] = np.clip(m["disk_free_pct"], 0, 100)
    metrics = pd.DataFrame({"vm_id": vm_id, "minute": t, **m, "health_ok": np.where(is_down, 0.0, health),
                            "is_down": is_down})
    metrics.loc[is_down, METRICS] = np.nan        # no telemetry while the VM is down

    rows = []
    for code, rate in rates.items():
        rate = np.where(is_down, 0, rate)
        counts = rng.poisson(rate)
        rows += [(int(i), code) for i in np.repeat(np.nonzero(counts)[0], counts[counts > 0])]
    rows += extra_events
    events = pd.DataFrame(rows, columns=["minute", "code"]).assign(vm_id=vm_id)
    return metrics, events, incidents


def generate(n_vms: int = 200, days: int = 2, failures_per_day: float = 1.0, seed: int = 7) -> Telemetry:
    rng = np.random.default_rng(seed)
    T = days * 1440
    configs = {p: load_data(p)[features(p)] for p in PLATFORMS}
    fleet, metrics, events, incidents = [], [], [], []
    for i in range(n_vms):
        platform = list(PLATFORMS)[i % len(PLATFORMS)]
        cfg = configs[platform].iloc[rng.integers(len(configs[platform]))].to_dict()  # a real config from the restore data
        hist = int(rng.poisson(1.0))
        vm_id = f"vm-{i:03d}"
        fleet.append({"vm_id": vm_id, "platform": platform,
                      "tier": rng.choice(["critical", "standard", "low"], p=[0.2, 0.5, 0.3]),
                      "hist_failures": hist, "restore_config": json.dumps(cfg, default=lambda x: x.item())})
        m, e, inc = _vm(rng, vm_id, T, hist, failures_per_day)
        metrics.append(m)
        events.append(e)
        incidents += [{**x, "platform": platform} for x in inc]
    events = pd.concat(events, ignore_index=True).sort_values(["vm_id", "minute"], ignore_index=True)
    events["source"] = events.code.map(lambda c: CODES[c][0])
    events["message"] = events.code.map(lambda c: CODES[c][1])
    incidents = pd.DataFrame(incidents)
    incidents.insert(0, "incident_id", [f"inc-{i:04d}" for i in range(len(incidents))])
    return Telemetry(pd.DataFrame(fleet), pd.concat(metrics, ignore_index=True),
                     events[["vm_id", "minute", "source", "code", "message"]], incidents)


def save(tel: Telemetry):
    SYNTH_DIR.mkdir(exist_ok=True)
    tel.fleet.to_csv(SYNTH_DIR / "fleet.csv", index=False)
    tel.incidents.to_csv(SYNTH_DIR / "incidents.csv", index=False)
    tel.metrics.to_parquet(SYNTH_DIR / "metrics.parquet", index=False)
    tel.events.to_parquet(SYNTH_DIR / "events.parquet", index=False)


def load(regenerate: bool = False) -> Telemetry:
    if regenerate or not (SYNTH_DIR / "metrics.parquet").exists():
        save(generate())
    return Telemetry(pd.read_csv(SYNTH_DIR / "fleet.csv"), pd.read_parquet(SYNTH_DIR / "metrics.parquet"),
                     pd.read_parquet(SYNTH_DIR / "events.parquet"), pd.read_csv(SYNTH_DIR / "incidents.csv"))
