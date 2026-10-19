"""Features from telemetry. Every window looks backwards only, so a prediction never sees the future."""
import numpy as np
import pandas as pd

from gmcr.telemetry import CODES, METRICS, Telemetry

HORIZON = 15  # minutes ahead the failure model predicts


def _event_matrix(events: pd.DataFrame, vm_id: str, T: int) -> pd.DataFrame:
    """Per-minute event counts for one VM: rows = minute 0..T-1, columns = event codes."""
    e = events[events.vm_id == vm_id]
    mat = pd.crosstab(e.minute, e.code).reindex(index=range(T), columns=list(CODES), fill_value=0)
    return mat.astype(float)


def vm_minute_features(metrics: pd.DataFrame, events: pd.DataFrame, vm_id: str, hist_failures: int,
                       fail_minutes=()) -> pd.DataFrame:
    """Features for every minute of one VM, from that minute and the minutes before it."""
    m = metrics[metrics.vm_id == vm_id].sort_values("minute").reset_index(drop=True)
    T = len(m)
    f = pd.DataFrame({"vm_id": vm_id, "minute": m.minute, "is_down": m.is_down})
    for c in METRICS:
        s = m[c]
        f[c] = s
        f[f"{c}_mean15"] = s.rolling(15, min_periods=1).mean()
        f[f"{c}_max15"] = s.rolling(15, min_periods=1).max()
        f[f"{c}_delta15"] = s - s.shift(15)
        f[f"{c}_ratio60"] = (s.rolling(5, min_periods=1).mean() + 1) / (s.rolling(60, min_periods=5).mean() + 1)
    slope = (m.disk_free_pct - m.disk_free_pct.shift(30)) / 30
    f["disk_minutes_to_full"] = np.clip(m.disk_free_pct / np.maximum(-slope, 1e-3), 0, 1000)
    f["health_fail15"] = (1 - m.health_ok).rolling(15, min_periods=1).sum()
    ev = _event_matrix(events, vm_id, T)
    for code in CODES:
        if code == "VM_STOPPED":   # the failure itself; never a precursor
            continue
        f[f"evt15_{code}"] = ev[code].rolling(15, min_periods=1).sum().to_numpy()
        f[f"evt60_{code}"] = ev[code].rolling(60, min_periods=1).sum().to_numpy()
    fails = np.zeros(T)
    fails[[x for x in fail_minutes if x < T]] = 1
    f["failures_so_far"] = hist_failures + np.cumsum(fails) - fails   # failures before this minute
    return f


def failure_dataset(tel: Telemetry) -> pd.DataFrame:
    """One row per VM-minute while the VM is up; label = a failure starts within the next HORIZON minutes."""
    frames = []
    for vm in tel.fleet.itertuples():
        fails = sorted(tel.incidents.loc[tel.incidents.vm_id == vm.vm_id, "fail_minute"])
        f = vm_minute_features(tel.metrics, tel.events, vm.vm_id, vm.hist_failures, fails)
        nxt = np.full(len(f), np.inf)
        for x in reversed(fails):
            nxt[: x + 1] = np.where(f.minute.to_numpy()[: x + 1] < x, x, nxt[: x + 1])
        until = nxt - f.minute.to_numpy()
        f["label"] = ((until >= 1) & (until <= HORIZON)).astype(int)
        f["minutes_to_failure"] = until
        frames.append(f[~f.is_down])
    return pd.concat(frames, ignore_index=True)


# ---------- incident window (root cause) ----------

def window_features(metrics: pd.DataFrame, events: pd.DataFrame, vm_id: str, end_minute: int,
                    window: int = 60, baseline: int = 180) -> dict:
    """Summary of the hour before end_minute (inclusive) versus the VM's normal level before that."""
    m = metrics[metrics.vm_id == vm_id]
    win = m[m.minute.between(end_minute - window, end_minute)]
    base = m[m.minute.between(end_minute - window - baseline, end_minute - window - 1) & ~m.is_down]
    e = events[(events.vm_id == vm_id) & events.minute.between(end_minute - window, end_minute)]
    out = {}
    for c in METRICS:
        w, b = win[c], base[c]
        out[f"{c}_max"] = w.max()
        out[f"{c}_min"] = w.min()
        out[f"{c}_mean"] = w.mean()
        out[f"{c}_vs_normal"] = (w.mean() + 1) / (b.mean() + 1) if len(b.dropna()) else 1.0
        out[f"{c}_peak_vs_normal"] = (w.max() + 1) / (b.mean() + 1) if len(b.dropna()) else 1.0
    out["health_fail"] = float((1 - win.health_ok).sum())
    counts = e.code.value_counts()
    for code in CODES:
        if code != "VM_STOPPED":
            out[f"evt60_{code}"] = float(counts.get(code, 0))
    return out


def incident_dataset(tel: Telemetry) -> pd.DataFrame:
    rows = [{"incident_id": i.incident_id, "vm_id": i.vm_id, "cause": i.cause,
             **window_features(tel.metrics, tel.events, i.vm_id, i.fail_minute)} for i in tel.incidents.itertuples()]
    return pd.DataFrame(rows)
