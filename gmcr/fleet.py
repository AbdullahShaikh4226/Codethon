"""Fleet replay for the frontend: failure risk for every held-out VM at every minute, triage and incident detail.

Only held-out VMs (never seen in training) are shown, so every number on screen is an honest test result."""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from gmcr import failure, pipeline, rootcause, telemetry
from gmcr.config import MODEL_DIR, PLATFORMS
from gmcr.explain import CODE_NAMES
from gmcr.features import HORIZON, vm_minute_features
from gmcr.recovery import plan
from gmcr.telemetry import CAUSE_LABEL, METRICS


SEV_RANK = {"SEV1": 0, "SEV2": 1, "SEV3": 2, "SEV4": 3}


class Fleet:
    def __init__(self):
        self.tel = telemetry.load()
        self.risk_model, self.cause_model = pipeline.load_models()
        self.vms = sorted(json.loads((MODEL_DIR / "ops_card.json").read_text())["test_vms"])
        self.info = self.tel.fleet.set_index("vm_id")
        self.incidents = self.tel.incidents[self.tel.incidents.vm_id.isin(self.vms)].sort_values("fail_minute", ignore_index=True)
        self.minutes = int(self.tel.metrics.minute.max()) + 1
        frames = []
        for vm_id in self.vms:
            fails = sorted(self.tel.incidents.loc[self.tel.incidents.vm_id == vm_id, "fail_minute"])
            f = vm_minute_features(self.tel.metrics, self.tel.events, vm_id, int(self.info.loc[vm_id, "hist_failures"]), fails)
            p = self.risk_model["model"].predict_proba(f[self.risk_model["features"]])[:, 1]
            frames.append(pd.DataFrame({"vm_id": vm_id, "minute": f.minute, "down": f.is_down.to_numpy(),
                                        "p": np.where(f.is_down, np.nan, p)}))
        self.risk = pd.concat(frames, ignore_index=True)

    # ---------- fleet over time ----------

    def overview(self, step: int = 10) -> list[dict]:
        r = self.risk.assign(bucket=self.risk.minute // step * step)
        state = r.assign(down=r.down.astype(int), high=((r.p >= 0.5) & ~r.down).astype(int),
                         elevated=((r.p >= 0.2) & (r.p < 0.5) & ~r.down).astype(int))
        per_min = state.groupby(["bucket", "minute"])[["down", "high", "elevated"]].sum()
        # Busiest minute in each bucket, so a short spike is never averaged away.
        return per_min.groupby("bucket").max().reset_index().rename(columns={"bucket": "minute"}).to_dict("records")

    def _vm_row(self, vm_id: str) -> pd.Series:
        vm = self.info.loc[vm_id].copy()
        vm["vm_id"] = vm_id
        return vm

    def _status(self, vm_id: str, minute: int) -> dict:
        row = self.risk[(self.risk.vm_id == vm_id) & (self.risk.minute == minute)].iloc[0]
        if row.down:
            return {"down": True, "probability": 1.0, "band": "Down"}
        return {"down": False, "probability": float(row.p), "band": failure.band(float(row.p))}

    def _down_since(self, vm_id: str, minute: int) -> int:
        r = self.risk[(self.risk.vm_id == vm_id) & (self.risk.minute <= minute)]
        up = r[~r.down]
        return int(up.minute.max()) + 1 if len(up) else minute

    def triage(self, minute: int, sla_min: float = 240) -> list[dict]:
        if not 0 <= minute < self.minutes:
            raise ValueError(f"minute must be between 0 and {self.minutes - 1}")
        at = self.risk[self.risk.minute == minute]
        rows = []
        for r in at[at.down | (at.p >= 0.2)].itertuples():
            status = self._status(r.vm_id, minute)
            diagnosis = rootcause.diagnose(self.cause_model, self.tel, r.vm_id, self._down_since(r.vm_id, minute)) \
                if status["down"] else None
            p = plan(self._vm_row(r.vm_id), status, diagnosis, sla_min)
            rows.append({"vm_id": r.vm_id, "platform": p["platform"], "tier": p["tier"], "severity": p["severity"],
                         "priority": p["priority_score"], "status": status["band"], "probability": status["probability"],
                         "likely_cause": CAUSE_LABEL.get(p["likely_cause"]) if p["likely_cause"] else None,
                         "eta_p50_min": p["eta_p50_min"], "eta_p90_min": p["eta_p90_min"], "next_step": p["steps"][0],
                         "incident_id": self._incident_at(r.vm_id, minute)})
        return sorted(rows, key=lambda x: (SEV_RANK[x["severity"]], -x["priority"]))

    def _incident_at(self, vm_id: str, minute: int) -> str | None:
        """The incident a VM is in or heading into at this minute, if any."""
        i = self.incidents[(self.incidents.vm_id == vm_id) & (self.incidents.fail_minute - HORIZON <= minute)
                           & (self.incidents.recover_minute >= minute)]
        return i.incident_id.iloc[0] if len(i) else None

    # ---------- incidents ----------

    @lru_cache(maxsize=None)
    def _diagnosis(self, incident_id: str) -> dict:
        i = self.incidents.set_index("incident_id").loc[incident_id]
        return rootcause.diagnose(self.cause_model, self.tel, i.vm_id, int(i.fail_minute))

    def _warning(self, vm_id: str, fail_minute: int) -> dict:
        w = self.risk[(self.risk.vm_id == vm_id) & self.risk.minute.between(fail_minute - HORIZON, fail_minute - 1)]
        hit = w[w.p >= 0.2]
        return {"warned": bool(len(hit)), "lead_min": int(fail_minute - hit.minute.min()) if len(hit) else None,
                "peak_probability": float(w.p.max()) if len(w) else None}

    def incident_list(self) -> list[dict]:
        out = []
        for i in self.incidents.itertuples():
            d = self._diagnosis(i.incident_id)
            out.append({"incident_id": i.incident_id, "vm_id": i.vm_id, "platform": PLATFORMS[i.platform]["name"],
                        "tier": self.info.loc[i.vm_id, "tier"], "fail_minute": int(i.fail_minute),
                        "true_cause": CAUSE_LABEL[i.cause], "predicted_cause": d["ranked"][0]["label"],
                        "confidence": d["ranked"][0]["probability"], **self._warning(i.vm_id, int(i.fail_minute))})
        return out

    def incident_detail(self, incident_id: str, before: int = 120, after: int = 30) -> dict:
        idx = self.incidents.set_index("incident_id")
        if incident_id not in idx.index:
            raise KeyError(f"Unknown incident {incident_id}")
        i = idx.loc[incident_id]
        f, vm_id = int(i.fail_minute), i.vm_id
        lo, hi = max(0, f - before), min(self.minutes - 1, f + after)
        m = self.tel.metrics[(self.tel.metrics.vm_id == vm_id) & self.tel.metrics.minute.between(lo, hi)]
        r = self.risk[(self.risk.vm_id == vm_id) & self.risk.minute.between(lo, hi)].set_index("minute").p
        series = m[["minute", *METRICS, "health_ok", "is_down"]].assign(risk=m.minute.map(r).to_numpy())
        ev = self.tel.events[(self.tel.events.vm_id == vm_id) & self.tel.events.minute.between(lo, hi)
                             & (self.tel.events.code != "VM_STOPPED")]
        warning = self._warning(vm_id, f)
        signal_minute = f - (warning["lead_min"] or 1)
        diagnosis = self._diagnosis(incident_id)
        recovery = plan(self._vm_row(vm_id), {"down": True, "probability": 1.0, "band": "Down"}, diagnosis)
        return {
            "incident_id": incident_id, "vm_id": vm_id, "platform": PLATFORMS[i.platform]["name"],
            "tier": self.info.loc[vm_id, "tier"], "fail_minute": f, "recover_minute": int(i.recover_minute),
            "true_cause": CAUSE_LABEL[i.cause], "signal_strength": float(i.strength),
            "series": series.to_dict("records"),
            "events": [{"minute": int(e.minute), "code": e.code, "label": CODE_NAMES.get(e.code, e.code), "source": e.source,
                        "message": e.message} for e in ev.itertuples()],
            "warning": {**warning, "message": failure.assess(self.risk_model, self.tel, vm_id, signal_minute)["message"]},
            "diagnosis": diagnosis, "plan": recovery,
        }
