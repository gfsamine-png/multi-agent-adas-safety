"""Shared alert pipeline: observations -> TTC -> risk -> validator/LDW -> CAN frame."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from adas.ecu.can_messages import FCW_ID, LDW_ID, CanFrame, encode_fcw, encode_ldw
from adas.safety.ldw import LaneDepartureWarner
from adas.safety.risk import assess_risk
from adas.safety.ttc import TTCTracker
from adas.safety.validator import AlertValidator, ValidatorConfig

# synthetic drift used only when simulate_drift=True (never road data)
DRIFT_START_S = 3.0
DRIFT_RAMP_S = 3.0
DRIFT_MAX = 0.45


@dataclass(frozen=True)
class AlertEvent:
    timestamp_s: float
    kind: str               # "collision" or "lane_departure"
    level: str
    reason: str
    track_id: int | None
    ttc_s: float | None
    offset: float | None
    frame: CanFrame


def iter_alert_events(
    jsonl_path: Path | str,
    alert_level: str = "moderate",
    simulate_drift: bool = False,
) -> Iterator[AlertEvent]:
    tracker = TTCTracker()
    fcw = AlertValidator(ValidatorConfig(alert_level=alert_level))
    ldw = LaneDepartureWarner()
    counters = {FCW_ID: 0, LDW_ID: 0}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obs = json.loads(line)
            t = obs["timestamp_s"]

            risks = {}
            for obj in obs["objects"]:
                tid = obj["track_id"]
                if tid is None:
                    continue
                tracker.update(tid, t, tuple(obj["bbox"]))
                risks[tid] = assess_risk(obj["lane_position"], tracker.estimate(tid))
            tracker.prune(t)

            for a in fcw.update(t, risks):
                frame = encode_fcw(counters[FCW_ID], a.level, a.track_id, a.ttc_s)
                counters[FCW_ID] += 1
                yield AlertEvent(t, "collision", a.level, a.reason,
                                 a.track_id, a.ttc_s, None, frame)

            lanes = obs.get("lanes") or {}
            offset = lanes.get("center_offset") if lanes.get("available") else None
            if simulate_drift and offset is not None:
                ramp = min(max((t - DRIFT_START_S) / DRIFT_RAMP_S, 0.0), 1.0)
                offset += ramp * DRIFT_MAX
            la = ldw.update(t, offset)
            if la is not None:
                frame = encode_ldw(counters[LDW_ID], la.side, la.offset)
                counters[LDW_ID] += 1
                yield AlertEvent(t, "lane_departure", "warning",
                                 f"drifting {la.side}, offset {la.offset:+.2f}",
                                 None, None, la.offset, frame)