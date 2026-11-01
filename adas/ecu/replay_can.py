from __future__ import annotations

import argparse
import json
from pathlib import Path

from adas.ecu.can_messages import FCW_ID, LDW_ID, CanFrame, encode_fcw, encode_ldw
from adas.ecu.virtual_ecu import DashboardEcu
from adas.safety.ldw import LaneDepartureWarner
from adas.safety.risk import assess_risk
from adas.safety.ttc import TTCTracker
from adas.safety.validator import AlertValidator, ValidatorConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Send alerts through the virtual CAN/ECU.")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--alert-level", default="moderate")
    p.add_argument("--simulate-drift", action="store_true",
                   help="SIMULATION: add a synthetic drift to the lane offset")
    args = p.parse_args()

    tracker = TTCTracker()
    fcw = AlertValidator(ValidatorConfig(alert_level=args.alert_level))
    ldw = LaneDepartureWarner()
    ecu = DashboardEcu()
    counters = {FCW_ID: 0, LDW_ID: 0}
    sent = 0

    def emit(t: float, frame: CanFrame) -> None:
        nonlocal sent
        sent += 1
        print(f"t={t:6.2f}s  TX id=0x{frame.arbitration_id:03X}  data={frame.data.hex(' ')}")
        print(f"           ECU: {ecu.receive(frame)}")

    if args.simulate_drift:
        print("*** SIMULATION: lane offset has a synthetic drift added (not road data) ***")

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
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
                emit(t, encode_fcw(counters[FCW_ID], a.level, a.track_id, a.ttc_s))
                counters[FCW_ID] += 1

            lanes = obs.get("lanes") or {}
            offset = lanes.get("center_offset") if lanes.get("available") else None
            if args.simulate_drift and offset is not None:
                offset += min(max((t - 3.0) / 3.0, 0.0), 1.0) * 0.45  # ramps 0 -> 0.45
            alert = ldw.update(t, offset)
            if alert is not None:
                emit(t, encode_ldw(counters[LDW_ID], alert.side, alert.offset))
                counters[LDW_ID] += 1

    print(f"\nFrames sent: {sent}, rejected by ECU: {ecu.rejected}")


if __name__ == "__main__":
    main()