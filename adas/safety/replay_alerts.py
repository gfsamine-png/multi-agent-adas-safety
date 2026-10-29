from __future__ import annotations

import argparse
import json
from pathlib import Path

from adas.safety.risk import assess_risk
from adas.safety.ttc import TTCTracker
from adas.safety.validator import AlertValidator, ValidatorConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Replay observations through risk + validator.")
    p.add_argument("jsonl", type=Path)
    args = p.parse_args()

    # alert at "moderate" (demo) and at "high" (real setting), side by side
    levels = ("moderate", "high")
    validators = {lv: AlertValidator(ValidatorConfig(alert_level=lv)) for lv in levels}
    alerts = {lv: [] for lv in levels}
    raw_changes = 0  # how often a per-frame level changed (what we would have printed)
    last_level: dict[int, str] = {}

    tracker = TTCTracker()
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            obs = json.loads(line)
            t = obs["timestamp_s"]
            frame_risks = {}
            for obj in obs["objects"]:
                tid = obj["track_id"]
                if tid is None:
                    continue
                tracker.update(tid, t, tuple(obj["bbox"]))
                risk = assess_risk(obj["lane_position"], tracker.estimate(tid))
                frame_risks[tid] = risk
                if risk.level != last_level.get(tid):
                    raw_changes += 1
                last_level[tid] = risk.level
            tracker.prune(t)
            for lv in levels:
                alerts[lv] += validators[lv].update(t, frame_risks)

    print(f"Per-frame level changes (before validator): {raw_changes}")
    for lv in levels:
        print(f"\nAlert level '{lv}': {len(alerts[lv])} alert(s)")
        for a in alerts[lv]:
            ttc = f"{a.ttc_s:.1f}s" if a.ttc_s is not None else "n/a"
            print(f"  t={a.timestamp_s:6.2f}s  #{a.track_id:<4} {a.level.upper():<9} "
                  f"ttc={ttc}  {a.reason}")


if __name__ == "__main__":
    main()