from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from adas.safety.risk import assess_risk
from adas.safety.ttc import TTCTracker

ALERT_LEVELS = ("moderate", "high", "critical")


def main() -> None:
    p = argparse.ArgumentParser(description="Replay observations and assess risk.")
    p.add_argument("jsonl", type=Path)
    args = p.parse_args()

    tracker = TTCTracker()
    last_level: dict[int, str] = {}
    counts: Counter[str] = Counter()
    printed = 0
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            obs = json.loads(line)
            t = obs["timestamp_s"]
            for obj in obs["objects"]:
                tid = obj["track_id"]
                if tid is None:
                    counts["no_track_id"] += 1
                    continue
                tracker.update(tid, t, tuple(obj["bbox"]))
                risk = assess_risk(obj["lane_position"], tracker.estimate(tid))
                counts[risk.level] += 1
                changed = risk.level != last_level.get(tid)
                if changed and risk.level in ALERT_LEVELS and printed < 40:
                    printed += 1
                    print(f"t={t:6.2f}s  #{tid:<4} {obj['class_name']:<8} "
                          f"{risk.level.upper():<9} {risk.reason}")
                last_level[tid] = risk.level
            tracker.prune(t)

    print("\nRisk level counts (per object per frame):")
    for level in ("critical", "high", "moderate", "low", "unknown", "no_track_id"):
        if counts[level]:
            print(f"  {level:<12} {counts[level]}")


if __name__ == "__main__":
    main()