from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from adas.safety.ttc import TTCTracker


def main() -> None:
    p = argparse.ArgumentParser(description="Replay observations and estimate TTC.")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--ttc-limit", type=float, default=6.0,
                   help="print approaching objects with TTC below this (seconds)")
    args = p.parse_args()

    tracker = TTCTracker()
    counts: Counter[str] = Counter()
    shown = 0
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
                est = tracker.estimate(tid)
                counts[est.status] += 1
                if (est.status == "approaching" and est.ttc_s is not None
                        and est.ttc_s < args.ttc_limit and shown < 25):
                    shown += 1
                    rng = (f"{est.ttc_min_s:.1f}-"
                           f"{'inf' if est.ttc_max_s is None else f'{est.ttc_max_s:.1f}'}")
                    print(f"t={t:6.2f}s  #{tid:<4} {obj['class_name']:<10} "
                          f"{obj['lane_position']:<14} TTC {est.ttc_s:4.1f} s "
                          f"(range {rng} s, {est.n_samples} samples)")
            tracker.prune(t)

    print("\nStatus counts (per object per frame):")
    for status, n in counts.most_common():
        print(f"  {status:<18} {n}")


if __name__ == "__main__":
    main()