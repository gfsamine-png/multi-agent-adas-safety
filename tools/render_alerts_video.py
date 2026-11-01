"""Render a video showing per-object risk and validator alerts.

Boxes are colored by risk. When the validator raises an alert, a banner
appears at the top for 2 seconds and the alerted object gets a thick box.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adas.safety.risk import assess_risk  # noqa: E402
from adas.safety.ttc import TTCTracker  # noqa: E402
from adas.safety.validator import AlertValidator, ValidatorConfig  # noqa: E402

COLORS = {  # BGR
    "unknown": (150, 150, 150),
    "low": (80, 180, 80),
    "moderate": (0, 165, 255),
    "high": (0, 0, 255),
    "critical": (200, 0, 200),
}
ORDER = ["unknown", "low", "moderate", "high", "critical"]
BANNER_S = 2.0


def put_label(img, text, x, y, color):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y, h + 4)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=Path("data/videos/drive1.mp4"))
    p.add_argument("--obs", type=Path, default=Path("outputs/observations.jsonl"))
    p.add_argument("--out", type=Path, default=Path("outputs/alerts.mp4"))
    p.add_argument("--alert-level", default="moderate")
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = TTCTracker()
    validator = AlertValidator(ValidatorConfig(alert_level=args.alert_level))
    banner_until = -1.0
    banner_text = ""
    alerted_id = None
    n = 0

    with open(args.obs, encoding="utf-8") as f:
        for line in f:
            ok, frame = cap.read()
            if not ok:
                break
            obs = json.loads(line)
            t = obs["timestamp_s"]

            lanes = obs.get("lanes") or {}
            for side, color in (("left", (255, 120, 0)), ("right", (0, 0, 255))):
                ln = lanes.get(side)
                if ln:
                    cv2.line(frame, (int(ln["x_bottom"]), int(ln["y_bottom"])),
                             (int(ln["x_top"]), int(ln["y_top"])), color, 3)

            risks = {}
            for obj in obs["objects"]:
                tid = obj["track_id"]
                if tid is None:
                    continue
                tracker.update(tid, t, tuple(obj["bbox"]))
                risks[tid] = assess_risk(obj["lane_position"], tracker.estimate(tid))
            tracker.prune(t)

            for a in validator.update(t, risks):
                ttc = f"{a.ttc_s:.1f}s" if a.ttc_s is not None else "n/a"
                banner_text = f"ALERT {a.level.upper()}  #{a.track_id}  TTC {ttc}"
                banner_until = t + BANNER_S
                alerted_id = a.track_id
                print(f"alert raised at t={t:.2f}s: {banner_text}")

            banner_on = t <= banner_until
            for obj in obs["objects"]:
                tid = obj["track_id"]
                x1, y1, x2, y2 = (int(v) for v in obj["bbox"])
                risk = risks.get(tid)
                level = risk.level if risk else "unknown"
                thick = 6 if (banner_on and tid == alerted_id) else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[level], thick)
                ttc_txt = f" {risk.ttc_s:.1f}s" if risk and risk.ttc_s else ""
                put_label(frame, f"#{tid} {obj['class_name']} {level}{ttc_txt}",
                          x1, y1, COLORS[level])

            top = max((r.level for r in risks.values()), key=ORDER.index, default="unknown")
            cv2.rectangle(frame, (0, 0), (w, 44), (30, 30, 30), -1)
            cv2.putText(frame, f"t={t:5.2f}s   max risk: {top.upper()}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[top], 2, cv2.LINE_AA)
            if banner_on:
                cv2.rectangle(frame, (0, 44), (w, 100), (0, 0, 200), -1)
                cv2.putText(frame, banner_text, (12, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

            writer.write(frame)
            n += 1

    cap.release()
    writer.release()
    print(f"Wrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()