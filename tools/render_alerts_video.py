"""Render a video showing per-object risk and validator alerts.

Boxes are colored by risk. When the validator raises an alert, a banner
appears at the top for 2 seconds and the alerted object gets a thick box.
All drawing sizes scale with the frame, so tall or large videos stay readable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adas.safety.corridor import effective_lane_position  # noqa: E402
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


def px(value: float, s: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value * s)))


def put_label(img, text, x, y, color, s):
    font = 0.6 * s
    thick = px(2, s)
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
    y = max(y, h + px(4, s))
    cv2.rectangle(img, (x, y - h - px(6, s)), (x + w + px(6, s), y + base - px(2, s)),
                  color, -1)
    cv2.putText(img, text, (x + px(3, s), y - px(4, s)), cv2.FONT_HERSHEY_SIMPLEX,
                font, (255, 255, 255), thick, cv2.LINE_AA)


def limit_width(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    return cv2.resize(frame, (max_width, round(h * max_width / w)),
                      interpolation=cv2.INTER_AREA)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=Path("data/videos/drive1.mp4"))
    p.add_argument("--obs", type=Path, default=Path("outputs/observations.jsonl"))
    p.add_argument("--out", type=Path, default=Path("outputs/alerts.mp4"))
    p.add_argument("--alert-level", default="moderate")
    p.add_argument("--max-width", type=int, default=1920,
                   help="must match the value used in run_detection")
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = min(src_w, args.max_width)
    h = round(src_h * w / src_w)
    s = max(w, h) / 1920.0  # drawing scale: 1.0 for a 1920x1080 frame
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = TTCTracker()
    validator = AlertValidator(ValidatorConfig(alert_level=args.alert_level))
    banner_until = -1.0
    banner_text = ""
    alerted_id = None
    n = 0

    try:
        with open(args.obs, encoding="utf-8") as f:
            for line in f:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = limit_width(frame, args.max_width)
                obs = json.loads(line)
                t = obs["timestamp_s"]

                lanes = obs.get("lanes") or {}
                for side, color in (("left", (255, 120, 0)), ("right", (0, 0, 255))):
                    ln = lanes.get(side)
                    if ln:
                        cv2.line(frame, (int(ln["x_bottom"]), int(ln["y_bottom"])),
                                 (int(ln["x_top"]), int(ln["y_top"])), color, px(3, s))

                risks = {}
                for obj in obs["objects"]:
                    tid = obj["track_id"]
                    if tid is None:
                        continue
                    tracker.update(tid, t, tuple(obj["bbox"]))
                    lane_pos = effective_lane_position(
                        obj["lane_position"], obj["bbox"], w
                    )
                    risks[tid] = assess_risk(lane_pos, tracker.estimate(tid))
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
                    thick = px(6 if (banner_on and tid == alerted_id) else 2, s)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[level], thick)
                    ttc_txt = f" {risk.ttc_s:.1f}s" if risk and risk.ttc_s else ""
                    put_label(frame, f"#{tid} {obj['class_name']} {level}{ttc_txt}",
                              x1, y1, COLORS[level], s)

                top = max((r.level for r in risks.values()), key=ORDER.index,
                          default="unknown")
                bar_h = px(44, s)
                cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)
                cv2.putText(frame, f"t={t:5.2f}s   max risk: {top.upper()}",
                            (px(12, s), px(30, s)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9 * s, COLORS[top], px(2, s), cv2.LINE_AA)
                if banner_on:
                    cv2.rectangle(frame, (0, bar_h), (w, bar_h + px(56, s)),
                                  (0, 0, 200), -1)
                    cv2.putText(frame, banner_text, (px(12, s), bar_h + px(42, s)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * s, (255, 255, 255),
                                px(3, s), cv2.LINE_AA)

                writer.write(frame)
                n += 1
    finally:
        cap.release()
        writer.release()  # always finish the file, even after an error
    print(f"Wrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()