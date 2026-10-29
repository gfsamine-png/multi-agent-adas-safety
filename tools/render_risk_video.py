from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from adas.perception.annotate import draw_lanes
from adas.perception.detector import YoloDetector
from adas.perception.lanes import LaneDetector
from adas.perception.models import lane_position
from adas.perception.video import get_video_info, read_frames
from adas.safety.risk import assess_risk
from adas.safety.ttc import TTCTracker

COLORS = {  # BGR
    "unknown": (160, 160, 160),   # gray
    "low": (0, 200, 0),           # green
    "moderate": (0, 165, 255),    # orange
    "high": (0, 0, 255),          # red
    "critical": (255, 0, 255),    # magenta
}
RANK = {"low": 1, "moderate": 2, "high": 3, "critical": 4}


def main() -> None:
    p = argparse.ArgumentParser(description="Render a video with risk levels.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/risk.mp4"))
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    info = get_video_info(args.video)
    detector = YoloDetector(track=True)
    lane_detector = LaneDetector()
    ttc = TTCTracker()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, info.fps,
                             (info.width, info.height))
    try:
        for idx, frame in read_frames(args.video, args.max_frames):
            t = idx / info.fps
            detections, _ = detector.detect(frame)
            lanes = lane_detector.detect(frame)

            out = frame.copy()
            draw_lanes(out, lanes)
            worst = "none"
            for d in detections:
                pos = lane_position(d, lanes)
                level, ttc_txt = "unknown", ""
                if d.track_id is not None:
                    ttc.update(d.track_id, t, d.bbox)
                    risk = assess_risk(pos, ttc.estimate(d.track_id))
                    level = risk.level
                    if pos == "in_lane" and risk.ttc_s is not None:
                        ttc_txt = f" {risk.ttc_s:.1f}s"
                if RANK.get(level, 0) > RANK.get(worst, 0):
                    worst = level

                color = COLORS[level]
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
                tag = f"#{d.track_id}" if d.track_id is not None else "#?"
                label = f"{tag} {d.class_name} {level.upper()}{ttc_txt}"
                cv2.putText(out, label, (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                            cv2.LINE_AA)
            ttc.prune(t)

            header = f"t={t:5.2f}s   max risk: {worst.upper()}"
            cv2.putText(out, header, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(out)
    finally:
        writer.release()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()