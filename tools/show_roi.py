"""Draw a lane-search trapezoid on one frame so it can be tuned by eye."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--t", type=float, default=2.0, help="time in seconds")
    p.add_argument("--out", type=Path, default=Path("outputs/roi_check.png"))
    p.add_argument("--top", type=float, default=0.60)
    p.add_argument("--bottom", type=float, default=0.88)
    p.add_argument("--top-left", type=float, default=0.42)
    p.add_argument("--top-right", type=float, default=0.53)
    p.add_argument("--bottom-left", type=float, default=0.25)
    p.add_argument("--bottom-right", type=float, default=0.72)
    a = p.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, a.t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read a frame from the video")

    h, w = frame.shape[:2]
    poly = np.array([
        [a.bottom_left * w, a.bottom * h],
        [a.top_left * w, a.top * h],
        [a.top_right * w, a.top * h],
        [a.bottom_right * w, a.bottom * h],
    ], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], (0, 255, 255))
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.polylines(frame, [poly], True, (0, 255, 255), 3)
    for frac in (0.5, 0.6, 0.7, 0.8, 0.9):  # height guide lines
        y = int(frac * h)
        cv2.line(frame, (0, y), (40, y), (255, 255, 255), 2)
        cv2.putText(frame, f"{frac:.1f}", (45, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    args_txt = (f"--top {a.top} --bottom {a.bottom} --top-left {a.top_left} "
                f"--top-right {a.top_right} --bottom-left {a.bottom_left} "
                f"--bottom-right {a.bottom_right}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), frame)
    print(f"Frame {w}x{h} saved to {a.out}")
    print(args_txt)


if __name__ == "__main__":
    main()