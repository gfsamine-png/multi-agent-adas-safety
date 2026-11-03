"""Show what the lane detector sees on one frame: edges and line segments."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--t", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("outputs/lane_debug.png"))
    p.add_argument("--max-width", type=int, default=1920)
    p.add_argument("--top", type=float, default=0.54)
    p.add_argument("--bottom", type=float, default=0.64)
    p.add_argument("--top-left", type=float, default=0.48)
    p.add_argument("--top-right", type=float, default=0.56)
    p.add_argument("--bottom-left", type=float, default=0.36)
    p.add_argument("--bottom-right", type=float, default=0.65)
    p.add_argument("--canny-low", type=int, default=50)
    p.add_argument("--canny-high", type=int, default=150)
    p.add_argument("--hough-threshold", type=int, default=25)
    p.add_argument("--min-length", type=int, default=30)
    p.add_argument("--max-gap", type=int, default=150)
    p.add_argument("--min-slope", type=float, default=0.5)
    a = p.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, a.t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read a frame from the video")
    h, w = frame.shape[:2]
    if w > a.max_width:
        frame = cv2.resize(frame, (a.max_width, round(h * a.max_width / w)),
                           interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, a.canny_low, a.canny_high)
    poly = np.array([
        [a.bottom_left * w, a.bottom * h], [a.top_left * w, a.top * h],
        [a.top_right * w, a.top * h], [a.bottom_right * w, a.bottom * h],
    ], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [poly], 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, a.hough_threshold,
                            minLineLength=a.min_length, maxLineGap=a.max_gap)
    segs = [] if lines is None else lines.reshape(-1, 4)

    view = frame.copy()
    cv2.polylines(view, [poly], True, (0, 255, 255), 2)
    neg = pos = flat = 0
    for x1, y1, x2, y2 in segs:
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 99.0
        if abs(slope) < a.min_slope:
            color, flat = (150, 150, 150), flat + 1      # too flat: rejected
        elif slope < 0:
            color, neg = (255, 120, 0), neg + 1          # left candidates (blue)
        else:
            color, pos = (0, 0, 255), pos + 1            # right candidates (red)
        cv2.line(view, (x1, y1), (x2, y2), color, 3)

    # crop the road band and show the frame above the masked edges
    y0, y1_ = int(0.45 * h), int(0.70 * h)
    top_img = view[y0:y1_]
    edge_img = cv2.cvtColor(masked[y0:y1_], cv2.COLOR_GRAY2BGR)
    out = np.vstack([top_img, edge_img])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), out)

    print(f"Frame {w}x{h}, saved {a.out}")
    print(f"Segments found: {len(segs)}  "
          f"left-type (blue): {neg}  right-type (red): {pos}  too flat (gray): {flat}")


if __name__ == "__main__":
    main()