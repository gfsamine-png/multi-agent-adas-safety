from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from adas.perception.annotate import draw_detections, draw_lanes, draw_overlay
from adas.perception.detector import YoloDetector
from adas.perception.lanes import LaneDetector
from adas.perception.video import get_video_info, read_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLO on a recorded driving video.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", default=None, help="e.g. cpu, 0 (first GPU)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save", type=Path, default=None, help="output .mp4 path")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--track", action="store_true", help="enable object tracking")
    p.add_argument("--lanes", action="store_true", help="enable lane detection")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = get_video_info(args.video)
    print(f"Video: {info.width}x{info.height} @ {info.fps:.1f} fps, "
          f"{info.frame_count} frames")

    detector = YoloDetector(args.model, args.conf, args.device, track=args.track)
    lane_detector = LaneDetector() if args.lanes else None

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (info.width, info.height))

    times_ms: list[float] = []
    lanes_found = 0
    frames_seen = 0
    try:
        for idx, frame in read_frames(args.video, args.max_frames):
            frames_seen += 1
            detections, ms = detector.detect(frame)
            if idx > 0:  # frame 0 includes model warm-up, so exclude it
                times_ms.append(ms)

            annotated = draw_detections(frame, detections)
            if lane_detector is not None:
                lanes = lane_detector.detect(frame)
                lanes_found += int(lanes.available)
                draw_lanes(annotated, lanes)
            draw_overlay(annotated, f"{ms:.0f} ms | {1000 / ms:.1f} FPS | "
                                    f"{len(detections)} objects")
            if writer:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("ADAS detection (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if times_ms:
        t = np.array(times_ms)
        print(f"Frames timed: {len(t)}")
        print(f"Inference: mean {t.mean():.1f} ms | median {np.median(t):.1f} ms | "
              f"p95 {np.percentile(t, 95):.1f} ms")
        print(f"Throughput: {1000 / t.mean():.1f} FPS (model only)")
    if lane_detector is not None and frames_seen:
        print(f"Lanes found (both sides): {lanes_found}/{frames_seen} frames "
              f"({100 * lanes_found / frames_seen:.0f}%)")


if __name__ == "__main__":
    main()