from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from adas.perception.annotate import draw_detections, draw_lanes, draw_overlay
from adas.perception.detector import YoloDetector
from adas.perception.lanes import LaneDetector
from adas.perception.models import FrameObservation
from adas.perception.video import get_video_info, read_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLO on a driving video.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", default=None, help="e.g. cpu, 0")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save", type=Path, default=None, help="output .mp4")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--track", action="store_true", help="object tracking")
    p.add_argument("--lanes", action="store_true", help="lane detection")
    p.add_argument("--json-out", type=Path, default=None,
                   help="write one JSON line per frame (.jsonl)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = get_video_info(args.video)
    size = f"{info.width}x{info.height}"
    print(f"Video: {size} @ {info.fps:.1f} fps, {info.frame_count} frames")

    detector = YoloDetector(args.model, args.conf, args.device,
                            track=args.track)
    lane_detector = LaneDetector() if args.lanes else None

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, info.fps,
                                 (info.width, info.height))

    json_file = None
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        json_file = open(args.json_out, "w", encoding="utf-8")

    times_ms: list[float] = []
    lanes_found = 0
    frames_seen = 0
    in_lane_objects = 0
    try:
        for idx, frame in read_frames(args.video, args.max_frames):
            frames_seen += 1
            detections, ms = detector.detect(frame)
            if idx > 0:  # frame 0 includes model warm-up
                times_ms.append(ms)

            lanes = lane_detector.detect(frame) if lane_detector else None
            if lanes is not None:
                lanes_found += int(lanes.available)

            obs = FrameObservation(
                frame_index=idx,
                timestamp_s=idx / info.fps,
                detections=tuple(detections),
                lanes=lanes,
                inference_ms=ms,
            )
            in_lane = sum(p == "in_lane" for p in obs.positions())
            in_lane_objects += in_lane
            if json_file:
                json_file.write(json.dumps(obs.to_dict()) + "\n")

            annotated = draw_detections(frame, detections)
            if lanes is not None:
                draw_lanes(annotated, lanes)
            status = (f"{ms:.0f} ms | {1000 / ms:.1f} FPS | "
                      f"{len(detections)} objects | {in_lane} in lane")
            draw_overlay(annotated, status)
            if writer:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("ADAS detection (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer:
            writer.release()
        if json_file:
            json_file.close()
        cv2.destroyAllWindows()

    if times_ms:
        t = np.array(times_ms)
        print(f"Frames timed: {len(t)}")
        print(f"Inference mean: {t.mean():.1f} ms")
        print(f"Inference median: {np.median(t):.1f} ms")
        print(f"Inference p95: {np.percentile(t, 95):.1f} ms")
        print(f"Throughput: {1000 / t.mean():.1f} FPS (model only)")
    if lane_detector is not None and frames_seen:
        pct = 100 * lanes_found / frames_seen
        print(f"Lanes found: {lanes_found}/{frames_seen} frames ({pct:.0f}%)")
        print(f"Object-in-lane observations: {in_lane_objects}")
    if args.json_out:
        print(f"Observations written to {args.json_out}")


if __name__ == "__main__":
    main()