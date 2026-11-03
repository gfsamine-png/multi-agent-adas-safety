from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from adas.perception.annotate import draw_detections, draw_lanes, draw_overlay
from adas.perception.detector import DEFAULT_EGO_ZONE, YoloDetector
from adas.perception.lanes import LaneDetector
from adas.perception.models import FrameObservation
from adas.perception.video import get_video_info, read_frames

# option name -> LaneDetector argument (None = keep the detector's default)
ROI_OPTIONS = {
    "roi_top": "--roi-top",
    "roi_bottom": "--roi-bottom",
    "roi_top_left": "--roi-top-left",
    "roi_top_right": "--roi-top-right",
    "roi_bottom_left": "--roi-bottom-left",
    "roi_bottom_right": "--roi-bottom-right",
    "split_x": "--split-x",
}


def limit_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink a frame wider than max_width (keeps the aspect ratio)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    new_h = round(h * max_width / w)
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


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
    p.add_argument("--max-width", type=int, default=1920,
                   help="shrink wider frames to this width (lane settings "
                        "were tuned at 1920 pixels)")
    # per-camera lane search area (fractions of the frame, see tools/show_roi.py)
    for dest, flag in ROI_OPTIONS.items():
        p.add_argument(flag, dest=dest, type=float, default=None)
    # area showing our own car (mirror, hood): detections there are ignored
    p.add_argument("--ego-zone", type=float, nargs=2, default=None,
                   metavar=("X_MAX", "Y_MIN"),
                   help="ignore boxes centered left of X_MAX*width and "
                        "below Y_MIN*height")
    p.add_argument("--no-ego-zone", action="store_true",
                   help="turn the exclusion zone off")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = get_video_info(args.video)
    print(f"Video: {info.width}x{info.height} @ {info.fps:.1f} fps, "
          f"{info.frame_count} frames")
    out_w = min(info.width, args.max_width)
    out_h = round(info.height * out_w / info.width)
    print(f"Processing size: {out_w}x{out_h}")

    if args.no_ego_zone:
        ego_zone = None
    elif args.ego_zone:
        ego_zone = tuple(args.ego_zone)
    else:
        ego_zone = DEFAULT_EGO_ZONE

    detector = YoloDetector(args.model, args.conf, args.device,
                            track=args.track, ego_zone=ego_zone)

    lane_detector = None
    if args.lanes:
        lane_kwargs = {k: getattr(args, k) for k in ROI_OPTIONS
                       if getattr(args, k) is not None}
        if lane_kwargs:
            print(f"Lane settings: {lane_kwargs}")
        lane_detector = LaneDetector(**lane_kwargs)

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, info.fps,
                                 (out_w, out_h))

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
            frame = limit_width(frame, args.max_width)
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
                row = obs.to_dict()
                row["frame"] = {"width": out_w, "height": out_h}
                json_file.write(json.dumps(row) + "\n")

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