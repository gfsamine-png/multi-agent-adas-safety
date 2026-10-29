from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from adas.perception.detector import YoloDetector
from adas.perception.video import read_frames

BYTETRACK_YAML = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: {buffer}
match_thresh: {match}
fuse_score: True
"""

VEHICLES = {"car", "truck", "bus", "motorcycle"}


def run(video: Path, max_frames: int, name: str, conf: float, tracker: str) -> None:
    detector = YoloDetector(conf=conf, track=True, tracker=tracker)  # fresh tracker state
    frames_per_id: Counter[int] = Counter()
    center_ids: set[int] = set()
    for _, frame in read_frames(video, max_frames):
        h, w = frame.shape[:2]
        detections, _ = detector.detect(frame)
        for d in detections:
            if d.track_id is None or d.class_name not in VEHICLES:
                continue
            frames_per_id[d.track_id] += 1
            cx = (d.bbox[0] + d.bbox[2]) / 2
            if 0.42 * w < cx < 0.58 * w and d.bbox[3] > 0.45 * h:
                center_ids.add(d.track_id)
    short = sum(1 for n in frames_per_id.values() if n < 10)
    print(f"{name:<30} vehicle IDs: {len(frames_per_id):3d} | "
          f"short tracks (<10 frames): {short:3d} | "
          f"IDs in the center band: {len(center_ids)} {sorted(center_ids)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare tracker settings on one video.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--max-frames", type=int, default=300)
    args = p.parse_args()

    out_dir = Path("outputs/trackers")
    out_dir.mkdir(parents=True, exist_ok=True)
    byte_default = out_dir / "byte_default.yaml"
    byte_lenient = out_dir / "byte_match09_buf60.yaml"
    byte_default.write_text(BYTETRACK_YAML.format(buffer=30, match=0.8), encoding="utf-8")
    byte_lenient.write_text(BYTETRACK_YAML.format(buffer=60, match=0.9), encoding="utf-8")

    configs = [
        ("bytetrack baseline, conf 0.35", 0.35, str(byte_default)),
        ("bytetrack m0.9 buf60, conf 0.15", 0.15, str(byte_lenient)),
        ("botsort default, conf 0.35", 0.35, "botsort.yaml"),
        ("botsort default, conf 0.15", 0.15, "botsort.yaml"),
    ]
    for name, conf, tracker in configs:
        run(args.video, args.max_frames, name, conf, tracker)


if __name__ == "__main__":
    main()