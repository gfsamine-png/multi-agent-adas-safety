from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from adas.perception.detector import Detection
from adas.perception.lanes import LaneEstimate, LaneLine

LanePosition = Literal["left_of_lane", "in_lane", "right_of_lane", "unknown"]


def _x_at(line: LaneLine, y: float) -> float:
    """x of the lane line at height y (straight-line extrapolation)."""
    t = (y - line.y_bottom) / (line.y_top - line.y_bottom)
    return line.x_bottom + t * (line.x_top - line.x_bottom)


def lane_position(
    det: Detection, lanes: LaneEstimate | None, y_margin: float = 60.0
) -> LanePosition:
    """Where is the object relative to our lane? 'unknown' if we cannot tell."""
    if lanes is None or not lanes.available:
        return "unknown"
    x1, _, x2, y2 = det.bbox
    y = y2                       # bottom of the box ~ where the object meets the road
    if y < lanes.left.y_top - y_margin:
        return "unknown"         # too far beyond the lane model's range
    x = (x1 + x2) / 2
    left_x = _x_at(lanes.left, y)
    right_x = _x_at(lanes.right, y)
    if right_x <= left_x:
        return "unknown"         # inconsistent lane geometry
    if x < left_x:
        return "left_of_lane"
    if x > right_x:
        return "right_of_lane"
    return "in_lane"


@dataclass(frozen=True)
class FrameObservation:
    frame_index: int
    timestamp_s: float
    detections: tuple[Detection, ...]
    lanes: LaneEstimate | None   # None = lane detection was not run
    inference_ms: float

    def positions(self) -> tuple[LanePosition, ...]:
        return tuple(lane_position(d, self.lanes) for d in self.detections)

    def to_dict(self) -> dict:
        """Plain, JSON-serializable form (for logs, agents, SQLite later)."""
        lanes = None
        if self.lanes is not None:
            lanes = {
                "available": self.lanes.available,
                "center_offset": self.lanes.center_offset,
                "left": asdict(self.lanes.left) if self.lanes.left else None,
                "right": asdict(self.lanes.right) if self.lanes.right else None,
            }
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 3),
            "inference_ms": round(self.inference_ms, 1),
            "lanes": lanes,
            "objects": [
                {
                    "track_id": d.track_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "bbox": [round(v, 1) for v in d.bbox],
                    "lane_position": pos,
                }
                for d, pos in zip(self.detections, self.positions())
            ],
        }