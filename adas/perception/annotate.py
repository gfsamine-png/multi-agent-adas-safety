from __future__ import annotations

import cv2
import numpy as np

from adas.perception.detector import Detection
from adas.perception.lanes import LaneEstimate


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of the frame with boxes and labels drawn."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        prefix = f"#{det.track_id} " if det.track_id is not None else ""
        label = f"{prefix}{det.class_name} {det.confidence:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def draw_overlay(frame: np.ndarray, text: str) -> None:
    """Draw a status line in the top-left corner (in place)."""
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)


def draw_lanes(frame: np.ndarray, lanes: LaneEstimate) -> None:
    """Draw lane lines and lane status (in place)."""
    for line, color in ((lanes.left, (255, 0, 0)), (lanes.right, (0, 0, 255))):
        if line is not None:
            cv2.line(frame, (int(line.x_bottom), int(line.y_bottom)),
                     (int(line.x_top), int(line.y_top)), color, 4)
    if lanes.available and lanes.center_offset is not None:
        text = f"Lane: offset {lanes.center_offset:+.2f}"
    else:
        text = "Lane: not found"
    cv2.putText(frame, text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2, cv2.LINE_AA)