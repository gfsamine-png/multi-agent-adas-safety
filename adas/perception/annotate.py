from __future__ import annotations

import cv2
import numpy as np

from adas.perception.detector import Detection


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of the frame with boxes and labels drawn."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def draw_overlay(frame: np.ndarray, text: str) -> None:
    """Draw a status line in the top-left corner (in place)."""
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)