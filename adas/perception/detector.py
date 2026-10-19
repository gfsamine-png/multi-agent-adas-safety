from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# COCO classes relevant for road scenes
RELEVANT_CLASSES = frozenset(
    {"person", "bicycle", "car", "motorcycle", "bus", "truck",
     "traffic light", "stop sign"}
)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    track_id: int | None = None  # None = no tracking, or track not confirmed yet


class YoloDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.35,
        device: str | None = None,
        classes: frozenset[str] = RELEVANT_CLASSES,
        track: bool = False,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model = YOLO(model_name)  # downloads weights on first use
        self.conf = conf
        self.device = device
        self.classes = classes
        self.track = track
        self.tracker = tracker

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Return (detections, inference_time_ms) for one BGR frame."""
        start = time.perf_counter()
        if self.track:
            result = self.model.track(
                frame, conf=self.conf, device=self.device,
                persist=True, tracker=self.tracker, verbose=False,
            )[0]
        else:
            result = self.model.predict(
                frame, conf=self.conf, device=self.device, verbose=False
            )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections: list[Detection] = []
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            if name not in self.classes:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            track_id = int(box.id[0]) if box.id is not None else None
            detections.append(
                Detection(name, float(box.conf[0]), (x1, y1, x2, y2), track_id)
            )
        return detections, elapsed_ms