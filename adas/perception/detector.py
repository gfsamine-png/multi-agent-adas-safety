from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from adas.perception.stitch import TrackStitcher

# COCO classes relevant for road scenes
RELEVANT_CLASSES = frozenset(
    {"person", "bicycle", "car", "motorcycle", "bus", "truck",
     "traffic light", "stop sign"}
)

# Part of the image that shows our own car (mirror, hood), as fractions of
# the frame: (x_max, y_min). A box whose center is left of x_max * width AND
# below y_min * height is ignored. Measured on drive1.mp4 (1920x1080):
# the mirror center is at about 22% width, 84% height.
DEFAULT_EGO_ZONE = (0.35, 0.72)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    track_id: int | None = None  # None = no tracking, or track not confirmed yet


def in_ego_zone(
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    ego_zone: tuple[float, float] = DEFAULT_EGO_ZONE,
) -> bool:
    """True if the box center lies in the part of the image showing our own car."""
    x_max_frac, y_min_frac = ego_zone
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return cx <= x_max_frac * frame_width and cy >= y_min_frac * frame_height


class YoloDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.35,
        device: str | None = None,
        classes: frozenset[str] = RELEVANT_CLASSES,
        track: bool = False,
        tracker: str = "bytetrack.yaml",
        iou: float = 0.5,             # NMS overlap limit (Ultralytics default: 0.7)
        agnostic_nms: bool = True,    # suppress overlapping boxes across classes
        stitch: bool = True,          # stabilise track IDs (only used when track=True)
        ego_zone: tuple[float, float] | None = DEFAULT_EGO_ZONE,  # None = off
    ) -> None:
        self.model = YOLO(model_name)  # downloads weights on first use
        self.conf = conf
        self.device = device
        self.classes = classes
        self.track = track
        self.tracker = tracker
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.ego_zone = ego_zone
        self.stitcher = TrackStitcher() if (track and stitch) else None

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Return (detections, inference_time_ms) for one BGR frame."""
        start = time.perf_counter()
        if self.track:
            result = self.model.track(
                frame, conf=self.conf, iou=self.iou,
                agnostic_nms=self.agnostic_nms, device=self.device,
                persist=True, tracker=self.tracker, verbose=False,
            )[0]
        else:
            result = self.model.predict(
                frame, conf=self.conf, iou=self.iou,
                agnostic_nms=self.agnostic_nms, device=self.device,
                verbose=False,
            )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        frame_h, frame_w = frame.shape[:2]
        detections: list[Detection] = []
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            if name not in self.classes:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            bbox = (x1, y1, x2, y2)
            if self.ego_zone is not None and in_ego_zone(
                bbox, frame_w, frame_h, self.ego_zone
            ):
                continue  # part of our own car, not a road object
            track_id = int(box.id[0]) if box.id is not None else None
            detections.append(
                Detection(name, float(box.conf[0]), bbox, track_id)
            )
        if self.stitcher is not None:
            detections = self.stitcher.update(detections)
        return detections, elapsed_ms