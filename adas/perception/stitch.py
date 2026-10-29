from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoids a circular import with detector.py
    from adas.perception.detector import Detection

BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TrackStitcher:
    """Turns the tracker's raw IDs into more stable IDs.

    Call update() once per frame, even when there are no detections,
    because the frame counter advances on every call.
    """

    def __init__(self, max_gap_frames: int = 30, min_iou: float = 0.3,
                 dup_iou: float = 0.5) -> None:
        self.max_gap_frames = max_gap_frames
        self.min_iou = min_iou
        self.dup_iou = dup_iou
        self._frame = -1
        self._next_id = 1
        self._map: dict[int, int] = {}                          # raw id -> stable id
        self._last: dict[int, tuple[int, BBox, str]] = {}       # stable id -> (frame, bbox, class)

    def _new_id(self) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    def _suppress_duplicates(self, dets: list[Detection]) -> list[Detection]:
        keep: list[Detection] = []
        for d in sorted(dets, key=lambda x: x.confidence, reverse=True):
            if any(k.class_name == d.class_name and iou(k.bbox, d.bbox) >= self.dup_iou
                   for k in keep):
                continue
            keep.append(d)
        kept = {id(k) for k in keep}
        return [d for d in dets if id(d) in kept]   # original order

    def update(self, detections: list[Detection]) -> list[Detection]:
        self._frame += 1
        dets = self._suppress_duplicates(detections)
        result: list[Detection | None] = [None] * len(dets)
        used: set[int] = set()
        pending: list[int] = []

        # Pass 1: raw IDs we already know keep their stable ID.
        for i, d in enumerate(dets):
            if d.track_id is None:
                result[i] = d
                continue
            sid = self._map.get(d.track_id)
            if sid is not None and sid not in used:
                used.add(sid)
                result[i] = replace(d, track_id=sid)
                self._last[sid] = (self._frame, d.bbox, d.class_name)
            else:
                pending.append(i)

        # Pass 2: new raw IDs may inherit a recently lost stable ID.
        for i in sorted(pending, key=lambda j: dets[j].confidence, reverse=True):
            d = dets[i]
            best, best_iou = None, self.min_iou
            for sid, (frame, bbox, cls) in self._last.items():
                if (sid in used or cls != d.class_name
                        or self._frame - frame > self.max_gap_frames):
                    continue
                v = iou(d.bbox, bbox)
                if v >= best_iou:
                    best, best_iou = sid, v
            sid = best if best is not None else self._new_id()
            self._map[d.track_id] = sid
            used.add(sid)
            result[i] = replace(d, track_id=sid)
            self._last[sid] = (self._frame, d.bbox, d.class_name)

        # Forget stable IDs that have been gone too long.
        self._last = {s: v for s, v in self._last.items()
                      if self._frame - v[0] <= self.max_gap_frames}
        return [r for r in result if r is not None]