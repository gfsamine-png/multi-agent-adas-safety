from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneLine:
    """One lane boundary as two points in pixel coordinates."""
    x_bottom: float
    y_bottom: float
    x_top: float
    y_top: float


@dataclass(frozen=True)
class LaneEstimate:
    left: LaneLine | None
    right: LaneLine | None
    # (image center - lane center) / lane width at the bottom of the road area.
    # 0 = centered, -0.5 = on left line, +0.5 = on right line. None if unknown.
    center_offset: float | None

    @property
    def available(self) -> bool:
        return self.left is not None and self.right is not None


class LaneDetector:
    """Classic lane detection. Default parameters are tuned on drive1.mp4
    (1920x1080, forward camera) and must be re-tuned for other cameras."""

    def __init__(
        self,
        roi_top: float = 0.60,          # ROI top edge, fraction of height
        roi_bottom: float = 0.88,       # ROI bottom edge (cuts off the dashboard)
        roi_top_left: float = 0.42,     # x fractions of the ROI's top edge
        roi_top_right: float = 0.53,
        roi_bottom_left: float = 0.25,  # x fractions of the ROI's bottom edge
        roi_bottom_right: float = 0.72,
        split_x: float = 0.47,          # left/right divider, fraction of width
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 25,
        min_line_length: int = 30,
        max_line_gap: int = 150,        # large gap: bridges dashed markings
        min_abs_slope: float = 0.5,     # ignore near-horizontal segments (dy/dx)
    ) -> None:
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.roi_x = (roi_bottom_left, roi_top_left, roi_top_right, roi_bottom_right)
        self.split_x = split_x
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_abs_slope = min_abs_slope

    def _roi_polygon(self, w: int, h: int) -> np.ndarray:
        bl, tl, tr, br = self.roi_x
        y_top = int(self.roi_top * h)
        y_bottom = int(self.roi_bottom * h)
        return np.array(
            [[(int(bl * w), y_bottom), (int(tl * w), y_top),
              (int(tr * w), y_top), (int(br * w), y_bottom)]],
            dtype=np.int32,
        )

    def detect(self, frame: np.ndarray) -> LaneEstimate:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self._roi_polygon(w, h), 255)
        edges = cv2.bitwise_and(edges, mask)

        segments = cv2.HoughLinesP(
            edges, 1, np.pi / 180, self.hough_threshold,
            minLineLength=self.min_line_length, maxLineGap=self.max_line_gap,
        )
        if segments is None:
            return LaneEstimate(None, None, None)
        segments = segments.reshape(-1, 4)  # works for (N, 1, 4) and (N, 4)

        split = self.split_x * w
        left_pts: list[tuple[float, float]] = []   # (x, y) points
        right_pts: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in segments.astype(float):
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < self.min_abs_slope:
                continue
            # y grows downward: left lane has negative slope, right has positive
            if slope < 0 and max(x1, x2) < split:
                left_pts += [(x1, y1), (x2, y2)]
            elif slope > 0 and min(x1, x2) > split:
                right_pts += [(x1, y1), (x2, y2)]

        y_top = self.roi_top * h
        y_bottom = self.roi_bottom * h
        left = self._fit(left_pts, y_bottom, y_top)
        right = self._fit(right_pts, y_bottom, y_top)

        offset = None
        if left is not None and right is not None:
            lane_width = right.x_bottom - left.x_bottom
            if lane_width > 0:
                lane_center = (left.x_bottom + right.x_bottom) / 2
                offset = (w / 2 - lane_center) / lane_width
        return LaneEstimate(left, right, offset)

    @staticmethod
    def _fit(points: list[tuple[float, float]], y_bottom: float,
             y_top: float) -> LaneLine | None:
        if len(points) < 2:
            return None
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        if np.ptp(ys) < 1:  # all points on one row: cannot fit x = a*y + b
            return None
        a, b = np.polyfit(ys, xs, 1)  # x = a*y + b
        return LaneLine(a * y_bottom + b, y_bottom, a * y_top + b, y_top)