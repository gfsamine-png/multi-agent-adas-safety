import cv2
import numpy as np

from adas.perception.lanes import LaneDetector


def _road_image() -> np.ndarray:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(img, (120, 470), (270, 300), (255, 255, 255), 4)   # left lane line
    cv2.line(img, (520, 470), (370, 300), (255, 255, 255), 4)   # right lane line
    return img


def _detector_for_synthetic_image() -> LaneDetector:
    # Parameters matching the 640x480 synthetic image, independent of the
    # defaults tuned for the real 1920x1080 video.
    return LaneDetector(
        roi_top=0.55, roi_bottom=1.0,
        roi_top_left=0.35, roi_top_right=0.65,
        roi_bottom_left=0.05, roi_bottom_right=0.95,
        split_x=0.5,
    )


def test_detects_both_lines_on_synthetic_road():
    est = _detector_for_synthetic_image().detect(_road_image())
    assert est.available
    assert est.left.x_bottom < est.right.x_bottom
    assert abs(est.center_offset) < 0.15  # symmetric lines: camera near center


def test_blank_image_reports_no_lane():
    est = LaneDetector().detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not est.available
    assert est.center_offset is None