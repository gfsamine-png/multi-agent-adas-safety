import cv2
import numpy as np
import pytest

from adas.perception.annotate import draw_detections
from adas.perception.detector import Detection
from adas.perception.video import get_video_info, read_frames


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                             10.0, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_video_info(tiny_video):
    info = get_video_info(tiny_video)
    assert (info.width, info.height) == (64, 48)


def test_read_frames_respects_max_frames(tiny_video):
    assert len(list(read_frames(tiny_video, max_frames=3))) == 3


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_video_info(tmp_path / "nope.mp4")


def test_draw_does_not_mutate_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(frame, [Detection("car", 0.9, (10, 10, 50, 50))])
    assert frame.sum() == 0 and out.sum() > 0