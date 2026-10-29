from adas.perception.detector import Detection
from adas.perception.stitch import TrackStitcher, iou


def _car(raw_id, box=(100, 100, 200, 200), conf=0.8, cls="car"):
    return Detection(cls, conf, box, raw_id)


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_raw_id_change_keeps_stable_id():
    s = TrackStitcher()
    first = s.update([_car(5)])[0].track_id
    second = s.update([_car(9, box=(102, 101, 203, 202))])[0].track_id
    assert second == first


def test_same_raw_id_stays_stable():
    s = TrackStitcher()
    ids = [s.update([_car(5)])[0].track_id for _ in range(3)]
    assert len(set(ids)) == 1


def test_different_class_is_not_stitched():
    s = TrackStitcher()
    first = s.update([_car(5)])[0].track_id
    second = s.update([_car(9, cls="truck")])[0].track_id
    assert second != first


def test_far_apart_box_is_not_stitched():
    s = TrackStitcher()
    first = s.update([_car(5)])[0].track_id
    second = s.update([_car(9, box=(600, 600, 700, 700))])[0].track_id
    assert second != first


def test_too_long_gap_is_not_stitched():
    s = TrackStitcher(max_gap_frames=5)
    first = s.update([_car(5)])[0].track_id
    for _ in range(10):
        s.update([])
    assert s.update([_car(9)])[0].track_id != first


def test_gap_within_limit_is_stitched():
    s = TrackStitcher(max_gap_frames=30)
    first = s.update([_car(5)])[0].track_id
    for _ in range(10):
        s.update([])
    assert s.update([_car(9)])[0].track_id == first


def test_duplicate_boxes_keep_the_more_confident_one():
    s = TrackStitcher()
    out = s.update([_car(1, conf=0.4), _car(2, box=(105, 105, 205, 205), conf=0.7)])
    assert len(out) == 1 and out[0].confidence == 0.7


def test_detection_without_track_id_passes_through():
    s = TrackStitcher()
    out = s.update([Detection("car", 0.9, (0, 0, 10, 10), None)])
    assert out[0].track_id is None