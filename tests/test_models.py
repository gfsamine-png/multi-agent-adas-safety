import json

from adas.perception.detector import Detection
from adas.perception.lanes import LaneEstimate, LaneLine
from adas.perception.models import FrameObservation, lane_position


def _lanes() -> LaneEstimate:
    return LaneEstimate(LaneLine(200, 400, 300, 200),
                        LaneLine(600, 400, 500, 200), 0.0)


def _det(x1, y1, x2, y2) -> Detection:
    return Detection("car", 0.9, (x1, y1, x2, y2), track_id=1)


def test_in_lane():
    assert lane_position(_det(350, 300, 450, 380), _lanes()) == "in_lane"


def test_left_and_right_of_lane():
    assert lane_position(_det(50, 300, 150, 380), _lanes()) == "left_of_lane"
    assert lane_position(_det(650, 300, 750, 380), _lanes()) == "right_of_lane"


def test_unknown_without_lanes():
    det = _det(350, 300, 450, 380)
    assert lane_position(det, None) == "unknown"
    assert lane_position(det, LaneEstimate(None, None, None)) == "unknown"


def test_unknown_when_far_above_lane_model():
    assert lane_position(_det(350, 20, 450, 50), _lanes()) == "unknown"


def test_observation_to_dict_is_json_serializable():
    obs = FrameObservation(3, 0.1, (_det(350, 300, 450, 380),), _lanes(), 27.4)
    data = json.loads(json.dumps(obs.to_dict()))
    assert data["objects"][0]["lane_position"] == "in_lane"
    assert data["lanes"]["available"] is True