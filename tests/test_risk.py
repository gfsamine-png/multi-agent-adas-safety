from adas.safety.risk import RiskConfig, assess_risk
from adas.safety.ttc import TTCEstimate


def _approaching(ttc_s: float) -> TTCEstimate:
    return TTCEstimate("approaching", ttc_s, ttc_s * 0.9, ttc_s * 1.1, 20)


def test_in_lane_levels_follow_ttc_thresholds():
    assert assess_risk("in_lane", _approaching(1.0)).level == "critical"
    assert assess_risk("in_lane", _approaching(2.5)).level == "high"
    assert assess_risk("in_lane", _approaching(5.0)).level == "moderate"
    assert assess_risk("in_lane", _approaching(12.0)).level == "low"


def test_threshold_boundary_is_inclusive():
    assert assess_risk("in_lane", _approaching(1.5)).level == "critical"
    assert assess_risk("in_lane", _approaching(3.0)).level == "high"
    assert assess_risk("in_lane", _approaching(6.0)).level == "moderate"


def test_passing_vehicle_outside_lane_is_low():
    # The false alarm seen on the real video: TTC 0.4 s for a bus at the side.
    assert assess_risk("left_of_lane", _approaching(0.4)).level == "low"
    assert assess_risk("right_of_lane", _approaching(0.4)).level == "low"


def test_unknown_lane_position_caps_at_moderate():
    assert assess_risk("unknown", _approaching(1.0)).level == "moderate"
    assert assess_risk("unknown", _approaching(12.0)).level == "low"


def test_not_approaching_is_low():
    est = TTCEstimate("not_approaching", None, None, None, 20)
    assert assess_risk("in_lane", est).level == "low"


def test_insufficient_data_is_unknown_even_in_lane():
    est = TTCEstimate("insufficient_data", None, None, None, 3)
    assert assess_risk("in_lane", est).level == "unknown"


def test_uncertain_ttc_never_exceeds_moderate():
    close = TTCEstimate("uncertain", 5.0, 0.8, None, 15)
    far = TTCEstimate("uncertain", 5.0, 10.0, None, 15)
    assert assess_risk("in_lane", close).level == "moderate"
    assert assess_risk("in_lane", far).level == "low"


def test_custom_thresholds_are_respected():
    cfg = RiskConfig(critical_ttc_s=0.5, high_ttc_s=1.0, moderate_ttc_s=2.0)
    assert assess_risk("in_lane", _approaching(1.2), cfg).level == "moderate"
    assert assess_risk("in_lane", _approaching(0.4), cfg).level == "critical"


def test_reason_is_always_given():
    assert assess_risk("in_lane", _approaching(2.0)).reason