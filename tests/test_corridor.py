from adas.safety.corridor import effective_lane_position

W = 1920


def test_known_positions_are_never_changed():
    assert effective_lane_position("in_lane", (0, 0, 100, 100), W) == "in_lane"
    assert effective_lane_position("left_of_lane", (900, 0, 1000, 100), W) == "left_of_lane"


def test_unknown_at_left_edge_becomes_left_of_lane():
    # the real car #8 from drive3
    assert effective_lane_position("unknown", (0, 1974, 181, 2204), W) == "left_of_lane"


def test_unknown_at_right_side_becomes_right_of_lane():
    assert effective_lane_position("unknown", (1500, 1000, 1700, 1200), W) == "right_of_lane"


def test_unknown_near_the_center_stays_unknown():
    assert effective_lane_position("unknown", (900, 1000, 1000, 1100), W) == "unknown"


def test_without_frame_width_nothing_changes():
    assert effective_lane_position("unknown", (0, 0, 100, 100), None) == "unknown"