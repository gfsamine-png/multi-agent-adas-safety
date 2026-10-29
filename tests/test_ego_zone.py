from adas.perception.detector import in_ego_zone

W, H = 1920, 1080


def test_mirror_box_is_in_ego_zone():
    # the mirror measured on drive1.mp4
    assert in_ego_zone((336, 780, 499, 1034), W, H)


def test_pedestrian_ahead_is_not_in_ego_zone():
    # pedestrian measured on drive1.mp4
    assert not in_ego_zone((542, 598, 610, 747), W, H)


def test_car_in_the_middle_of_the_road_is_not_in_ego_zone():
    assert not in_ego_zone((900, 570, 950, 610), W, H)


def test_object_low_on_the_right_is_not_in_ego_zone():
    assert not in_ego_zone((1400, 800, 1600, 1000), W, H)