import pytest

from adas.safety.ldw import LaneDepartureWarner


def simulate(offsets, dt=0.1):
    w = LaneDepartureWarner()
    alerts = []
    for i, off in enumerate(offsets):
        alert = w.update(i * dt, off)
        if alert is not None:
            alerts.append(alert)
    return alerts


def test_centered_car_never_warns():
    assert simulate([0.0] * 50) == []


def test_no_warning_before_confirm_time():
    assert simulate([-0.4] * 5) == []  # 0.0 .. 0.4 s


def test_warns_left_after_confirm_time():
    alerts = simulate([-0.4] * 10)
    assert len(alerts) == 1
    assert alerts[0].side == "left"
    assert alerts[0].timestamp_s == pytest.approx(0.5)


def test_warns_right_for_positive_offset():
    alerts = simulate([0.4] * 10)
    assert len(alerts) == 1
    assert alerts[0].side == "right"


def test_no_repeat_while_drifting():
    assert len(simulate([-0.4] * 100)) == 1


def test_short_return_to_center_does_not_create_a_second_warning():
    offsets = [-0.4] * 8 + [0.0] * 5 + [-0.4] * 10  # 0.5 s back, less than 1.0 s
    assert len(simulate(offsets)) == 1


def test_cooldown_blocks_a_quick_second_warning():
    offsets = [-0.4] * 6 + [0.0] * 12 + [-0.4] * 28
    alerts = simulate(offsets)
    assert len(alerts) == 2
    assert alerts[1].timestamp_s > 3.0


def test_lane_not_found_never_warns():
    assert simulate([None] * 50) == []


def test_non_increasing_timestamps_are_rejected():
    w = LaneDepartureWarner()
    w.update(1.0, 0.0)
    with pytest.raises(ValueError):
        w.update(1.0, 0.0)