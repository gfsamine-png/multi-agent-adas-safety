from adas.dashboard.summary import summarize


def test_empty_run():
    s = summarize([])
    assert s == {"total": 0, "collisions": 0, "lane_departures": 0,
                 "worst_collision_level": "none"}


def test_counts_and_worst_collision_level():
    alerts = [
        {"kind": "collision", "level": "moderate"},
        {"kind": "collision", "level": "high"},
        {"kind": "lane_departure", "level": "warning"},
    ]
    s = summarize(alerts)
    assert s["total"] == 3
    assert s["collisions"] == 2
    assert s["lane_departures"] == 1
    assert s["worst_collision_level"] == "high"


def test_lane_departure_does_not_affect_worst_collision_level():
    s = summarize([{"kind": "lane_departure", "level": "warning"}])
    assert s["worst_collision_level"] == "none"