import json

from adas.pipeline import iter_alert_events


def write_obs(path, offsets, dt=0.1):
    with open(path, "w", encoding="utf-8") as f:
        for i, off in enumerate(offsets):
            lanes = {"available": off is not None, "center_offset": off}
            f.write(json.dumps({"timestamp_s": i * dt, "lanes": lanes, "objects": []}) + "\n")


def test_centered_drive_produces_no_events(tmp_path):
    p = tmp_path / "obs.jsonl"
    write_obs(p, [0.0] * 30)
    assert list(iter_alert_events(p)) == []


def test_drift_produces_one_lane_departure_event_with_can_frame(tmp_path):
    p = tmp_path / "obs.jsonl"
    write_obs(p, [-0.4] * 20)
    events = list(iter_alert_events(p))
    assert len(events) == 1
    assert events[0].kind == "lane_departure"
    assert events[0].frame.arbitration_id == 0x301


def test_simulated_drift_adds_a_departure_to_a_centered_drive(tmp_path):
    p = tmp_path / "obs.jsonl"
    write_obs(p, [0.0] * 100)  # 10 s
    assert list(iter_alert_events(p)) == []
    assert len(list(iter_alert_events(p, simulate_drift=True))) == 1