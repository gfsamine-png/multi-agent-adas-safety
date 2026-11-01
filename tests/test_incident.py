import sqlite3

import pytest

from adas.storage.incident_db import IncidentDb


def test_store_and_read_back_alerts_in_time_order():
    db = IncidentDb()
    run = db.start_run("clip.mp4", note="test")
    db.add_alert(run, 7.29, "collision", "moderate", "in our lane", track_id=21, ttc_s=4.5)
    db.add_alert(run, 5.88, "lane_departure", "warning", "drifting right", offset=0.36)
    rows = db.alerts(run)
    assert [r["timestamp_s"] for r in rows] == [5.88, 7.29]
    assert rows[1]["track_id"] == 21
    assert rows[1]["ttc_s"] == pytest.approx(4.5)


def test_alerts_of_different_runs_are_separate():
    db = IncidentDb()
    a = db.start_run("a.mp4")
    b = db.start_run("b.mp4", simulated=True)
    db.add_alert(a, 1.0, "collision", "high", "x")
    assert len(db.alerts(a)) == 1
    assert db.alerts(b) == []
    assert [r["simulated"] for r in db.runs()] == [1, 0]  # newest first


def test_unknown_kind_is_rejected():
    db = IncidentDb()
    run = db.start_run("a.mp4")
    with pytest.raises(ValueError):
        db.add_alert(run, 1.0, "banana", "high", "x")


def test_alert_for_missing_run_is_rejected():
    db = IncidentDb()
    with pytest.raises(sqlite3.IntegrityError):
        db.add_alert(999, 1.0, "collision", "high", "x")


def test_text_with_quotes_is_stored_safely():
    db = IncidentDb()
    run = db.start_run("a.mp4")
    nasty = "x'); DROP TABLE alerts; --"
    db.add_alert(run, 1.0, "collision", "high", nasty)
    assert db.alerts(run)[0]["reason"] == nasty