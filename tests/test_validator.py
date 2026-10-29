import pytest

from adas.safety.risk import RiskAssessment
from adas.safety.validator import AlertValidator, ValidatorConfig


def simulate(levels, cfg=ValidatorConfig(), dt=0.1):
    """levels: list of risk levels, one per frame (object id 1). Returns alerts."""
    v = AlertValidator(cfg)
    alerts = []
    for i, level in enumerate(levels):
        a = RiskAssessment(level=level, reason="test", ttc_s=2.0)
        alerts += v.update(i * dt, {1: a})
    return alerts


def test_low_risk_never_alerts():
    assert simulate(["low"] * 50) == []


def test_no_alert_before_confirm_time():
    assert simulate(["high"] * 3) == []  # 0.0, 0.1, 0.2 s


def test_alert_after_confirm_time():
    alerts = simulate(["high"] * 11)
    assert len(alerts) == 1
    assert alerts[0].timestamp_s == pytest.approx(0.3)
    assert alerts[0].level == "high"


def test_no_repeat_while_alert_is_active():
    assert len(simulate(["high"] * 100)) == 1


def test_short_dip_does_not_create_a_second_alert():
    levels = ["high"] * 11 + ["low"] * 5 + ["high"] * 10  # 0.5 s dip < 1.0 s
    assert len(simulate(levels)) == 1


def test_cooldown_blocks_quick_second_alert():
    # alert at 0.3 s, cleared after 1.0 s of low, high again from 2.1 s
    levels = ["high"] * 6 + ["low"] * 15 + ["high"] * 20
    alerts = simulate(levels)
    assert len(alerts) == 2
    assert alerts[1].timestamp_s > 3.0  # not at 2.4 s: cooldown held it back


def test_escalation_alerts_immediately():
    levels = ["high"] * 6 + ["critical"] * 10
    alerts = simulate(levels)
    assert [a.level for a in alerts] == ["high", "critical"]


def test_non_increasing_timestamps_are_rejected():
    v = AlertValidator()
    a = RiskAssessment(level="low", reason="x", ttc_s=None)
    v.update(1.0, {1: a})
    with pytest.raises(ValueError):
        v.update(1.0, {1: a})


def test_unknown_level_is_rejected():
    v = AlertValidator()
    a = RiskAssessment(level="banana", reason="x", ttc_s=None)
    with pytest.raises(ValueError):
        v.update(0.0, {1: a})