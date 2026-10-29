from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from adas.perception.models import LanePosition
from adas.safety.ttc import TTCEstimate

RiskLevel = Literal["unknown", "low", "moderate", "high", "critical"]


@dataclass(frozen=True)
class RiskConfig:
    """Experimental thresholds in seconds (not automotive safety standards)."""
    critical_ttc_s: float = 1.5
    high_ttc_s: float = 3.0
    moderate_ttc_s: float = 6.0


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    reason: str
    ttc_s: float | None


def _level_from_ttc(ttc_s: float, cfg: RiskConfig) -> RiskLevel:
    if ttc_s <= cfg.critical_ttc_s:
        return "critical"
    if ttc_s <= cfg.high_ttc_s:
        return "high"
    if ttc_s <= cfg.moderate_ttc_s:
        return "moderate"
    return "low"


def assess_risk(
    lane_pos: LanePosition,
    ttc: TTCEstimate,
    cfg: RiskConfig = RiskConfig(),
) -> RiskAssessment:
    """Combine lane position and TTC into a risk level, conservatively."""
    if ttc.status == "insufficient_data":
        return RiskAssessment("unknown", "not enough tracking history", None)

    if lane_pos in ("left_of_lane", "right_of_lane"):
        return RiskAssessment("low", f"object is outside our lane ({lane_pos})",
                              ttc.ttc_s)

    if ttc.status == "not_approaching":
        return RiskAssessment("low", "object is not getting closer", None)

    if ttc.status == "uncertain":
        worst = ttc.ttc_min_s
        if worst is not None and worst <= cfg.moderate_ttc_s:
            return RiskAssessment(
                "moderate", "TTC uncertain, worst case is within threshold", None)
        return RiskAssessment("low", "TTC uncertain, worst case is not close", None)

    # status == "approaching"
    if ttc.ttc_s is None:
        return RiskAssessment("unknown", "approaching but TTC unavailable", None)
    level = _level_from_ttc(ttc.ttc_s, cfg)
    if lane_pos == "unknown":
        if level in ("high", "critical"):
            return RiskAssessment(
                "moderate", "lane position unknown, capped at moderate", ttc.ttc_s)
        return RiskAssessment(level, "lane position unknown", ttc.ttc_s)
    return RiskAssessment(level, f"in our lane, TTC {ttc.ttc_s:.1f} s", ttc.ttc_s)