from __future__ import annotations

from adas.safety.validator import RANK


def summarize(alerts: list[dict]) -> dict:
    """Numbers for the dashboard tiles, from the alert rows of one run."""
    collisions = [a for a in alerts if a["kind"] == "collision"]
    lane = [a for a in alerts if a["kind"] == "lane_departure"]
    worst = "none"
    if collisions:
        worst = max((a["level"] for a in collisions), key=lambda lv: RANK.get(lv, 0))
    return {
        "total": len(alerts),
        "collisions": len(collisions),
        "lane_departures": len(lane),
        "worst_collision_level": worst,
    }