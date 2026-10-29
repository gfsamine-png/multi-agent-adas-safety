from __future__ import annotations

from dataclasses import dataclass

from adas.safety.risk import RiskAssessment

LEVELS = ("unknown", "low", "moderate", "high", "critical")
RANK = {name: i for i, name in enumerate(LEVELS)}


@dataclass(frozen=True)
class ValidatorConfig:
    alert_level: str = "high"   # minimum level that can become an alert
    confirm_s: float = 0.3      # risk must persist this long before alerting
    clear_s: float = 1.0        # risk must stay below this long to end an alert
    cooldown_s: float = 3.0     # minimum time between alerts for one object


@dataclass(frozen=True)
class Alert:
    track_id: int
    level: str
    timestamp_s: float
    reason: str
    ttc_s: float | None


class _TrackState:
    def __init__(self) -> None:
        self.pending_since: float | None = None
        self.below_since: float | None = None
        self.active = False
        self.last_alert_s: float | None = None
        self.alert_rank = 0


class AlertValidator:
    """Turns per-frame risk levels into stable alerts (confirm, hold, cooldown)."""

    def __init__(self, cfg: ValidatorConfig = ValidatorConfig()) -> None:
        if cfg.alert_level not in RANK:
            raise ValueError(f"unknown alert level: {cfg.alert_level}")
        self.cfg = cfg
        self._threshold = RANK[cfg.alert_level]
        self._states: dict[int, _TrackState] = {}
        self._last_t: float | None = None

    def update(
        self, timestamp_s: float, assessments: dict[int, RiskAssessment]
    ) -> list[Alert]:
        """Feed one frame. Returns the new alerts raised on this frame."""
        if self._last_t is not None and timestamp_s <= self._last_t:
            raise ValueError("timestamps must strictly increase")
        for a in assessments.values():
            if a.level not in RANK:
                raise ValueError(f"unknown risk level: {a.level}")
        self._last_t = timestamp_s

        alerts: list[Alert] = []
        for tid in set(self._states) | set(assessments):
            st = self._states.setdefault(tid, _TrackState())
            a = assessments.get(tid)
            rank = RANK[a.level] if a is not None else 0
            if rank >= self._threshold and a is not None:
                alert = self._on_high(st, tid, a, rank, timestamp_s)
                if alert is not None:
                    alerts.append(alert)
            else:
                self._on_low(st, timestamp_s)
            # forget objects that are gone and have nothing pending
            if a is None and not st.active and st.pending_since is None:
                del self._states[tid]
        return alerts

    def _on_high(
        self, st: _TrackState, tid: int, a: RiskAssessment, rank: int, t: float
    ) -> Alert | None:
        st.below_since = None
        if st.pending_since is None:
            st.pending_since = t
        if t - st.pending_since < self.cfg.confirm_s:
            return None
        escalated = st.active and rank > st.alert_rank
        cooled = st.last_alert_s is None or t - st.last_alert_s >= self.cfg.cooldown_s
        if (not st.active and cooled) or escalated:
            st.active = True
            st.last_alert_s = t
            st.alert_rank = rank
            return Alert(tid, a.level, t, a.reason, a.ttc_s)
        return None

    def _on_low(self, st: _TrackState, t: float) -> None:
        st.pending_since = None
        if st.active:
            if st.below_since is None:
                st.below_since = t
            if t - st.below_since >= self.cfg.clear_s:
                st.active = False
                st.alert_rank = 0
                st.below_since = None