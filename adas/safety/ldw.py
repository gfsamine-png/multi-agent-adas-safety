from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LDWConfig:
    warn_offset: float = 0.30   # |center_offset| (fraction of lane width) that triggers
    confirm_s: float = 0.5      # must persist this long before warning
    clear_s: float = 1.0        # must stay back near center this long to end a warning
    cooldown_s: float = 3.0     # minimum time between two warnings


@dataclass(frozen=True)
class LDWAlert:
    timestamp_s: float
    side: str       # "left" or "right"
    offset: float   # center_offset at the moment of the alert


class LaneDepartureWarner:
    """Warns when the car drifts toward a lane line (confirm, hold, cooldown).

    center_offset: fraction of lane width, negative = left of lane center,
    positive = right. None means the lane was not found: no decision is made.
    """

    def __init__(self, cfg: LDWConfig = LDWConfig()) -> None:
        self.cfg = cfg
        self._last_t: float | None = None
        self._pending_since: float | None = None
        self._below_since: float | None = None
        self._active = False
        self._last_alert_s: float | None = None

    def update(self, timestamp_s: float, center_offset: float | None) -> LDWAlert | None:
        if self._last_t is not None and timestamp_s <= self._last_t:
            raise ValueError("timestamps must strictly increase")
        self._last_t = timestamp_s
        if center_offset is not None and not math.isfinite(center_offset):
            center_offset = None

        if center_offset is not None and abs(center_offset) >= self.cfg.warn_offset:
            self._below_since = None
            if self._pending_since is None:
                self._pending_since = timestamp_s
            confirmed = timestamp_s - self._pending_since >= self.cfg.confirm_s
            cooled = (
                self._last_alert_s is None
                or timestamp_s - self._last_alert_s >= self.cfg.cooldown_s
            )
            if confirmed and cooled and not self._active:
                self._active = True
                self._last_alert_s = timestamp_s
                side = "left" if center_offset < 0 else "right"
                return LDWAlert(timestamp_s, side, center_offset)
            return None

        self._pending_since = None
        if self._active:
            if self._below_since is None:
                self._below_since = timestamp_s
            if timestamp_s - self._below_since >= self.cfg.clear_s:
                self._active = False
                self._below_since = None
        return None