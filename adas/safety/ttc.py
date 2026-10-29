from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

TTCStatus = Literal[
    "approaching", "not_approaching", "uncertain", "insufficient_data"
]

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TTCEstimate:
    status: TTCStatus
    ttc_s: float | None      # best estimate in seconds, None if not meaningful
    ttc_min_s: float | None  # pessimistic end of the range (smaller = worse)
    ttc_max_s: float | None  # optimistic end, None = unbounded
    n_samples: int


def bbox_scale(bbox: BBox) -> float:
    """Apparent size of an object: box width in pixels.

    Width is used instead of area or height because a box often stretches
    vertically when it merges with an object behind or above it (for example
    a rickshaw behind a car), which would look like a sudden approach.
    """
    x1, _, x2, _ = bbox
    return max(x2 - x1, 0.0)


def estimate_ttc(
    times: list[float],
    scales: list[float],
    min_samples: int = 8,
    min_span_s: float = 0.5,
    g_min: float = 0.05,  # growth rate (1/s) below which: not approaching
) -> TTCEstimate:
    """TTC from looming: fit ln(scale) = g * t + c, then TTC = 1 / g.

    Also returns an uncertainty range from the standard error of the slope g.
    """
    pairs = [(t, s) for t, s in zip(times, scales) if s > 0]
    n = len(pairs)
    insufficient = TTCEstimate("insufficient_data", None, None, None, n)
    if n < max(min_samples, 3):
        return insufficient
    t = np.array([p[0] for p in pairs], dtype=float)
    y = np.log(np.array([p[1] for p in pairs], dtype=float))
    if t[-1] - t[0] < min_span_s:
        return insufficient

    tc = t - t.mean()
    sxx = float(np.sum(tc ** 2))
    g = float(np.sum(tc * (y - y.mean())) / sxx)
    resid = y - (y.mean() + g * tc)
    se = math.sqrt(float(np.sum(resid ** 2)) / (n - 2) / sxx)

    lower, upper = g - 2 * se, g + 2 * se    # roughly a 95% range for g
    if lower > g_min:
        status: TTCStatus = "approaching"
    elif upper < g_min:
        return TTCEstimate("not_approaching", None, None, None, n)
    else:
        status = "uncertain"

    ttc = 1.0 / g if g > 0 else None
    ttc_min = 1.0 / upper if upper > 0 else None
    ttc_max = 1.0 / lower if lower > 0 else None
    return TTCEstimate(status, ttc, ttc_min, ttc_max, n)


class TTCTracker:
    """Keeps a short size history per track ID and estimates TTC from it."""

    def __init__(self, window_s: float = 1.5, max_age_s: float = 2.0,
                 **params) -> None:
        self.window_s = window_s
        self.max_age_s = max_age_s
        self._params = params
        self._hist: dict[int, deque[tuple[float, float]]] = {}

    def update(self, track_id: int, t: float, bbox: BBox) -> None:
        scale = bbox_scale(bbox)
        if scale <= 0:
            return
        hist = self._hist.setdefault(track_id, deque())
        if hist and t - hist[-1][0] > self.max_age_s:
            hist.clear()                  # long gap: do not trust old samples
        hist.append((t, scale))
        while hist and t - hist[0][0] > self.window_s:
            hist.popleft()

    def estimate(self, track_id: int) -> TTCEstimate:
        hist = self._hist.get(track_id)
        if not hist:
            return TTCEstimate("insufficient_data", None, None, None, 0)
        times = [p[0] for p in hist]
        scales = [p[1] for p in hist]
        return estimate_ttc(times, scales, **self._params)

    def prune(self, now: float) -> None:
        """Forget tracks that have not been seen recently."""
        stale = [tid for tid, h in self._hist.items()
                 if not h or now - h[-1][0] > self.max_age_s]
        for tid in stale:
            del self._hist[tid]