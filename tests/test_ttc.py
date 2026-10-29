import numpy as np
import pytest

from adas.safety.ttc import TTCTracker, bbox_scale, estimate_ttc


def _times(n: int = 31, fps: float = 30.0) -> list[float]:
    return [i / fps for i in range(n)]          # 1 second of video


def test_exact_ttc_for_constant_growth_rate():
    times = _times()
    scales = [50 * np.exp(0.5 * t) for t in times]   # grows 0.5/s -> TTC = 2 s
    est = estimate_ttc(times, scales)
    assert est.status == "approaching"
    assert est.ttc_s == pytest.approx(2.0, rel=1e-6)


def test_constant_size_is_not_approaching():
    times = _times()
    est = estimate_ttc(times, [80.0] * len(times))
    assert est.status == "not_approaching"
    assert est.ttc_s is None


def test_shrinking_object_is_not_approaching():
    times = _times()
    est = estimate_ttc(times, [80 * np.exp(-0.5 * t) for t in times])
    assert est.status == "not_approaching"


def test_too_few_samples_is_insufficient():
    times = _times(n=4)
    assert estimate_ttc(times, [50, 51, 52, 53]).status == "insufficient_data"


def test_short_time_span_is_insufficient():
    times = [i / 120 for i in range(20)]        # 20 samples but only 0.16 s
    scales = [50 * np.exp(0.5 * t) for t in times]
    assert estimate_ttc(times, scales).status == "insufficient_data"


def test_noisy_data_still_gives_reasonable_range():
    rng = np.random.default_rng(0)
    times = _times()
    scales = [50 * np.exp(0.5 * t) * (1 + rng.normal(0, 0.05)) for t in times]
    est = estimate_ttc(times, scales)
    assert est.status == "approaching"
    assert 1.5 < est.ttc_s < 2.7
    assert est.ttc_min_s < est.ttc_s


def test_tracker_follows_one_track_and_ignores_others():
    tracker = TTCTracker()
    for t in _times():
        size = 50 * np.exp(0.5 * t)
        tracker.update(7, t, (0, 0, size, size))
        tracker.update(8, t, (0, 0, 60, 60))     # constant size
    assert tracker.estimate(7).status == "approaching"
    assert tracker.estimate(8).status == "not_approaching"
    assert tracker.estimate(99).status == "insufficient_data"


def test_tracker_clears_history_after_long_gap():
    tracker = TTCTracker(max_age_s=1.0)
    for t in _times():
        tracker.update(1, t, (0, 0, 50, 50))
    tracker.update(1, 10.0, (0, 0, 50, 50))      # 9 s later
    assert tracker.estimate(1).n_samples == 1


def test_bbox_scale():
    assert bbox_scale((0, 0, 100, 100)) == pytest.approx(100.0)
    assert bbox_scale((10, 10, 5, 5)) == 0.0     # inverted box -> no size
def test_bbox_scale_ignores_height_changes():
    assert bbox_scale((0, 0, 60, 40)) == bbox_scale((0, 0, 60, 100))