import math

import numpy as np
import pytest

from cwfc.policy import alert_rates, apply_band, coverage, headroom, interval_offsets


def test_headroom_is_a_quantile_of_under_forecast_errors():
    actual = np.arange(1.0, 11.0)          # errors 1..10 when pred = 0
    assert headroom(actual, np.zeros(10), level=0.5) == pytest.approx(5.5)


def test_headroom_is_never_negative():
    assert headroom([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], level=0.9) == 0.0


def test_interval_offsets_are_ordered_and_signed_correctly():
    actual = np.linspace(-5, 5, 101)
    lo, hi = interval_offsets(actual, np.zeros(101), level=0.90)
    assert lo < 0 < hi and lo == pytest.approx(-4.5) and hi == pytest.approx(4.5)


def test_apply_band_clips_to_valid_range():
    lower, upper = apply_band([1.0, 99.0], -5.0, 5.0)
    assert lower.tolist() == [0.0, 94.0]
    assert upper.tolist() == [6.0, 100.0]


def test_coverage_counts_values_inside_band():
    assert coverage([1, 2, 3, 4], [0, 0, 0, 0], [2, 2, 2, 2]) == 0.5


def test_alert_rates_hits_and_false_alarms():
    actual = [10, 10, 1, 1, 1]            # two real high-load points
    upper = [11, 5, 11, 5, 5]             # alert at index 0 (hit) and 2 (false alarm)
    out = alert_rates(actual, upper, threshold=9)
    assert out["events"] == 2 and out["normal_points"] == 3
    assert out["hit_rate"] == 0.5
    assert out["false_alarm_rate"] == pytest.approx(1 / 3)


def test_alert_rates_without_events_gives_nan_hit_rate():
    out = alert_rates([1, 2], [3, 4], threshold=100)
    assert out["events"] == 0 and math.isnan(out["hit_rate"])
    assert out["false_alarm_rate"] == 0.0


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        headroom([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        headroom([1.0], [1.0], level=1.5)
    with pytest.raises(ValueError):
        interval_offsets([], [])