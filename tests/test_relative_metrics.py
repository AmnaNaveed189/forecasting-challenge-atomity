import math

import pytest

from cwfc.metrics import relative_improvement, smape, subset_metrics


def test_relative_improvement_halved_error_is_fifty_percent():
    y = [0.0, 0.0]
    assert relative_improvement(y, [1.0, 1.0], [2.0, 2.0]) == pytest.approx(0.5)


def test_relative_improvement_is_negative_when_model_is_worse():
    y = [0.0, 0.0]
    assert relative_improvement(y, [3.0, 3.0], [2.0, 2.0]) < 0


def test_relative_improvement_is_nan_when_baseline_error_is_zero():
    y = [5.0, 5.0]
    assert math.isnan(relative_improvement(y, [5.0, 5.0], [5.0, 5.0]))


def test_subset_metrics_uses_only_masked_points():
    y = [0.0, 0.0, 0.0, 0.0]
    p = [1.0, 9.0, 3.0, 9.0]
    out = subset_metrics(y, p, [True, False, True, False])
    assert out["n"] == 2
    assert out["mae"] == pytest.approx(2.0)
    assert out["rmse"] == pytest.approx(math.sqrt((1 + 9) / 2))


def test_subset_metrics_empty_mask_gives_nan_not_an_error():
    out = subset_metrics([1.0, 2.0], [1.0, 2.0], [False, False])
    assert out["n"] == 0
    assert math.isnan(out["mae"]) and math.isnan(out["rmse"])


def test_subset_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        subset_metrics([1.0, 2.0], [1.0, 2.0], [True])


def test_smape_handles_zero_denominator():
    # both actual and prediction are 0: no division error, no NaN
    assert smape([0.0, 0.0], [0.0, 0.0]) == 0.0