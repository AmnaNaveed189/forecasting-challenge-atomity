import numpy as np
import pandas as pd
import pytest

from cwfc.models import damped_persistence_forecast


def make_train(values):
    return pd.DataFrame({"timestamp": [300 * i for i in range(len(values))], "y": values})


def make_template(n, fill=0.0):
    return pd.DataFrame({"timestamp": [10_000 + 300 * i for i in range(n)], "y": fill})


def test_constant_series_gives_constant_forecast_without_nan():
    out = damped_persistence_forecast(make_train([5.0] * 10), make_template(4), "y")
    assert out.tolist() == [5.0] * 4
    assert not out.isna().any()


def test_no_autocorrelation_falls_back_to_the_mean():
    # alternating series: lag-1 autocorrelation is negative, phi is clipped to 0
    out = damped_persistence_forecast(make_train([1.0, 3.0] * 5), make_template(3), "y")
    assert np.allclose(out.to_numpy(), 2.0)


def test_forecast_moves_back_toward_the_mean_step_by_step():
    values = [float(v) for v in range(1, 11)]   # ramp: strong positive autocorrelation
    mean = float(np.mean(values))
    out = damped_persistence_forecast(make_train(values), make_template(24), "y").to_numpy()
    gaps = np.abs(out - mean)
    assert np.all(np.diff(gaps) <= 1e-12)        # distance to the mean never grows
    assert mean < out[0] <= values[-1]           # first step lies between mean and last value


def test_keeps_template_index_and_length():
    template = make_template(5)
    template.index = [10, 11, 12, 13, 14]
    out = damped_persistence_forecast(make_train([1.0, 2.0, 4.0, 3.0]), template, "y")
    assert out.index.tolist() == [10, 11, 12, 13, 14]


def test_ignores_target_values_in_template():
    train = make_train([1.0, 2.0, 4.0, 3.0, 5.0])
    a = damped_persistence_forecast(train, make_template(4, fill=0.0), "y")
    b = damped_persistence_forecast(train, make_template(4, fill=999.0), "y")
    assert a.tolist() == b.tolist()


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        damped_persistence_forecast(make_train([1.0, 2.0]), make_template(3), "y")

# ---- ridge_recursive_forecast ----
from cwfc.models import ridge_recursive_forecast


def _ar_train(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = [9.4]
    for _ in range(n - 1):
        y.append(9.4 + 0.7 * (y[-1] - 9.4) + 0.15 * rng.standard_normal())
    return make_train(y)


def test_ridge_constant_series_gives_constant_forecast():
    out = ridge_recursive_forecast(make_train([5.0] * 40), make_template(6), "y")
    assert np.allclose(out.to_numpy(), 5.0)


def test_ridge_returns_finite_forecast_with_template_index():
    template = make_template(24)
    template.index = range(100, 124)
    out = ridge_recursive_forecast(_ar_train(), template, "y", alpha=5.0)
    assert len(out) == 24 and out.index.tolist() == list(range(100, 124))
    assert np.isfinite(out.to_numpy()).all()


def test_ridge_ignores_target_values_in_template():
    train = _ar_train()
    a = ridge_recursive_forecast(train, make_template(10, fill=0.0), "y")
    b = ridge_recursive_forecast(train, make_template(10, fill=999.0), "y")
    assert a.tolist() == b.tolist()


def test_ridge_too_short_raises():
    with pytest.raises(ValueError):
        ridge_recursive_forecast(make_train([1.0, 2.0, 3.0] * 5), make_template(3), "y")

# ---- ses_forecast ----
from cwfc.models import ses_forecast


def test_ses_constant_series_gives_the_constant():
    out = ses_forecast(make_train([5.0] * 10), make_template(4), "y")
    assert np.allclose(out.to_numpy(), 5.0)


def test_ses_is_flat_and_between_min_and_max_of_history():
    train = _ar_train(40)
    out = ses_forecast(train, make_template(6), "y").to_numpy()
    assert np.allclose(out, out[0])
    assert train["y"].min() <= out[0] <= train["y"].max()


def test_ses_ignores_template_values_and_checks_length():
    train = _ar_train(40)
    a = ses_forecast(train, make_template(5, fill=0.0), "y")
    b = ses_forecast(train, make_template(5, fill=999.0), "y")
    assert a.tolist() == b.tolist()
    with pytest.raises(ValueError):
        ses_forecast(make_train([1.0, 2.0]), make_template(2), "y")