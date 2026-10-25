import pandas as pd
import pytest

from cwfc.baselines import (
    moving_average_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
    train_mean_forecast,
)


def make_train(values):
    return pd.DataFrame({"timestamp": [300 * i for i in range(len(values))], "y": values})


def make_template(target_values):
    n = len(target_values)
    return pd.DataFrame(
        {"timestamp": [900 + 300 * i for i in range(n)], "y": target_values},
        index=[3 + i for i in range(n)],
    )


def test_persistence_repeats_last_value_and_keeps_index():
    out = persistence_forecast(make_train([1.0, 2.0, 3.0]), make_template([0.0, 0.0]), "y")
    assert out.tolist() == [3.0, 3.0]
    assert out.index.tolist() == [3, 4]


def test_moving_average_uses_last_window():
    out = moving_average_forecast(make_train([1.0, 2.0, 3.0]), make_template([0.0, 0.0]), "y", window=2)
    assert out.tolist() == [2.5, 2.5]


def test_train_mean():
    out = train_mean_forecast(make_train([1.0, 2.0, 3.0]), make_template([0.0, 0.0]), "y")
    assert out.tolist() == [2.0, 2.0]


def test_seasonal_naive_repeats_last_season():
    out = seasonal_naive_forecast(
        make_train([1.0, 2.0, 3.0]), make_template([0.0, 0.0, 0.0]), "y", season_length=2
    )
    assert out.tolist() == [2.0, 3.0, 2.0]


def test_bad_inputs_raise():
    train = make_train([1.0, 2.0, 3.0])
    template = make_template([0.0])
    with pytest.raises(ValueError):
        moving_average_forecast(train, template, "y", window=0)
    with pytest.raises(ValueError):
        moving_average_forecast(train, template, "y", window=4)
    with pytest.raises(ValueError):
        persistence_forecast(train.iloc[0:0], template, "y")
    with pytest.raises(ValueError):
        train_mean_forecast(train.iloc[0:0], template, "y")
    with pytest.raises(ValueError):
        seasonal_naive_forecast(train, template, "y", season_length=0)


def test_forecasts_ignore_target_values_in_template():
    """Leakage guard: whatever sits in the template's target column must not matter."""
    train = make_train([1.0, 2.0, 3.0, 4.0])
    low = make_template([0.0, 0.0])
    high = make_template([999.0, -999.0])
    fns = [
        lambda t: persistence_forecast(train, t, "y"),
        lambda t: moving_average_forecast(train, t, "y", window=3),
        lambda t: train_mean_forecast(train, t, "y"),
        lambda t: seasonal_naive_forecast(train, t, "y", season_length=2),
    ]
    for fn in fns:
        assert fn(low).tolist() == fn(high).tolist()