import pandas as pd
import pytest

from cwfc.baselines import persistence_forecast
from cwfc.validation import rolling_origin_forecasts


def make_frame(n=110):
    return pd.DataFrame({
        "timestamp": [300 * i for i in range(n)],
        "y": [float(i) for i in range(n)],
    })


def test_default_setup_gives_seven_origins_of_24_steps():
    out = rolling_origin_forecasts(make_frame(), "y", persistence_forecast)
    assert out["origin"].nunique() == 7
    assert (out.groupby("origin")["step"].max() == 24).all()
    assert len(out) == 7 * 24
    assert list(out.columns) == ["origin", "step", "timestamp", "actual", "pred"]


def test_first_origin_window_and_integer_timestamps():
    out = rolling_origin_forecasts(make_frame(), "y", persistence_forecast)
    first = out[out["origin"] == 0]
    assert first["timestamp"].iloc[0] == 48 * 300          # original seconds, not dates
    assert first["timestamp"].iloc[-1] == (48 + 23) * 300
    assert pd.api.types.is_integer_dtype(out["timestamp"])


def test_persistence_uses_only_the_value_before_the_window():
    out = rolling_origin_forecasts(make_frame(), "y", persistence_forecast)
    for _, g in out.groupby("origin"):
        assert (g["pred"] == g["actual"].iloc[0] - 1).all()


def test_forecast_function_never_sees_target_values_of_the_window():
    seen = {}

    def spy(train, template, target):
        seen["template_columns"] = list(template.columns)
        assert train["timestamp"].max() < template["timestamp"].min()
        return pd.Series(0.0, index=template.index)

    rolling_origin_forecasts(make_frame(), "y", spy)
    assert seen["template_columns"] == ["timestamp"]       # no target column at all


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        rolling_origin_forecasts(make_frame(30), "y", persistence_forecast)


def test_custom_parameters_change_the_number_of_origins():
    out = rolling_origin_forecasts(
        make_frame(40), "y", persistence_forecast, horizon=6, min_train_size=12, step=6
    )
    assert out["origin"].nunique() == 4   # origins at 12, 18, 24, 30 points of history