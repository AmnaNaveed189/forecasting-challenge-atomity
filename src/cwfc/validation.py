from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out

def to_datetime_timestamps(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Copy of `frame` with integer-second timestamps converted to real Timestamps."""
    out = frame.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], unit="s", utc=True)
    return out


def rolling_origin_forecasts(
    frame: pd.DataFrame,
    target: str,
    forecast_fn,
    timestamp_col: str = "timestamp",
    horizon: int = 24,
    min_train_size: int = 48,
    step: int = 6,
) -> pd.DataFrame:
    """Run `forecast_fn(train, template, target)` from every origin.

    `template` holds only the timestamps to forecast, never the target values.
    Returns one long table: origin, step, timestamp, actual, pred.
    """
    dated = to_datetime_timestamps(frame, timestamp_col)
    splits = rolling_origin_splits(dated, timestamp_col, horizon, min_train_size, step)
    if not splits:
        raise ValueError("no validation origins fit; reduce min_train_size or horizon")
    ts = dated[timestamp_col]
    parts = []
    for k, sp in enumerate(splits):
        train = dated[ts <= sp.train_end]
        val = dated[(ts >= sp.validation_start) & (ts <= sp.validation_end)]
        template = val[[timestamp_col]]
        pred = forecast_fn(train, template, target)
        parts.append(
            pd.DataFrame(
                {
                    "origin": k,
                    "step": range(1, len(val) + 1),
                    "timestamp": frame.loc[val.index, timestamp_col].to_numpy(),
                    "actual": val[target].to_numpy(),
                    "pred": pred.to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)