"""Simple capacity-planning helpers built from forecast errors.

Errors are defined as actual - forecast, so a positive error means the forecast
was too low (the risk of under-provisioning).
"""
from __future__ import annotations

import numpy as np


def _errors(actual, pred) -> np.ndarray:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    if a.shape != p.shape:
        raise ValueError("actual and pred must have the same shape")
    if a.size == 0:
        raise ValueError("need at least one error")
    return a - p


def headroom(actual, pred, level: float = 0.90) -> float:
    """One-sided safety margin: the `level` quantile of (actual - forecast), never below 0."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    return float(max(np.quantile(_errors(actual, pred), level), 0.0))


def interval_offsets(actual, pred, level: float = 0.90) -> tuple[float, float]:
    """(lower, upper) offsets to add to a forecast for a central `level` interval."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    err = _errors(actual, pred)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(err, alpha)), float(np.quantile(err, 1.0 - alpha))


def apply_band(pred, lower_offset: float, upper_offset: float, low: float = 0.0, high: float = 100.0):
    """Forecast band, clipped to the valid range [low, high]."""
    p = np.asarray(pred, dtype=float)
    return np.clip(p + lower_offset, low, high), np.clip(p + upper_offset, low, high)


def coverage(actual, lower, upper) -> float:
    """Share of actual values that fall inside [lower, upper]."""
    a = np.asarray(actual, dtype=float)
    return float(np.mean((a >= np.asarray(lower)) & (a <= np.asarray(upper))))


def alert_rates(actual, upper, threshold: float) -> dict:
    """Compare 'alert when upper bound >= threshold' with real high-load points.

    hit_rate: share of real high-load points that had an alert.
    false_alarm_rate: share of normal points that had an alert.
    """
    a = np.asarray(actual, dtype=float)
    u = np.asarray(upper, dtype=float)
    event = a >= threshold
    alert = u >= threshold
    n_event, n_normal = int(event.sum()), int((~event).sum())
    return {
        "events": n_event,
        "normal_points": n_normal,
        "hit_rate": float(alert[event].mean()) if n_event else float("nan"),
        "false_alarm_rate": float(alert[~event].mean()) if n_normal else float("nan"),
    }