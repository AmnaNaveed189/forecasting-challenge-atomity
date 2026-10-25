from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    model_mae = mae(y_true, y_pred)
    baseline_mae = mae(y_true, baseline_pred)
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / baseline_mae)

def relative_improvement(y_true, y_pred, baseline_pred) -> float:
    """1 - MAE(model)/MAE(baseline). NaN when the baseline error is zero."""
    base = mae(y_true, baseline_pred)
    if base == 0:
        return float("nan")
    return float(1.0 - mae(y_true, y_pred) / base)


def subset_metrics(y_true, y_pred, mask) -> dict:
    """MAE and RMSE on the points where mask is True (e.g. peak periods)."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if a.shape != b.shape or a.shape != m.shape:
        raise ValueError("y_true, y_pred and mask must have the same shape")
    if not m.any():
        return {"n": 0, "mae": float("nan"), "rmse": float("nan")}
    return {"n": int(m.sum()), "mae": mae(a[m], b[m]), "rmse": rmse(a[m], b[m])}