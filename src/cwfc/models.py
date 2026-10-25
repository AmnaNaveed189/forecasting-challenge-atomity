from __future__ import annotations

import numpy as np
import pandas as pd


def damped_persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Mean-reverting forecast: mean + phi**h * (last value - mean).

    phi is the lag-1 autocorrelation of the history, clipped to [0, 1].
    """
    values = train[target].astype(float).to_numpy()
    if len(values) < 3:
        raise ValueError("need at least 3 observations")
    mean = float(values.mean())
    dev = values - mean
    denom = float(np.dot(dev[:-1], dev[:-1]))
    phi = 0.0 if denom == 0 else float(np.clip(np.dot(dev[1:], dev[:-1]) / denom, 0.0, 1.0))
    steps = np.arange(1, len(template) + 1)
    return pd.Series(mean + (phi ** steps) * dev[-1], index=template.index, dtype=float)

def ridge_recursive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, alpha: float = 1.0) -> pd.Series:
    """One-step Ridge regression on lag/rolling-mean features, applied recursively.

    Fitted only on `train`. Each predicted value is appended to the history and used
    to build the features for the next step (no true future values are ever used).
    """
    from cwfc.features import build_training_matrix, next_step_features

    values = train[target].astype(float).to_numpy()
    X, y = build_training_matrix(values)
    if len(y) < 10:
        raise ValueError("not enough training rows for ridge")
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    y_mean = float(y.mean())
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y_mean))

    history = list(values)
    preds = []
    for _ in range(len(template)):
        x = (next_step_features(history) - mu) / sd
        nxt = float(y_mean + x @ w)
        preds.append(nxt)
        history.append(nxt)
    return pd.Series(preds, index=template.index, dtype=float)

def ses_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Simple exponential smoothing: a flat forecast equal to the last smoothed level.

    The smoothing weight alpha is chosen from a small grid by one-step-ahead squared
    error on `train` only, so nothing outside the history is used.
    """
    values = train[target].astype(float).to_numpy()
    if len(values) < 3:
        raise ValueError("need at least 3 observations")
    best = None
    for alpha in np.linspace(0.05, 1.0, 20):
        level = values[0]
        sse = 0.0
        for v in values[1:]:
            sse += (v - level) ** 2
            level = alpha * v + (1.0 - alpha) * level
        if best is None or sse < best[0]:
            best = (sse, level)
    return pd.Series(best[1], index=template.index, dtype=float)