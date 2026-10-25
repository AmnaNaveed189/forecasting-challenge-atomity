from __future__ import annotations

import numpy as np

MAX_LAG = 12  # longest look-back used by any feature (12 points = 1 hour)


def next_step_features(history) -> np.ndarray:
    """Features for predicting the value right after `history`.

    Uses only the values inside `history`, so they are known at the forecast origin.
    """
    h = np.asarray(history, dtype=float)
    if len(h) < MAX_LAG:
        raise ValueError(f"need at least {MAX_LAG} past values")
    return np.array([h[-1], h[-2], h[-3:].mean(), h[-6:].mean(), h[-12:].mean()])


def build_training_matrix(values):
    """One row per time t >= MAX_LAG: features from values[:t] (strictly before t) -> values[t]."""
    v = np.asarray(values, dtype=float)
    rows = [next_step_features(v[:t]) for t in range(MAX_LAG, len(v))]
    if not rows:
        raise ValueError("series too short to build any training row")
    return np.vstack(rows), v[MAX_LAG:]