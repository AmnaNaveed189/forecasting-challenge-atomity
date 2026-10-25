import numpy as np
import pytest

from cwfc.features import MAX_LAG, build_training_matrix, next_step_features


def test_features_values():
    h = np.arange(1.0, 21.0)   # 1..20
    f = next_step_features(h)
    assert f.tolist() == [20.0, 19.0, 19.0, 17.5, 14.5]


def test_training_row_for_time_t_uses_only_values_before_t():
    rng = np.random.default_rng(0)
    v = rng.normal(size=40)
    X, y = build_training_matrix(v)
    t = 25
    row_before = X[t - MAX_LAG].copy()
    v2 = v.copy()
    v2[t:] = 999.0                      # change t and everything after it
    X2, _ = build_training_matrix(v2)
    assert np.array_equal(X2[t - MAX_LAG], row_before)   # row for t is unchanged
    assert y[t - MAX_LAG] == v[t]


def test_matrix_shape():
    X, y = build_training_matrix(np.arange(30.0))
    assert X.shape == (30 - MAX_LAG, 5) and y.shape == (30 - MAX_LAG,)


def test_too_short_raises():
    with pytest.raises(ValueError):
        next_step_features([1.0] * (MAX_LAG - 1))
    with pytest.raises(ValueError):
        build_training_matrix(np.arange(MAX_LAG, dtype=float))