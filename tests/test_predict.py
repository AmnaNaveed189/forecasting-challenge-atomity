import numpy as np
import pandas as pd
import pytest

from cwfc.predict import COLUMNS, main, make_predictions, validate_predictions


def make_targets(n=134, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp": [300 * i for i in range(n)],
        "active_vm_count": 10000,
        "aggregate_cpu_utilization": 9.4 + 0.2 * rng.standard_normal(n),
    })


def test_output_has_required_columns_rows_and_timestamps():
    targets = make_targets()
    pred = make_predictions(targets)
    assert list(pred.columns) == COLUMNS
    assert len(pred) == 24
    assert pred["timestamp"].tolist() == targets["timestamp"].tail(24).tolist()
    validate_predictions(pred, targets["timestamp"].tail(24).tolist())


def test_predictions_do_not_depend_on_holdout_target_values():
    targets = make_targets()
    altered = targets.copy()
    altered.loc[altered.index[-24:], "aggregate_cpu_utilization"] = 99.0
    altered.loc[altered.index[-24:], "active_vm_count"] = 1
    a = make_predictions(targets)
    b = make_predictions(altered)
    pd.testing.assert_frame_equal(a, b)


def test_values_are_finite_and_in_range():
    pred = make_predictions(make_targets())
    assert np.isfinite(pred[COLUMNS[1:]].to_numpy()).all()
    assert (pred["active_vm_count"] >= 0).all()
    assert pred["aggregate_cpu_utilization"].between(0, 100).all()


def test_validate_rejects_bad_outputs():
    targets = make_targets()
    ts = targets["timestamp"].tail(24).tolist()
    good = make_predictions(targets)

    with pytest.raises(ValueError):
        validate_predictions(good.iloc[:23], ts)                        # too few rows
    with pytest.raises(ValueError):
        validate_predictions(good[COLUMNS[::-1]], ts)                    # wrong column order
    bad = good.copy(); bad.loc[3, "aggregate_cpu_utilization"] = np.nan
    with pytest.raises(ValueError):
        validate_predictions(bad, ts)                                    # not finite
    bad = good.copy(); bad.loc[3, "aggregate_cpu_utilization"] = 120.0
    with pytest.raises(ValueError):
        validate_predictions(bad, ts)                                    # out of range
    bad = good.copy(); bad.loc[3, "active_vm_count"] = -1.0
    with pytest.raises(ValueError):
        validate_predictions(bad, ts)                                    # negative count
    bad = good.copy(); bad.loc[5, "timestamp"] = bad.loc[4, "timestamp"]
    with pytest.raises(ValueError):
        validate_predictions(bad, ts)                                    # wrong / duplicate timestamps


def test_command_line_writes_a_valid_csv(tmp_path):
    rng = np.random.default_rng(1)
    rows = [(300 * t, f"vm{v}", 1.0, 50.0, float(rng.uniform(1, 30)))
            for t in range(134) for v in range(5)]
    pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]).to_csv(
        tmp_path / "vm_cpu_readings.csv.gz", index=False
    )
    out_file = tmp_path / "predictions.csv"
    assert main(["--data-dir", str(tmp_path), "--output", str(out_file)]) == 0

    written = pd.read_csv(out_file)
    assert list(written.columns) == COLUMNS          # no extra index column
    assert len(written) == 24
    assert written["timestamp"].tolist() == [300 * i for i in range(110, 134)]