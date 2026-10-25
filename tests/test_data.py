import numpy as np
import pandas as pd

from cwfc.data import build_targets, split_train_holdout


def make_cpu(rows):
    return pd.DataFrame(
        rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
    )


def test_targets_match_worked_example():
    # Same numbers as the worked example in the project guide
    cpu = make_cpu([
        (300, "A", 1, 50, 20.0),
        (300, "B", 1, 50, 40.0),
        (300, "C", 1, 70, 60.0),
        (600, "A", 1, 50, 30.0),
        (600, "B", 1, 50, np.nan),
    ])
    out = build_targets(cpu)
    assert out["timestamp"].tolist() == [300, 600]
    assert out["active_vm_count"].tolist() == [3, 1]
    assert out["aggregate_cpu_utilization"].tolist() == [40.0, 30.0]


def test_active_vm_count_counts_distinct_vms_not_rows():
    cpu = make_cpu([
        (300, "A", 1, 5, 2.0),
        (300, "A", 1, 5, 2.0),   # exact duplicate row
        (300, "B", 1, 5, 4.0),
    ])
    out = build_targets(cpu)
    assert out["active_vm_count"].tolist() == [2]
    assert out["aggregate_cpu_utilization"].tolist() == [3.0]


def test_cpu_is_a_mean_not_a_sum():
    cpu = make_cpu([(300, "A", 1, 9, 10.0), (300, "B", 1, 9, 30.0)])
    out = build_targets(cpu)
    assert out["aggregate_cpu_utilization"].iloc[0] == 20.0


def test_invalid_values_are_not_counted():
    cpu = make_cpu([
        (300, "A", 1, 9, 10.0),
        (300, "B", 1, 9, np.nan),
        (300, "C", 1, 9, np.inf),
        (300, "D", 1, 9, -5.0),
        (300, "E", 1, 9, 150.0),
    ])
    out = build_targets(cpu)
    assert out["active_vm_count"].tolist() == [1]
    assert out["aggregate_cpu_utilization"].tolist() == [10.0]


def test_output_is_sorted_with_one_row_per_timestamp():
    cpu = make_cpu([
        (900, "A", 1, 9, 5.0),
        (300, "A", 1, 9, 1.0),
        (600, "A", 1, 9, 3.0),
        (300, "B", 1, 9, 2.0),
    ])
    out = build_targets(cpu)
    assert out["timestamp"].tolist() == [300, 600, 900]
    assert out["timestamp"].is_unique


def test_aggregate_cpu_stays_between_0_and_100():
    rng = np.random.default_rng(0)
    rows = [(t, f"vm{i}", 0, 100, float(rng.uniform(0, 100)))
            for t in (0, 300, 600) for i in range(50)]
    out = build_targets(make_cpu(rows))
    assert out["aggregate_cpu_utilization"].between(0, 100).all()


def test_vm_count_is_not_hardcoded():
    cpu = make_cpu([(0, f"vm{i}", 1, 9, 5.0) for i in range(7)])
    assert build_targets(cpu)["active_vm_count"].tolist() == [7]


def test_split_sizes_and_order():
    targets = pd.DataFrame({
        "timestamp": [300 * i for i in range(134)],
        "active_vm_count": 10000,
        "aggregate_cpu_utilization": np.linspace(9, 10, 134),
    })
    train, holdout = split_train_holdout(targets)
    assert (len(train), len(holdout)) == (110, 24)
    assert train["timestamp"].max() < holdout["timestamp"].min()
    assert holdout["timestamp"].tolist() == targets["timestamp"].tail(24).tolist()
    assert holdout["timestamp"].iloc[0] - train["timestamp"].iloc[-1] == 300