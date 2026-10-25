import numpy as np
import pandas as pd

from cwfc.baselines import moving_average_forecast, persistence_forecast
from cwfc.data import build_targets, split_train_holdout
from cwfc.models import damped_persistence_forecast
from cwfc.validation import rolling_origin_forecasts


def tiny_cpu(n_vms=6, n_times=60):
    rng = np.random.default_rng(0)
    rows = [
        (300 * t, f"vm{v}", 1.0, 90.0, float(rng.uniform(1, 30)))
        for t in range(n_times) for v in range(n_vms)
    ]
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


def test_pipeline_runs_end_to_end_on_a_small_dataset():
    targets = build_targets(tiny_cpu())
    assert len(targets) == 60 and (targets["active_vm_count"] == 6).all()

    train, holdout = split_train_holdout(targets, n_holdout=12)
    assert (len(train), len(holdout)) == (48, 12)

    fns = [persistence_forecast, damped_persistence_forecast,
           lambda tr, te, t: moving_average_forecast(tr, te, t, window=3)]
    for fn in fns:
        res = rolling_origin_forecasts(
            train, "aggregate_cpu_utilization", fn, horizon=6, min_train_size=18, step=6
        )
        assert len(res) > 0
        assert np.isfinite(res[["actual", "pred"]].to_numpy()).all()
        assert res["pred"].between(0, 100).all()