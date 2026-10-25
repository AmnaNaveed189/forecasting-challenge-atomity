"""Generate predictions.csv for the final 24 timestamps.

Usage (from the repository root):
    python -m cwfc.predict --data-dir challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.baselines import persistence_forecast
from cwfc.data import build_targets, load_cpu, split_train_holdout
from cwfc.models import damped_persistence_forecast

COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
CPU_FILE = "vm_cpu_readings.csv.gz"
N_HOLDOUT = 24


def make_predictions(targets: pd.DataFrame, n_holdout: int = N_HOLDOUT) -> pd.DataFrame:
    """Fit on everything before the holdout and forecast the holdout timestamps."""
    train, holdout = split_train_holdout(targets, n_holdout)
    template = holdout[["timestamp"]]   # only the timestamps, never the holdout target values

    out = template.copy()
    out["active_vm_count"] = persistence_forecast(train, template, "active_vm_count")
    out["aggregate_cpu_utilization"] = damped_persistence_forecast(
        train, template, "aggregate_cpu_utilization"
    )
    out["active_vm_count"] = out["active_vm_count"].clip(lower=0.0)
    out["aggregate_cpu_utilization"] = out["aggregate_cpu_utilization"].clip(0.0, 100.0)
    return out[COLUMNS].reset_index(drop=True)


def validate_predictions(pred: pd.DataFrame, expected_timestamps) -> None:
    """Raise ValueError if the output breaks the submission format."""
    if list(pred.columns) != COLUMNS:
        raise ValueError(f"columns must be exactly {COLUMNS}")
    if len(pred) != len(expected_timestamps):
        raise ValueError(f"expected {len(expected_timestamps)} rows, got {len(pred)}")
    if pred["timestamp"].tolist() != list(expected_timestamps):
        raise ValueError("timestamps do not match the holdout timestamps")
    if not pred["timestamp"].is_monotonic_increasing or pred["timestamp"].duplicated().any():
        raise ValueError("timestamps must be strictly increasing without duplicates")
    values = pred[COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("all predictions must be finite numbers")
    if (pred["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be >= 0")
    if not pred["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("aggregate_cpu_utilization must be within [0, 100]")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="challenge_data")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args(argv)

    cpu = load_cpu(Path(args.data_dir) / CPU_FILE)
    targets = build_targets(cpu)
    pred = make_predictions(targets)
    validate_predictions(pred, targets["timestamp"].tail(N_HOLDOUT).tolist())
    pred.to_csv(args.output, index=False)
    print(f"wrote {len(pred)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())