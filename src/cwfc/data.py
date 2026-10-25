import numpy as np
import pandas as pd


def load_cpu(path):
    return pd.read_csv(path)


def build_targets(cpu: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp: distinct active VMs and mean avg_cpu."""
    valid = cpu.dropna(subset=["avg_cpu"])
    valid = valid[np.isfinite(valid["avg_cpu"])]
    valid = valid[(valid["avg_cpu"] >= 0) & (valid["avg_cpu"] <= 100)]
    valid = valid.drop_duplicates(["vm_id", "timestamp"])

    out = valid.groupby("timestamp").agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    )
    return out.sort_index().reset_index()


def split_train_holdout(targets: pd.DataFrame, n_holdout: int = 24):
    train = targets.iloc[:-n_holdout].reset_index(drop=True)
    holdout = targets.iloc[-n_holdout:].reset_index(drop=True)
    return train, holdout