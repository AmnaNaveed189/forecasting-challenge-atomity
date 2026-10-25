"""Checks on the real release. Skipped automatically if challenge_data/ is absent."""
from pathlib import Path

import pytest

from cwfc.data import build_targets, load_cpu, split_train_holdout

CPU_FILE = Path(__file__).resolve().parents[1] / "challenge_data" / "vm_cpu_readings.csv.gz"


@pytest.mark.skipif(not CPU_FILE.exists(), reason="challenge_data/ not downloaded")
def test_real_release_targets_and_split():
    targets = build_targets(load_cpu(CPU_FILE))
    assert len(targets) == 134
    assert targets["timestamp"].diff().dropna().eq(300).all()
    assert targets["aggregate_cpu_utilization"].between(0, 100).all()
    assert (targets["active_vm_count"] == targets["active_vm_count"].iloc[0]).all()

    train, holdout = split_train_holdout(targets)
    assert (len(train), len(holdout)) == (110, 24)