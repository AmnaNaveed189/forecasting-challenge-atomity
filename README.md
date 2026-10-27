# Cloud Workload Forecasting Challenge

Forecast aggregate cloud workload demand from VM-level CPU readings.

## Requirements

- Python 3.11 or newer
- Git
- The fixed `data-v1` challenge dataset

## Installation

From a clean checkout, create and activate a virtual environment.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the automated tests:

```bash
python -m pytest -q
```

On Windows PowerShell, use the same command after completing the installation
steps above:

```powershell
python -m pytest -q
```

## Dataset

The required dataset is distributed as the fixed `data-v1` GitHub release:

[Download the data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

Download `data-v1.tar.gz`, then extract it into the repository root:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

On Windows PowerShell, run:

```powershell
New-Item -ItemType Directory -Force challenge_data | Out-Null
tar -xzf data-v1.tar.gz -C challenge_data
```

The expected layout is:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

The forecasting pipeline reads `challenge_data/vm_cpu_readings.csv.gz`.
That file contains the following columns:

```text
timestamp,vm_id,min_cpu,max_cpu,avg_cpu
```

The release contains 10,000 VMs and 134 observations per VM at five-minute
intervals. The final 24 aggregate timestamps are the two-hour holdout.

### Dataset integrity check

On Linux/macOS:

```bash
(
  cd challenge_data
  sha256sum -c SHA256SUMS
)
```

On Windows, compare these hashes with the entries in
`challenge_data/SHA256SUMS`:

```powershell
Get-FileHash challenge_data\vm_cpu_readings.csv.gz -Algorithm SHA256
Get-FileHash challenge_data\vm_metadata.csv.gz -Algorithm SHA256
```

## Train and run the forecasting pipeline

The project does not save a separate serialized model file. Training and
forecasting are performed together by the prediction CLI. Each run:

1. Loads and preprocesses the data;
2. Builds the aggregate target series;
3. Uses the first 110 timestamps as the training window;
4. Fits the configured forecasting rules on that training window;
5. Forecasts the final 24 timestamps (the two-hour holdout);
6. Writes and validates `predictions.csv`.

From the repository root on Linux/macOS, run:

```bash
python -m cwfc.predict \
  --data-dir challenge_data \
  --output predictions.csv
```

On Windows PowerShell, run:

```powershell
python -m cwfc.predict `
  --data-dir challenge_data `
  --output predictions.csv
```

The command:

1. Loads `vm_cpu_readings.csv.gz`.
2. Removes invalid CPU readings.
3. Removes duplicate `(vm_id, timestamp)` observations.
4. Aggregates the readings by timestamp.
5. Uses all observations except the final 24 as training data.
6. Fits persistence for `active_vm_count`.
7. Fits damped persistence for `aggregate_cpu_utilization`.
8. Forecasts the final 24 timestamps.
9. Validates the output schema and value ranges.
10. Writes `predictions.csv`.

Expected output:

```text
wrote 24 rows to predictions.csv
```

The generated file has exactly these columns and no index column:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

It contains exactly 24 rows, with timestamps matching the final 24 timestamps
in chronological order.

## Modeling decisions

### Validation and model selection

Models were compared with expanding-window rolling-origin validation inside
the 110-point training window (7 origins, 24-step horizon, minimum history 48,
step 6). Candidates: persistence, training mean, moving averages (3/6/12),
seasonal-naive (reference only), simple exponential smoothing, recursive
Ridge (alpha 1/10/100) and damped persistence. Damped persistence had the
lowest rolling-origin MAE and RMSE. Its margin over the training mean is small,
so the top rankings are not strong evidence. On the final holdout it is
effectively tied with persistence (MAE 0.153 vs 0.147, RMSE 0.183 vs 0.185).
The holdout was never used for fitting, tuning or model selection.

### Capacity policy

The report recommends the damped forecast plus a one-sided safety margin
from validation residuals (see `src/cwfc/policy.py` and notebook 06).

### Target construction

For each timestamp, the pipeline calculates:

```text
active_vm_count =
    number of distinct VM IDs with a valid CPU observation
```

```text
aggregate_cpu_utilization =
    mean of valid VM-level avg_cpu readings
```

Only finite `avg_cpu` values in the inclusive range `[0, 100]` are retained.

### Forecasting methods

`active_vm_count` uses persistence forecasting: every holdout prediction
equals the final observed training value.

`aggregate_cpu_utilization` uses damped persistence forecasting:

```text
forecast(h) = mean + phi^h * (last_value - mean)
```

Here, `phi` is the lag-1 autocorrelation estimated from the training history
and clipped to `[0, 1]`. This creates a stable forecast that gradually
reverts toward the historical mean instead of extrapolating an unrealistic
trend.

### Leakage prevention

The final 24 target values are never used for fitting or forecasting. The
prediction template contains only the holdout timestamps. This behavior is
covered by `tests/test_predict.py`.

Predictions are constrained to:

```text
active_vm_count >= 0
0 <= aggregate_cpu_utilization <= 100
```

## Smoke test with a small dataset

The pipeline can be tested without downloading the full challenge dataset.
The CLI only requires a file named `vm_cpu_readings.csv.gz` containing the
required CPU columns.

Create a temporary smoke dataset with five VMs and 134 timestamps:

```bash
python -c "import numpy as np, pandas as pd; from pathlib import Path; p=Path('smoke_data'); p.mkdir(exist_ok=True); rng=np.random.default_rng(1); rows=[(300*t, f'vm{v}', 1.0, 50.0, float(rng.uniform(1,30))) for t in range(134) for v in range(5)]; pd.DataFrame(rows, columns=['timestamp','vm_id','min_cpu','max_cpu','avg_cpu']).to_csv(p/'vm_cpu_readings.csv.gz', index=False)"
```

On Windows PowerShell, use this equivalent multiline command:

```powershell
@'
import numpy as np
import pandas as pd
from pathlib import Path

p = Path("smoke_data")
p.mkdir(exist_ok=True)
rng = np.random.default_rng(1)
rows = [
    (300 * t, f"vm{v}", 1.0, 50.0, float(rng.uniform(1, 30)))
    for t in range(134)
    for v in range(5)
]
pd.DataFrame(
    rows,
    columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"],
).to_csv(p / "vm_cpu_readings.csv.gz", index=False)
'@ | python -
```

Run the complete prediction pipeline against the smoke data:

```bash
python -m cwfc.predict \
  --data-dir smoke_data \
  --output smoke_predictions.csv
```

On Windows PowerShell:

```powershell
python -m cwfc.predict `
  --data-dir smoke_data `
  --output smoke_predictions.csv
```

Expected output:

```text
wrote 24 rows to smoke_predictions.csv
```

Validate the smoke output:

```bash
python -c "import pandas as pd; p=pd.read_csv('smoke_predictions.csv'); assert list(p.columns)==['timestamp','active_vm_count','aggregate_cpu_utilization']; assert len(p)==24; assert p['timestamp'].tolist()==[300*i for i in range(110,134)]; assert p['active_vm_count'].ge(0).all(); assert p['aggregate_cpu_utilization'].between(0,100).all(); print(p.head())"
```

On Windows PowerShell:

```powershell
python -c "import pandas as pd; p=pd.read_csv('smoke_predictions.csv'); assert list(p.columns)==['timestamp','active_vm_count','aggregate_cpu_utilization']; assert len(p)==24; assert p['timestamp'].tolist()==[300*i for i in range(110,134)]; assert p['active_vm_count'].ge(0).all(); assert p['aggregate_cpu_utilization'].between(0,100).all(); print(p.head())"
```

The repository also includes automated smoke coverage:

```bash
python -m pytest -q tests/test_smoke.py tests/test_predict.py
```

The smoke dataset is intentionally small and synthetic. It verifies that the
pipeline is not hard-coded to the released dataset and that the generated
file has the required format. Smoke testing verifies that the complete
prediction workflow executes correctly and that the generated output follows
the required schema.

## Useful commands

Run all tests:

```bash
python -m pytest -q
```

On Windows PowerShell:

```powershell
python -m pytest -q
```

Run only prediction and smoke tests:

```bash
python -m pytest -q tests/test_predict.py tests/test_smoke.py
```

On Windows PowerShell:

```powershell
python -m pytest -q tests/test_predict.py tests/test_smoke.py
```

Generate the required submission file:

```bash
python -m cwfc.predict --data-dir challenge_data --output predictions.csv
```

On Windows PowerShell:

```powershell
python -m cwfc.predict `
  --data-dir challenge_data `
  --output predictions.csv
```

The final submission artifact is `predictions.csv`.

## Project structure

```text
forecasting-challenge/
├── challenge_data/                  # Fixed Azure VM workload dataset
│   ├── vm_cpu_readings.csv.gz       # Five-minute VM CPU readings
│   ├── vm_metadata.csv.gz           # VM metadata and workload segments
│   ├── DATA_CARD.md                 # Dataset documentation
│   └── SHA256SUMS                   # Dataset integrity checksums
├── notebooks/                       # Exploratory analysis and evaluation
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baselines_and_models.ipynb
│   ├── 04_model_selection.ipynb
│   ├── 05_holdout_evaluation.ipynb
│   └── 06_capacity_policy.ipynb
├── src/cwfc/                        # Reusable forecasting package
│   ├── data.py                      # Data loading, cleaning, and targets
│   ├── baselines.py                 # Persistence and simple baselines
│   ├── features.py                  # Time-series feature construction
│   ├── models.py                    # Forecasting models
│   ├── metrics.py                   # Forecast and relative metrics
│   ├── validation.py                # Time-aware validation utilities
│   ├── policy.py                    # Headroom and capacity-policy helpers
│   └── predict.py                   # Final holdout prediction CLI
├── tests/                           # Automated unit, integration, and smoke tests
├── report/
│   └── Atomity_forecasting_challenge.pdf
├── predictions.csv                  # Generated 24-step holdout submission
├── ASSIGNMENT.md                    # Challenge requirements
├── DATASET.md                      # Dataset contract
├── SUBMISSION.md                   # Submission guidance
└── pyproject.toml                  # Package metadata and dependencies
```

## Final report

The completed project report is available as a PDF in the [`report/`](report/)
folder:

- [Atomity forecasting challenge report](report/Atomity_forecasting_challenge.pdf)

The report covers the executive summary, data preparation, workload analysis,
forecasting approach, time-aware validation, baseline comparison, error
analysis, capacity-planning recommendation, limitations, and future work.
