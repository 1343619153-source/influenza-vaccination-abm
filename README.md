# Prediction of influenza vaccination uptake

This repository contains the source code and input data for influenza‑vaccination uptake and transmission in Kunshan.

## Repository contents

- `historical_simulation.py`: historical simulation and baseline analysis.
- `forecast_2026.py`: forecast and policy-scenario analysis.
- `sarima_baseline.py`: SARIMA forecast benchmark for the 2025/2026 season.
- `negative_binomial_baseline.py`: nested negative-binomial/Poisson benchmark evaluated on 2025 observations.
- `data/`: aggregated weekly influenza-like illness (ILI) and positivity-rate inputs.
- `requirements.txt`: Python dependencies.

Generated figures, spreadsheets, and CSV files are written under `outputs/`. This directory is excluded from version control.

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducing the analyses

Run the historical simulation:

```bash
python historical_simulation.py
```

Run the forecast and policy scenarios:

```bash
python forecast_2026.py
```

Run the SARIMA benchmark:

```bash
python sarima_baseline.py
```

Run the negative-binomial benchmark:

```bash
python negative_binomial_baseline.py
```

The network simulations and SARIMA grid search may require substantial computation time. Random seeds and simulation settings are defined in each script.

## Data

The included CSV files contain weekly aggregated ILI and positivity rates plus monthly vaccination inputs for the benchmark models. They do not contain individual-level records. All scripts resolve data and output paths relative to this repository, so no machine-specific paths need to be configured.
