# Influenza Vaccination and Transmission ABM

This anonymous repository contains the code and aggregated weekly inputs used for an agent-based model (ABM) of influenza vaccination and transmission in Kunshan. It is provided for double-blind peer review.

## Repository contents

- `historical_simulation.py`: historical simulation and baseline analysis.
- `forecast_2026.py`: forecast and policy-scenario analysis.
- `parameter_calibration.py`: genetic-algorithm calibration of vaccination parameters.
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

Run parameter calibration:

```bash
python parameter_calibration.py
```

The calibration performs repeated Monte Carlo simulations and may require substantial computation time. Random seeds and simulation settings are defined near the configuration sections of each script.

## Data

The included CSV files contain weekly, aggregated ILI percentages and influenza positivity rates. They do not contain individual-level records. The scripts resolve data paths relative to this repository, so no machine-specific paths need to be configured.

## Review status

Author names, affiliations, manuscript title, acknowledgements, citation metadata, and repository license are intentionally omitted during double-blind review. They should be added when the repository is made public after peer review.
