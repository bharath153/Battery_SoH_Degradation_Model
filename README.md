# Battery State-of-Health Degradation Model

> Double-exponential baseline for synthetic Li-ion capacity-fade trajectories

> [!NOTE]
> The repository name is retained for continuity, but the current implementation is **not an LSTM**. The code fits a double-exponential degradation model using `scipy.optimize.curve_fit`. This README describes the implementation as it actually exists.

## Overview

This project explores battery State-of-Health (SoH) estimation on a synthetic ageing dataset inspired by common battery-degradation studies.

The workflow:
- generates eight synthetic battery ageing trajectories;
- varies temperature and C-rate;
- fits a double-exponential degradation curve;
- evaluates prediction error on a held-out synthetic cell;
- derives a simple end-of-life / remaining-useful-life estimate;
- exports plots and metrics for reproducible inspection.

## Model

The fitted model is:

```text
SoH(cycle) = a · exp(-b · cycle) + c
```

This is a compact analytical baseline, not a recurrent neural network.

## Data

The dataset is **synthetic** and generated inside the repository. It is intended for methodology demonstration and software prototyping, not for production BMS validation.

## Results

The repository reports:
- R²;
- RMSE in SoH percentage points;
- MAE;
- predicted vs. synthetic reference degradation;
- end-of-life threshold crossing;
- temperature and C-rate sensitivity plots.

See `metrics.json` for the saved metrics and `plots/` for generated figures.

## Run

```bash
git clone https://github.com/bharath153/P02_Battery_SoH_LSTM.git
cd P02_Battery_SoH_LSTM
pip install -r requirements.txt
python src/train_lstm.py
```

## Repository Structure

```text
P02_Battery_SoH_LSTM/
├── src/
│   └── train_lstm.py
├── plots/
├── docs/
├── metrics.json
├── requirements.txt
└── README.md
```

## Scope

This repository demonstrates:
- synthetic degradation modelling;
- curve fitting;
- error analysis;
- simple RUL threshold logic;
- reproducible engineering visualization.

A genuine LSTM implementation would require an actual recurrent neural-network training pipeline and sequence-window formulation; that is outside the current repository scope.
