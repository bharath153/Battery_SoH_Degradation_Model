# Technical Notes — Battery SoH Degradation Baseline

## Purpose

This repository demonstrates a compact State-of-Health modelling workflow on synthetic Li-ion ageing trajectories.

## Data Generation

Eight synthetic battery trajectories are generated in code. Capacity fade varies with:
- charge cycle;
- C-rate;
- temperature;
- cell-specific degradation-rate variation;
- small random perturbations.

The dataset is a methodology demonstration and is not sourced from a battery test laboratory.

## Implemented Model

The current model is a fitted double-exponential-style degradation baseline implemented with `scipy.optimize.curve_fit`.

It is **not an LSTM** and does not train a recurrent neural network.

## Evaluation

The repository reports:
- R²;
- RMSE;
- MAE;
- threshold-based end-of-life / RUL estimates;
- degradation-factor plots.

## Limitations

- synthetic data only;
- no electrochemical state model;
- no measured battery ageing data;
- no sequence-window neural-network training;
- no uncertainty calibration for deployment.

A real BMS study would require longitudinal cell measurements, temperature/current histories, ageing protocols and out-of-sample validation across cells.
