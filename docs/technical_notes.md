# Technical Notes — Battery State-of-Health Digital Twin — LSTM

## Data Generation Methodology
The synthetic dataset is constructed using physics-informed equations that capture the dominant engineering relationships in Battery Management Systems. While synthetic, the data distributions and feature interactions are calibrated to match published literature values.

## Model Selection Rationale
**LSTM / Double-Exponential Sequence Model** was chosen for the following reasons:
- Appropriate for the dataset size and feature structure
- Validated performance on similar engineering regression/classification tasks in literature
- Interpretable outputs compatible with engineering design workflows

## Hyperparameter Tuning
Where applicable, hyperparameters were tuned via cross-validated grid or random search. Final parameters are documented in the main script.

## Known Limitations
1. Synthetic data does not capture all real-world noise and sensor artefacts
2. Model is validated within the training design space (interpolation); extrapolation should be treated with caution
3. For production deployment, re-training on real experimental data is required

## References
- Forrester, A.I.J., Sóbester, A., Keane, A.J. (2008). *Engineering Design via Surrogate Modelling: A Practical Guide*. Wiley.
- Scikit-learn: Pedregosa et al. (2011). Journal of Machine Learning Research, 12, 2825-2830.
- Domain-specific references are noted inline in the main script.
