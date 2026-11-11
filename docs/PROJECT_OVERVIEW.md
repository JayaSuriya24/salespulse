# Project Overview — SalesPulse

## Purpose
SalesPulse explores retail sales history and predicts near-term demand for selected product/store combinations using Walmart M5 data.

## Users
Students, instructors, and analysts reviewing historical sales, comparing forecasting approaches, and exploring 7- or 28-day predictions.

## Main capabilities
1. Upload official M5 CSV files and process a manageable subset.
2. Review aggregate daily sales history.
3. Compare baseline and supervised forecasting methods.
4. Request recursive forecasts for a selected item and store.
5. Inspect model evaluation through JSON APIs and the dashboard.

## Scope
The accelerated build processes the first 3 stores and first 30 items from the M5 source order. The processing code can be adjusted to expand the subset. It does not claim to reproduce the full M5 competition leaderboard.

## Success criteria
- The official dataset is processed without replacing it.
- Zero-sales observations remain in the data.
- Evaluation uses a chronological holdout.
- Forecasting is backed by fitted models or explicit baselines.
- API and dashboard load and return consistent information.
