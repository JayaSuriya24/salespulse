# SalesPulse preliminary experiment results

**Dataset:** official Walmart M5 `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv`.

**Subset:** first 3 stores and first 30 products in source order; 172,170 daily product-store observations; date range 2011-01-29 through 2016-04-24.

**Evaluation:** chronological final-28-day holdout (2016-03-28 through 2016-04-24). The supervised models are evaluated row-wise for one-step prediction using observed historical lag features. This is not a fully recursive 28-day backtest. MAPE excludes zero-actual rows; WAPE is computed over total absolute actual sales.

| Model | MAE | RMSE | MAPE (%) | WAPE (%) |
|---|---:|---:|---:|---:|
| Linear Regression | 1.3977 | 2.8963 | 65.05 | 88.63 |
| Random Forest | 1.4253 | 2.9087 | 71.43 | 90.38 |
| Moving Average (7-day) | 1.4683 | 3.0532 | 70.66 | 93.11 |
| Decision Tree | 1.5064 | 3.2919 | 75.03 | 95.52 |
| Naive (lag 1) | 1.7321 | 3.9713 | 99.75 | 109.84 |
| Seasonal Naive (lag 7) | 1.8306 | 4.1225 | 98.02 | 116.08 |

**Best supervised model by holdout MAE:** Linear Regression.

**Training times measured in this workspace:** Linear Regression 0.08s; Decision Tree 0.71s; Random Forest 12.38s. Times depend on hardware and environment.

These are preliminary subset results, not full M5 competition results and not a claim of high forecast accuracy. The relatively high percentage errors reflect the sparse, low-volume series in the selected subset.
