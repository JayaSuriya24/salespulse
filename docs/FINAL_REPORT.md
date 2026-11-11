# SalesPulse: Automated Retail Sales Prediction System

## Abstract
SalesPulse is a retail sales analytics and forecasting application built using the Walmart M5 Forecasting dataset. The application processes selected product and store sales histories, adds calendar and pricing information, builds time-series features, compares forecasting approaches, and exposes future predictions through a web dashboard and REST API. The current implementation demonstrates the pipeline on a manageable subset of the M5 data.

## Introduction and problem statement
Retailers need estimates of future product demand to support inventory and replenishment decisions. Historical sales contain trends, weekly patterns, event effects, and many zero-sales observations. SalesPulse provides a small, reproducible system for preparing these data and comparing forecasting methods.

## Objectives
- Prepare official Walmart M5 data for analysis.
- Explore historical sales by date, product, and store.
- Build lag-based forecasting features.
- Compare baseline and supervised learning methods.
- Evaluate predictions with common regression metrics.
- Provide forecasts through an API and dashboard.

## Dataset
The project uses `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` from the official Kaggle M5 competition. The local experiment selected the first three stores and first thirty products in source order. The resulting subset contains 172,170 product-store-day observations over 2011-01-29 through 2016-04-24. Raw data are not distributed with the source package.

## Methodology
The wide sales table is converted into long date-indexed rows. Calendar information and weekly sell prices are joined. Zero-sales observations are retained. Features include day-of-week, day, month, weekend, event indicator, sell price, item/store codes, lags 1/7/14/28, and rolling means 7/14/28. All lag and rolling features are shifted so the target day's sales are not included.

The evaluation uses a chronological final-28-day holdout from 2016-03-28 through 2016-04-24. Supervised models are evaluated row-wise for one-step prediction using observed historical lag values. The API uses recursive predictions for multi-day horizons. These are different evaluation settings and should not be conflated.

## System implementation
The backend uses FastAPI and Pydantic. Pandas handles data ingestion and feature preparation. Scikit-learn provides Linear Regression, Decision Tree, and Random Forest models. Naive, seasonal-naive, and moving-average baselines are included. The browser dashboard uses HTML, CSS, JavaScript, and Plotly. Docker and Render deployment configuration are included.

## Algorithms
- Naive: most recent observed sales value.
- Seasonal naive: sales from seven days earlier.
- Moving average: mean of the previous seven days.
- Linear Regression: global regression over time, calendar, item/store and lag features.
- Decision Tree: nonlinear regression tree over the same features.
- Random Forest: ensemble of regression trees (used in the full local training run).

## Experimental results

| Model | MAE | RMSE | MAPE (%) | WAPE (%) |
|---|---:|---:|---:|---:|
| Linear Regression | 1.3977 | 2.8963 | 65.05 | 88.63 |
| Random Forest | 1.4253 | 2.9087 | 71.43 | 90.38 |
| Moving Average | 1.4683 | 3.0532 | 70.66 | 93.11 |
| Decision Tree | 1.5064 | 3.2919 | 75.03 | 95.52 |
| Naive | 1.7321 | 3.9713 | 99.75 | 109.84 |
| Seasonal Naive | 1.8306 | 4.1225 | 98.02 | 116.08 |

Linear Regression had the lowest MAE among the tested approaches. MAPE excludes rows where actual sales are zero. The percentage errors are high, which is important to report rather than hide. These preliminary results apply only to the selected subset and one-step holdout evaluation; they are not full M5 competition results.

## Discussion and limitations
The experiment shows that the lag-feature regression model improves on the tested baselines for the chosen subset. The dataset is sparse and many series have low daily volume, so percentage metrics can be high. The subset is small relative to the full M5 dataset. Recursive forecasts can accumulate errors over longer horizons. The generic CSV upload validates and previews data, while the M5-specific upload performs preprocessing and model training. Forecast IDs are not persisted. MLflow logging is supported by the optional dependency but was not executed in the current offline build environment. Free hosting has ephemeral storage and may sleep.

## Conclusion
SalesPulse demonstrates an end-to-end retail forecasting workflow using genuine M5 data and trained machine-learning models. It includes data preparation, feature engineering, baseline comparison, model evaluation, an API, and an interactive dashboard. Further work can expand the data coverage, strengthen recursive multi-step evaluation, add persistent experiment storage, and improve deployment durability.

## References
1. Kaggle, “M5 Forecasting – Accuracy,” https://www.kaggle.com/competitions/m5-forecasting-accuracy/data.
2. Scikit-learn documentation, https://scikit-learn.org/stable/.
3. FastAPI documentation, https://fastapi.tiangolo.com/.
4. MLflow documentation, https://mlflow.org/docs/latest/.
5. Render documentation, https://render.com/docs/.
