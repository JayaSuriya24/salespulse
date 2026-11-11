# Final presentation outline

1. **Title:** SalesPulse — Automated Retail Sales Prediction System.
2. **Problem:** Retail demand varies across products, stores, weekdays, and events.
3. **Objectives:** Prepare sales data, compare forecasting approaches, and expose predictions.
4. **Dataset:** Walmart M5 files; 3 stores, 30 products, 172,170 observations in the local experiment.
5. **Architecture:** Dashboard → FastAPI → preprocessing/features → models/evaluation.
6. **Data preparation:** Wide-to-long conversion, calendar/price joins, missing value handling, zero-sales preservation.
7. **Feature engineering:** Calendar variables, lags 1/7/14/28, rolling means 7/14/28.
8. **Models:** Naive, seasonal naive, moving average, Linear Regression, Decision Tree, Random Forest.
9. **Evaluation:** Chronological final-28-day holdout; show MAE/RMSE/MAPE/WAPE table and explain zero handling.
10. **Dashboard demo:** Historical trends, model comparison, select store/product, generate 7- or 28-day forecast.
11. **API and deployment:** Swagger docs, Docker, Render free service limitations.
12. **Limitations and next steps:** Larger M5 coverage, recursive backtesting, durable storage, MLflow run verification.
