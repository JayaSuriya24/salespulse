# SalesPulse project status

## Completed and verified in the local workspace
- M5 ZIP inspection and required source-file verification.
- Chunked extraction/preprocessing for the official sales, calendar, and price CSVs.
- 3-store/30-product subset: 172,170 rows, 2011-01-29 through 2016-04-24.
- Calendar and sell-price joins; zero-sales observations retained.
- Lag and rolling-window feature engineering.
- Naive, seasonal-naive, moving-average, linear regression, decision tree, and random forest comparison.
- MAE, RMSE, MAPE (excluding zero-actual rows), and WAPE calculations.
- Best model selected by the measured chronological holdout MAE.
- Recursive 1–28 day prediction endpoint.
- FastAPI dashboard, Swagger docs, M5 upload-and-train workflow, and API endpoints.
- Dockerfile, docker-compose, Render free-service Blueprint, and GitHub Actions CI workflow.
- Documentation, architecture diagrams, evaluation report, and presentation outline.
- Seven pytest tests pass; API smoke tests pass, including 7-day forecast calls for all three stores.

## Not completed / requires user account access or environment capability
- MLflow package could not be installed in this offline runtime. Optional integration code and `requirements-mlflow.txt` are included; an MLflow run has not been verified.
- Docker is not installed in this runtime, so Docker image build/start was not tested.
- No GitHub repository URL was supplied and no GitHub connection is active; repository creation/push is pending.
- No Render account/repository connection is available; public deployment is pending.
- No real browser screenshot was captured.

## Hosting constraint
Hugging Face Docker Spaces are not a zero-cost choice for this user requirement because the current documentation says Docker/Gradio compute requires a paid plan. Render Free is selected instead, with 512 MB RAM, 0.1 CPU, 15-minute idle spin-down, and ephemeral local storage. The hosted M5 upload uses the lighter model set (Linear Regression and Decision Tree plus baselines) and streams source CSVs in chunks. Uploaded data and models may need to be reloaded after restarts.

## Evaluation caveat
The supervised models are scored on one-step holdout rows using observed historical lag values; the API uses recursive forecasts. Do not describe the reported metrics as a recursive 28-day backtest.
