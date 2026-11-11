# SalesPulse — Walmart M5 Retail Sales Forecasting

SalesPulse is a compact retail analytics and forecasting application built around the official Walmart M5 Forecasting dataset. It provides M5 preprocessing, time-series feature engineering, baseline and machine-learning model evaluation, recursive forecasts, a FastAPI backend, and a browser dashboard.

## Architecture

- **Data:** official M5 `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv`.
- **Processing:** pandas wide-to-long conversion, calendar and price joins, zero-sales preservation, and lag/rolling features.
- **Models:** naive, seasonal-naive, moving-average, linear regression, decision tree, and random forest.
- **API/UI:** FastAPI + responsive HTML dashboard; API docs at `/docs`.
- **Tracking:** optional local MLflow experiment logging (`pip install -r requirements-mlflow.txt`); JSON evaluation report is always saved.
- **Deployment:** Render free Python web service using `render.yaml`; Docker configuration is also included.

## Quick start

**Supported Python: 3.11–3.13.** (Docker and CI use 3.11. pandas 2.2.3 has no Python 3.14 support; on 3.14 the test suite crashes with a segmentation fault inside pandas.)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
# Optional: install MLflow tracking support
# pip install -r requirements-mlflow.txt
python scripts/prepare_data.py --zip /path/to/m5-forecasting-accuracy.zip --stores 3 --products 30
python scripts/train.py
uvicorn app.main:app --reload --port 7860
```

Open `http://localhost:7860` for the dashboard and `http://localhost:7860/docs` for Swagger.

## Security: M5 upload token

`POST /api/v1/datasets/m5/upload` replaces the live dataset and retrains models, so it requires a bearer token. Everything else (dashboard, forecasts, read endpoints, `/docs`) is public.

- Set the server-side secret with the `M5_UPLOAD_TOKEN` environment variable (at least 16 characters). Generate one: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.
- Clients send `Authorization: Bearer <token>`. The dashboard has an "Upload token" field (masked, never stored).
- Responses: `401` missing/malformed header, `403` wrong token, `503` upload disabled because no valid token is configured (the endpoint fails closed).
- The token is accepted only in the header (never in the URL), is compared in constant time, and is never logged or returned.

```bash
export M5_UPLOAD_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
uvicorn app.main:app --port 7860
curl -H "Authorization: Bearer $M5_UPLOAD_TOKEN" -F sales_file=@sales_train_validation.csv -F calendar_file=@calendar.csv -F prices_file=@sell_prices.csv http://localhost:7860/api/v1/datasets/m5/upload
```

## Data handling

The preprocessing script selects the first three stores and first thirty products in the source order, converts daily columns to long format, joins calendar fields and sell prices, and retains zero sales. Raw competition files are excluded from Git. Do not redistribute the raw dataset; obtain it from the [Kaggle M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) under its terms.

The dashboard includes an M5 uploader for the three required CSV files. Uploaded data is processed locally by the app and a subset is used for training. On free ephemeral hosting, uploaded files and generated models can disappear after a restart or sleep cycle; upload again when necessary.

## API

- `GET /api/v1/health`
- `GET /api/v1/datasets`
- `POST /api/v1/datasets/upload`
- `POST /api/v1/datasets/m5/upload` (three official M5 CSV files; processes and trains)
- `GET /api/v1/sales/summary`
- `GET /api/v1/sales/trends`
- `GET /api/v1/models`
- `GET /api/v1/models/compare`
- `POST /api/v1/forecasts`
- `GET /api/v1/forecasts/{forecast_id}`
- `GET /api/v1/experiments`
- `GET /api/v1/options`

Forecast requests accept `product_id`, `store_id`, `horizon` (1–28), and `model` (`best` or a fitted model name). Multi-step predictions are recursive and do not consume future actual sales.

## Evaluation note

The current evaluation report uses a chronological final-28-day holdout. The sklearn models are scored on one-step rows where lag features come from observed history; recursive future forecasting is used by the API. This distinction is disclosed so the reported metrics are not misrepresented as a recursive 28-day backtest. See `docs/EXPERIMENT_RESULTS.md` for measured preliminary subset results. A separate genuine recursive 28-day backtest (run through the API forecast code, using only history before 2016-03-28) is reported in `SALES PULSE COMPLETE TEST REPORT.md` at the submission root.

## Render free deployment

1. Push this repository to GitHub without raw M5 files.
2. In Render, create a new Blueprint and connect the repository containing `render.yaml`.
3. Confirm the service plan is **Free** before creating it; do not select a paid instance.
4. After deployment, open the generated `onrender.com` URL and `/api/v1/health`.
5. Open the service's **Environment** tab, copy the generated `M5_UPLOAD_TOKEN`, then upload the three official M5 CSVs in the dashboard (paste the token into the "Upload token" field) to initialize data and train models.

Free Render web services use 0.1 CPU and 512 MB RAM, sleep after 15 minutes without traffic, and use an ephemeral filesystem. The hosted M5 upload runs the lighter model set (Linear Regression and Decision Tree plus baselines); uploaded data/model files may need to be uploaded and trained again after a restart. Do not rely on the free service for persistent storage. The M5 processing flow streams the large CSVs in chunks to limit memory use.

## Docker

```bash
docker build -t salespulse .
docker run --rm -p 7860:7860 -e M5_UPLOAD_TOKEN -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" salespulse
```

## Current scope and limitations

This is an accelerated project implementation. It is configured for a 3-store/30-product subset, local artifacts, and a single-process public demo. Generic CSV uploads are validated and previewed; the M5-specific upload performs processing and training. Forecast request history is not persisted. Local MLflow tracking is attempted when the optional dependency is installed. Free-hosted data is ephemeral and the service may sleep.
