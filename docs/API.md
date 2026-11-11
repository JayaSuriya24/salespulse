# API guide

Base URL: `http://localhost:7860` locally; use the deployed Render URL after deployment.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Health status |
| GET | `/api/v1/datasets` | Available processed dataset summary |
| POST | `/api/v1/datasets/upload` | Validate and preview a generic CSV |
| POST | `/api/v1/datasets/m5/upload` | Upload M5 sales, calendar and price CSVs; process and train. **Requires `Authorization: Bearer <M5_UPLOAD_TOKEN>`** |
| GET | `/api/v1/sales/summary` | Aggregate sales KPIs |
| GET | `/api/v1/sales/trends` | Daily historical sales series |
| GET | `/api/v1/models` | Model evaluation report |
| GET | `/api/v1/models/compare` | Model comparison |
| POST | `/api/v1/forecasts` | Generate a forecast |
| GET | `/api/v1/forecasts/{forecast_id}` | Forecast lookup (not persisted in this MVP) |
| GET | `/api/v1/experiments` | Local experiment metadata |
| GET | `/api/v1/options` | Product and store options |

## Forecast request

```json
{
  "dataset_id": "m5_subset",
  "product_id": "HOBBIES_1_001",
  "store_id": "CA_1",
  "horizon": 7,
  "model": "best"
}
```

`horizon` must be 1–28. `model` can be `best` or a model name present in the current evaluation report. The response contains the model, horizon, and date/value pairs.

Additional response fields:

- `method`: `trained_model_recursive` (Linear Regression / Decision Tree / Random Forest artifact) or `baseline_rule_recursive` (Naive / Seasonal Naive / Moving Average rule).
- `history`: the last 60 observed days of the **same** store/product series, so charts compare like with like.

Status codes: `422` invalid input (e.g. horizon outside 1–28), `404` no processed dataset or unknown product/store, `400` model not in the current training run, `503` models not trained yet or artifact missing.

## Upload limits

- `/api/v1/datasets/upload` (preview only) rejects files larger than `MAX_UPLOAD_MB` (default 50).
- `/api/v1/datasets/m5/upload` accepts `stores` (1–10) and `products` (1–3049) query parameters. Processing and training happen in a staging folder; live data and models are replaced only if both succeed.
- `/api/v1/health` also returns `data_loaded` and `models_trained` readiness flags.

Interactive Swagger documentation is available at `/docs`.

## Authentication (M5 upload only)

`POST /api/v1/datasets/m5/upload` is the only protected endpoint. Send `Authorization: Bearer <token>`, where the token equals the server's `M5_UPLOAD_TOKEN` environment variable (minimum 16 characters).

| Status | Meaning |
|---|---|
| 401 | `Authorization` header missing or not a non-empty `Bearer` credential (response carries `WWW-Authenticate: Bearer`) |
| 403 | Token does not match |
| 503 | Server has no valid `M5_UPLOAD_TOKEN` (unset, blank or shorter than 16 characters), so uploads are disabled |

Authentication is checked before the request body is read, so rejected requests cost no bandwidth or disk. The token is read from the header only (a token in the query string is ignored because URLs are logged) and is never written to logs or responses.
