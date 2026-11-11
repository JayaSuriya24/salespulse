# Requirements

## Functional requirements
- Accept the three official M5 CSV files.
- Validate file extensions and expected M5 columns.
- Select a configurable store/product subset.
- Reshape daily sales to long format and join calendar and pricing data.
- Calculate lag, rolling, and calendar features without using future sales as lag inputs.
- Train and compare at least three forecasting approaches.
- Calculate MAE, RMSE, MAPE, and WAPE.
- Generate recursive 1–28 day forecasts.
- Expose dataset, analytics, model, experiment, and forecast APIs.
- Provide an interactive dashboard and CSV upload workflow.

## Non-functional requirements
- Use free-tier hosting only.
- Do not store secrets in source control.
- Do not commit or publicly redistribute raw M5 competition files.
- Keep the code modular and testable.
- Clearly disclose evaluation and hosting limitations.
