# System architecture

```mermaid
flowchart LR
    U[User] --> D[SalesPulse Dashboard]
    D --> A[FastAPI REST API]
    A --> P[M5 Ingestion and Preprocessing]
    P --> C[(Processed Time Series)]
    C --> F[Feature Engineering]
    F --> M[Forecasting Models]
    M --> E[Evaluation and Model Artifacts]
    E --> A
    A --> D
    M --> L[Local MLflow Tracking - Optional]
```

## Data processing flow

```mermaid
flowchart TD
    S[sales_train_validation.csv] --> V[Validate and select stores/items]
    C[calendar.csv] --> J[Join by d identifier]
    P[sell_prices.csv] --> K[Join by store, item and week]
    V --> W[Wide-to-long transformation]
    W --> J
    J --> K
    K --> Z[Preserve zero sales and clean types]
    Z --> O[Processed sales.csv]
```

## Model pipeline

```mermaid
flowchart TD
    A[Chronological sales data] --> B[Lag and rolling features]
    B --> C[Chronological train/test split]
    C --> D[Naive baselines]
    C --> E[Linear Regression]
    C --> F[Decision Tree]
    C --> G[Random Forest - local full run]
    D --> H[MAE RMSE MAPE WAPE]
    E --> H
    F --> H
    G --> H
    H --> I[Select best supervised model]
    I --> J[Saved model artifact and forecast API]
```

## Deployment architecture

```mermaid
flowchart LR
    G[GitHub repository] --> R[Render Free Web Service]
    R --> API[FastAPI on dynamic PORT]
    API --> H[Health and Swagger endpoints]
    API --> UI[Dashboard]
    API --> FS[Ephemeral local files]
```

The Render free service uses ephemeral storage; uploaded data and models may need to be reloaded after restart or sleep.
