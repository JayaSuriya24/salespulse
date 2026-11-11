import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
from ml.train import train_all

csv_path = ROOT / "data/processed/sales.csv"
artifact_dir = ROOT / "artifacts"
r = train_all(str(csv_path), str(artifact_dir))
print(json.dumps(r, indent=2))
try:
    # MLflow >= 3 refuses the local file store unless explicitly allowed (it is in maintenance mode).
    # Local ./mlruns is this project's documented tracking location, so opt in unless the user set it.
    import os
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    mlflow_dir = ROOT / "mlruns"
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment('SalesPulse-M5')
    with mlflow.start_run(run_name='baseline-model-comparison'):
        mlflow.log_params({'training_rows': r['training_rows'], 'test_rows': r['test_rows'], 'products': r['products'], 'stores': r['stores']})
        for row in r['results']:
            for metric in ('MAE', 'RMSE', 'MAPE', 'WAPE'):
                if row.get(metric) is not None:
                    mlflow.log_metric(f"{row['model'].replace(' ', '_')}_{metric}", row[metric])
        mlflow.log_dict(r, 'evaluation.json')
        mlflow.log_artifacts(str(artifact_dir), artifact_path="model_artifacts")
    print(f'MLflow run logged to {mlflow_dir}')
except Exception as e:
    print(f'MLflow logging skipped: {e}')
