import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib
from ml.train import train_all, metrics
from ml.pipeline import FEATURES

def test_metrics_calculation_accuracy():
    actual = [1.0, 2.0, 3.0, 4.0]
    pred = [1.0, 1.0, 5.0, 4.0]
    # MAE = (0 + 1 + 2 + 0)/4 = 0.75
    # MSE = (0 + 1 + 4 + 0)/4 = 1.25 -> RMSE = sqrt(1.25) = 1.1180339887
    res = metrics(actual, pred)
    assert np.isclose(res['MAE'], 0.75)
    assert np.isclose(res['RMSE'], np.sqrt(1.25))
    # WAPE = sum(|a-p|)/sum(|a|) * 100 = 3/10 * 100 = 30.0%
    assert np.isclose(res['WAPE'], 30.0)

@pytest.mark.requires_data
def test_train_all_reproducibility():
    csv_path = Path(__file__).resolve().parents[1] / "data/processed/sales.csv"
    if not csv_path.exists():
        pytest.skip("Processed dataset not available")
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    eval_json = artifacts / "evaluation.json"
    assert eval_json.exists()
    report = json.loads(eval_json.read_text())
    assert report['products'] == 30
    assert report['stores'] == 3
    assert report['training_rows'] == 167130
    assert report['test_rows'] == 2520
    assert report['best_model'] == 'Linear Regression'

@pytest.mark.requires_data
def test_artifacts_exist_and_loadable():
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    for m in ["linear_regression", "decision_tree", "random_forest"]:
        p = artifacts / f"{m}.joblib"
        assert p.exists()
        model = joblib.load(p)
        assert hasattr(model, 'predict')

@pytest.mark.requires_data
def test_feature_columns_json_matches_pipeline():
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    fc = json.loads((artifacts / "feature_columns.json").read_text())
    assert fc == FEATURES

def test_train_all_execution(tmp_path):
    dates = pd.date_range("2020-01-01", periods=150)
    rows = []
    for s in ["CA_1"]:
        for it in ["ITEM_1", "ITEM_2"]:
            for i, d in enumerate(dates):
                rows.append({
                    "store_id": s,
                    "item_id": it,
                    "date": d,
                    "sales": float(i % 5),
                    "event_flag": 0,
                    "sell_price": 2.5
                })
    csv_path = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    art_dir = tmp_path / "artifacts"
    report = train_all(str(csv_path), str(art_dir), include_forest=False)
    assert report["best_model"] is not None
    assert (art_dir / "evaluation.json").exists()
    assert (art_dir / "linear_regression.joblib").exists()
    assert (art_dir / "decision_tree.joblib").exists()
