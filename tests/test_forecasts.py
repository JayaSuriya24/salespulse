import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.requires_data
def test_forecast_best_model():
    r = client.post('/api/v1/forecasts', json={
        "product_id": "HOBBIES_1_001",
        "store_id": "CA_1",
        "horizon": 7,
        "model": "best"
    })
    assert r.status_code == 200
    data = r.json()
    assert data['model'] in ["Linear Regression", "Random Forest", "Decision Tree"]
    assert len(data['predictions']) == 7
    for p in data['predictions']:
        assert 'date' in p
        assert p['predicted_sales'] >= 0.0

@pytest.mark.requires_data
def test_forecast_horizon_28_days():
    r = client.post('/api/v1/forecasts', json={
        "product_id": "HOBBIES_1_001",
        "store_id": "CA_1",
        "horizon": 28,
        "model": "Linear Regression"
    })
    assert r.status_code == 200
    assert len(r.json()['predictions']) == 28

@pytest.mark.requires_data
def test_forecast_all_stores():
    for store in ["CA_1", "CA_2", "CA_3"]:
        r = client.post('/api/v1/forecasts', json={
            "product_id": "HOBBIES_1_002",
            "store_id": store,
            "horizon": 7,
            "model": "Linear Regression"
        })
        assert r.status_code == 200
        assert len(r.json()['predictions']) == 7

@pytest.mark.requires_data
def test_forecast_baselines_are_distinct():
    preds = {}
    for model in ["Naive", "Seasonal Naive", "Moving Average"]:
        r = client.post('/api/v1/forecasts', json={
            "product_id": "HOBBIES_1_001",
            "store_id": "CA_1",
            "horizon": 7,
            "model": model
        })
        assert r.status_code == 200
        preds[model] = [p['predicted_sales'] for p in r.json()['predictions']]

    # Verify that Naive, Seasonal Naive, and Moving Average produce distinct sequences
    assert preds["Naive"] != preds["Seasonal Naive"]
    assert preds["Naive"] != preds["Moving Average"]
    assert preds["Seasonal Naive"] != preds["Moving Average"]

@pytest.mark.requires_data
def test_forecast_random_forest_and_decision_tree():
    for model in ["Decision Tree", "Random Forest"]:
        r = client.post('/api/v1/forecasts', json={
            "product_id": "HOBBIES_1_001",
            "store_id": "CA_1",
            "horizon": 7,
            "model": model
        })
        assert r.status_code == 200
        assert r.json()['model'] == model
        assert len(r.json()['predictions']) == 7

def test_forecast_invalid_horizon_low():
    r = client.post('/api/v1/forecasts', json={
        "product_id": "HOBBIES_1_001",
        "store_id": "CA_1",
        "horizon": 0,
        "model": "best"
    })
    assert r.status_code == 422

def test_forecast_invalid_horizon_high():
    r = client.post('/api/v1/forecasts', json={
        "product_id": "HOBBIES_1_001",
        "store_id": "CA_1",
        "horizon": 29,
        "model": "best"
    })
    assert r.status_code == 422

def test_forecast_missing_product():
    r = client.post('/api/v1/forecasts', json={
        "store_id": "CA_1",
        "horizon": 7
    })
    assert r.status_code == 422

@pytest.mark.requires_data
def test_forecast_unknown_combination():
    r = client.post('/api/v1/forecasts', json={
        "product_id": "NONEXISTENT_ITEM",
        "store_id": "CA_1",
        "horizon": 7
    })
    assert r.status_code == 404
    assert 'combination not found' in r.json()['detail']

@pytest.mark.requires_data
def test_forecast_invalid_model():
    r = client.post('/api/v1/forecasts', json={
        "product_id": "HOBBIES_1_001",
        "store_id": "CA_1",
        "horizon": 7,
        "model": "GhostModel_9000"
    })
    assert r.status_code == 400
    assert 'not available' in r.json()['detail']

