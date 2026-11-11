import pytest
import io
from fastapi.testclient import TestClient
from app.main import app
from tests.conftest import AUTH

client = TestClient(app)

def test_health():
    r = client.get('/api/v1/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'

def test_dashboard_loads():
    r = client.get('/')
    assert r.status_code == 200 and 'SalesPulse' in r.text

@pytest.mark.requires_data
def test_summary_has_real_dataset():
    r = client.get('/api/v1/sales/summary')
    assert r.status_code == 200
    assert r.json()['products'] == 30
    assert r.json()['stores'] == 3

def test_datasets_endpoint():
    r = client.get('/api/v1/datasets')
    assert r.status_code == 200
    data = r.json()
    assert 'datasets' in data
    if data['datasets']:
        ds = data['datasets'][0]
        assert ds['id'] == 'm5_subset'
        assert ds['products'] == 30
        assert ds['stores'] == 3
        assert ds['rows'] == 172170

@pytest.mark.requires_data
def test_trends_endpoint():
    r = client.get('/api/v1/sales/trends')
    assert r.status_code == 200
    data = r.json()
    assert 'data' in data
    assert len(data['data']) > 1000
    assert 'date' in data['data'][0]
    assert 'sales' in data['data'][0]

@pytest.mark.requires_data
def test_models_and_compare_endpoints():
    r1 = client.get('/api/v1/models')
    r2 = client.get('/api/v1/models/compare')
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()
    data = r1.json()
    assert 'best_model' in data
    assert 'results' in data
    assert len(data['results']) >= 6

@pytest.mark.requires_data
def test_experiments_endpoint():
    r = client.get('/api/v1/experiments')
    assert r.status_code == 200
    data = r.json()
    assert 'tracking' in data
    assert 'mlflow_available' in data
    assert data['evaluation'] is not None

@pytest.mark.requires_data
def test_options_endpoint():
    r = client.get('/api/v1/options')
    assert r.status_code == 200
    data = r.json()
    assert 'stores' in data and len(data['stores']) == 3
    assert 'products' in data and len(data['products']) == 30

def test_forecast_id_lookup_404():
    r = client.get('/api/v1/forecasts/fake-id-123')
    assert r.status_code == 404
    assert 'Forecast IDs' in r.json()['detail']

def test_generic_upload_valid_csv():
    csv_bytes = b"date,sales\n2020-01-01,10\n2020-01-02,15\n"
    r = client.post('/api/v1/datasets/upload', files={'file': ('sample.csv', io.BytesIO(csv_bytes), 'text/csv')})
    assert r.status_code == 200
    assert r.json()['status'] == 'uploaded'

def test_generic_upload_non_csv():
    r = client.post('/api/v1/datasets/upload', files={'file': ('sample.txt', io.BytesIO(b"hello"), 'text/plain')})
    assert r.status_code == 400
    assert 'Upload a CSV file' in r.json()['detail']

def test_generic_upload_invalid_columns():
    csv_bytes = b"foo,bar\n1,2\n"
    r = client.post('/api/v1/datasets/upload', files={'file': ('bad.csv', io.BytesIO(csv_bytes), 'text/csv')})
    assert r.status_code == 400
    assert 'Expected long format' in r.json()['detail']

def test_m5_upload_invalid_schema():
    files = {
        'sales_file': ('sales.csv', io.BytesIO(b"bad,cols\n1,2\n"), 'text/csv'),
        'calendar_file': ('calendar.csv', io.BytesIO(b"bad,cols\n1,2\n"), 'text/csv'),
        'prices_file': ('prices.csv', io.BytesIO(b"bad,cols\n1,2\n"), 'text/csv')
    }
    r = client.post('/api/v1/datasets/m5/upload', files=files, headers=AUTH)
    assert r.status_code == 400
    assert 'must contain' in r.json()['detail']
