import numpy as np
from ml.train import metrics

def test_metrics_handle_zero_actuals():
    result=metrics([0,2,4],[1,2,5])
    assert result['MAE'] > 0
    assert result['RMSE'] > 0
    assert result['MAPE'] is not None
    assert result['WAPE'] is not None

def test_mape_all_zero_actuals():
    result=metrics([0,0],[1,2])
    assert result['MAPE'] is None

def test_wape_zero_total_is_none():
    result=metrics([0,0],[1,2])
    assert result['WAPE'] is None
