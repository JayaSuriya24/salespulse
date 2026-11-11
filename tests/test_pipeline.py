import pytest
import zipfile
import pandas as pd
from ml.pipeline import prepare_m5, add_features, extract_m5

def test_prepare_m5_and_feature_lags(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    dates = pd.date_range('2015-01-01', periods=70)
    rows = []
    for store in ['CA_1', 'TX_1', 'WI_1']:
        for item in ['ITEM_1', 'ITEM_2']:
            row = {'id': f'{item}_{store}', 'item_id': item, 'dept_id': 'D', 'cat_id': 'C', 'store_id': store, 'state_id': store[:2]}
            row.update({f'd_{i+1}': (i % 5) for i in range(70)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(raw / 'sales_train_validation.csv', index=False)
    cal = pd.DataFrame({'d': [f'd_{i+1}' for i in range(70)], 'date': dates, 'wm_yr_wk': [i // 7 for i in range(70)], 'wday': [d.dayofweek for d in dates], 'month': [d.month for d in dates], 'year': [d.year for d in dates], 'event_name_1': [None] * 70, 'event_type_1': [None] * 70, 'snap_CA': [0] * 70, 'snap_TX': [0] * 70, 'snap_WI': [0] * 70})
    cal.to_csv(raw / 'calendar.csv', index=False)
    price = pd.DataFrame([{'store_id': s, 'item_id': it, 'wm_yr_wk': wk, 'sell_price': 2.0} for s in ['CA_1', 'TX_1', 'WI_1'] for it in ['ITEM_1', 'ITEM_2'] for wk in range(11)])
    price.to_csv(raw / 'sell_prices.csv', index=False)
    out = tmp_path / 'processed.csv'
    df = prepare_m5(str(raw), str(out), stores=3, products=2)
    assert len(df) == 70 * 3 * 2
    assert (df.sales == 0).any()
    feat = add_features(df)
    first = feat[(feat.store_id == 'CA_1') & (feat.item_id == 'ITEM_1')].iloc[0]
    assert first.lag_1 == first.sales or first.lag_1 >= 0
    assert out.exists()

def test_lag_features_do_not_include_current_target():
    dates = pd.date_range('2020-01-01', periods=70)
    df = pd.DataFrame({'store_id': ['S'] * 70, 'item_id': ['I'] * 70, 'date': dates, 'sales': list(range(70)),
                     'event_flag': [0] * 70, 'sell_price': [1.0] * 70})
    before = add_features(df)
    changed = df.copy()
    changed.loc[60, 'sales'] = 9999
    after = add_features(changed)
    b = before[before.date == dates[60]].iloc[0]
    a = after[after.date == dates[60]].iloc[0]
    assert b.lag_1 == a.lag_1
    assert b.rolling_mean_7 == a.rolling_mean_7

def test_prepare_m5_missing_files_error(tmp_path):
    empty_dir = tmp_path / "empty_raw"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        prepare_m5(raw_dir=str(empty_dir), out_path=str(tmp_path / "out.csv"))
    assert "Missing required M5 source file" in str(excinfo.value)

def test_extract_m5_missing_file_in_archive(tmp_path):
    zip_path = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("calendar.csv", "dummy")
    with pytest.raises(ValueError) as excinfo:
        extract_m5(str(zip_path), raw_dir=str(tmp_path / "raw"))
    assert "M5 archive is missing" in str(excinfo.value)

def test_extract_m5_success(tmp_path):
    zip_path = tmp_path / "complete.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv"]:
            z.writestr(f, "header\nval\n")
    raw_dir = tmp_path / "extracted_raw"
    saved = extract_m5(str(zip_path), raw_dir=str(raw_dir))
    assert len(saved) == 3
    for f in ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv"]:
        assert (raw_dir / f).exists()
