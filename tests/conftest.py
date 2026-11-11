import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
TEST_UPLOAD_TOKEN = "test-token-" + "x" * 24  # synthetic value used only by the test-suite
AUTH = {"Authorization": f"Bearer {TEST_UPLOAD_TOKEN}"}
REAL_DATA_AVAILABLE = (CODE_ROOT / "data/processed/sales.csv").exists() and (CODE_ROOT / "artifacts/evaluation.json").exists()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_data: needs the processed M5 subset and trained artifacts (gitignored, so absent in CI)")
    # Spurious FP warnings from Apple Accelerate BLAS in LinearRegression.predict (results verified finite and
    # identical to a non-BLAS einsum; see evidence/audit/24_linear_regression_warning_check.txt).
    config.addinivalue_line("filterwarnings", "ignore:.*encountered in matmul:RuntimeWarning")


def pytest_collection_modifyitems(config, items):
    # Explicit, reported skip (shown in the summary with a reason) - not a silent pass.
    if REAL_DATA_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="Processed M5 data/artifacts not present (run scripts/prepare_data.py and scripts/train.py)")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolate_raw_uploads(tmp_path, monkeypatch):
    """Uploads during tests go to a temp dir instead of polluting the real code/data/raw."""
    import app.main as main
    raw = tmp_path / "raw_uploads"
    raw.mkdir()
    monkeypatch.setattr(main, "RAW_DIR", raw)
    yield raw


def write_synthetic_m5(raw_dir: Path, days: int = 120, stores=("CA_1", "TX_1", "WI_1"), items=("ITEM_1", "ITEM_2")):
    """Small file set with the official M5 schemas (synthetic values, used only for tests)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2015-01-01", periods=days)
    rows = []
    for s in stores:
        for k, it in enumerate(items):
            row = {"id": f"{it}_{s}_validation", "item_id": it, "dept_id": "D", "cat_id": "C", "store_id": s, "state_id": s[:2]}
            row.update({f"d_{i+1}": (i + k) % 5 for i in range(days)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(raw_dir / "sales_train_validation.csv", index=False)
    pd.DataFrame({"d": [f"d_{i+1}" for i in range(days)], "date": dates.strftime("%Y-%m-%d"),
                  "wm_yr_wk": [i // 7 for i in range(days)], "wday": [d.dayofweek for d in dates],
                  "month": [d.month for d in dates], "year": [d.year for d in dates],
                  "event_name_1": [None] * days, "event_type_1": [None] * days,
                  "snap_CA": [0] * days, "snap_TX": [0] * days, "snap_WI": [0] * days}).to_csv(raw_dir / "calendar.csv", index=False)
    pd.DataFrame([{"store_id": s, "item_id": it, "wm_yr_wk": wk, "sell_price": 2.0}
                  for s in stores for it in items for wk in range(days // 7 + 1)]).to_csv(raw_dir / "sell_prices.csv", index=False)
    return raw_dir


@pytest.fixture(autouse=True)
def configure_upload_token(monkeypatch):
    """Every test runs with a known, synthetic upload token; tests that need other states override it."""
    monkeypatch.setenv("M5_UPLOAD_TOKEN", TEST_UPLOAD_TOKEN)
