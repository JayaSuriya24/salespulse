"""Regression tests for issues found in the end-to-end audit (see SALES PULSE COMPLETE TEST REPORT.md)."""
import inspect
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from ml.pipeline import prepare_m5
from ml.train import train_all
from tests.conftest import write_synthetic_m5, CODE_ROOT, AUTH

client = TestClient(main.app)


def _m5_files(raw: Path):
    return {
        "sales_file": ("sales_train_validation.csv", (raw / "sales_train_validation.csv").read_bytes(), "text/csv"),
        "calendar_file": ("calendar.csv", (raw / "calendar.csv").read_bytes(), "text/csv"),
        "prices_file": ("sell_prices.csv", (raw / "sell_prices.csv").read_bytes(), "text/csv"),
    }


@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    """Point the app at empty temp data/artifact dirs so tests never touch real project files."""
    data, art = tmp_path / "processed", tmp_path / "artifacts"
    data.mkdir(); art.mkdir()
    monkeypatch.setattr(main, "DATA_DIR", data)
    monkeypatch.setattr(main, "ARTIFACT_DIR", art)
    main.clear_sales_cache(); main._model_cache.clear()
    yield data, art
    main.clear_sales_cache(); main._model_cache.clear()


@pytest.fixture
def trained_synthetic(tmp_path, isolated_app):
    data, art = isolated_app
    raw = write_synthetic_m5(tmp_path / "src")
    prepare_m5(str(raw), str(data / "sales.csv"), stores=3, products=2)
    train_all(str(data / "sales.csv"), str(art), include_forest=False)
    return data, art


# --- BUG: blocking work inside `async def` froze the event loop during uploads -------------
def test_upload_handlers_are_not_blocking_coroutines():
    assert not inspect.iscoroutinefunction(main.upload)
    assert not inspect.iscoroutinefunction(main.upload_m5)


# --- BUG: unvalidated stores/products query params ------------------------------------------
@pytest.mark.parametrize("qs", ["stores=-1", "stores=0", "products=0", "stores=11", "products=abc"])
def test_m5_upload_rejects_invalid_subset_params(qs, tmp_path):
    raw = write_synthetic_m5(tmp_path / "src")
    r = client.post(f"/api/v1/datasets/m5/upload?{qs}", files=_m5_files(raw), headers=AUTH)
    assert r.status_code == 422


# --- BUG: tests/uploads wrote stray upload_*.csv files into the real code/data/raw ----------
def test_generic_upload_goes_to_configured_raw_dir(isolate_raw_uploads):
    project_raw = CODE_ROOT / "data/raw"
    before = sorted(project_raw.glob("upload_*.csv")) if project_raw.exists() else []
    r = client.post("/api/v1/datasets/upload", files={"file": ("s.csv", io.BytesIO(b"date,sales\n2020-01-01,1\n"), "text/csv")})
    assert r.status_code == 200
    assert len(list(isolate_raw_uploads.glob("upload_*.csv"))) == 1
    after = sorted(project_raw.glob("upload_*.csv")) if project_raw.exists() else []
    assert before == after


# --- BUG: no size limit on generic uploads ---------------------------------------------------
def test_generic_upload_size_limit(monkeypatch, isolate_raw_uploads):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 1)
    big = b"date,sales\n" + b"2020-01-01,1\n" * 90000  # ~1.2 MB
    r = client.post("/api/v1/datasets/upload", files={"file": ("big.csv", io.BytesIO(big), "text/csv")})
    assert r.status_code == 400 and "upload limit" in r.json()["detail"]
    assert list(isolate_raw_uploads.iterdir()) == []  # partial file removed


# --- M5 upload happy path (previously untested end-to-end) ----------------------------------
def test_m5_upload_processes_trains_and_serves_forecasts(tmp_path, isolated_app, isolate_raw_uploads):
    data, art = isolated_app
    raw = write_synthetic_m5(tmp_path / "src")
    r = client.post("/api/v1/datasets/m5/upload?stores=3&products=2", files=_m5_files(raw), headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "processed_and_trained" and body["rows"] == 120 * 3 * 2
    assert (data / "sales.csv").exists() and (art / "evaluation.json").exists()
    assert (isolate_raw_uploads / "calendar.csv").exists()
    assert not list(isolate_raw_uploads.glob("_staging_*"))
    f = client.post("/api/v1/forecasts", json={"product_id": "ITEM_1", "store_id": "TX_1", "horizon": 5})
    assert f.status_code == 200 and len(f.json()["predictions"]) == 5


# --- BUG: a failed M5 upload replaced the raw files before processing (non-atomic) -----------
def test_failed_m5_processing_leaves_live_files_untouched(tmp_path, isolated_app, isolate_raw_uploads):
    data, art = isolated_app
    (data / "sales.csv").write_text("LIVE")
    (art / "evaluation.json").write_text("LIVE")
    (isolate_raw_uploads / "calendar.csv").write_text("LIVE")
    raw = write_synthetic_m5(tmp_path / "src")
    # Passes the header check but breaks processing (melt needs id/dept_id/cat_id/state_id).
    df = pd.read_csv(raw / "sales_train_validation.csv").drop(columns=["dept_id"])
    df.to_csv(raw / "sales_train_validation.csv", index=False)
    r = client.post("/api/v1/datasets/m5/upload?stores=3&products=2", files=_m5_files(raw), headers=AUTH)
    assert r.status_code == 400 and "processing/training failed" in r.json()["detail"]
    assert (data / "sales.csv").read_text() == "LIVE"
    assert (art / "evaluation.json").read_text() == "LIVE"
    assert (isolate_raw_uploads / "calendar.csv").read_text() == "LIVE"
    assert not list(isolate_raw_uploads.glob("_staging_*"))


# --- BUG: without evaluation.json the forecast returned a misleading 400 --------------------
def test_forecast_without_trained_models_returns_clear_503(tmp_path, isolated_app):
    data, _ = isolated_app
    prepare_m5(str(write_synthetic_m5(tmp_path / "src")), str(data / "sales.csv"), stores=3, products=2)
    r = client.post("/api/v1/forecasts", json={"product_id": "ITEM_1", "store_id": "CA_1"})
    assert r.status_code == 503 and "not been trained" in r.json()["detail"]


# --- BUG: dashboard plotted a single-product forecast against 90-series aggregate history ----
def test_forecast_returns_same_series_history_and_method(trained_synthetic):
    data, _ = trained_synthetic
    r = client.post("/api/v1/forecasts", json={"product_id": "ITEM_2", "store_id": "WI_1", "horizon": 7, "model": "Linear Regression"})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "trained_model_recursive"
    src = pd.read_csv(data / "sales.csv")
    series = src[(src.item_id == "ITEM_2") & (src.store_id == "WI_1")].sort_values("date")
    assert [h["sales"] for h in body["history"]] == series.sales.tail(60).astype(float).tolist()
    assert body["history"][-1]["date"] < body["predictions"][0]["date"]
    b = client.post("/api/v1/forecasts", json={"product_id": "ITEM_2", "store_id": "WI_1", "horizon": 3, "model": "Naive"}).json()
    assert b["method"] == "baseline_rule_recursive"


def test_naive_forecast_is_recursive_not_using_future_actuals(trained_synthetic):
    data, _ = trained_synthetic
    src = pd.read_csv(data / "sales.csv")
    last = float(src[(src.item_id == "ITEM_1") & (src.store_id == "CA_1")].sort_values("date").sales.iloc[-1])
    r = client.post("/api/v1/forecasts", json={"product_id": "ITEM_1", "store_id": "CA_1", "horizon": 10, "model": "Naive"})
    assert [p["predicted_sales"] for p in r.json()["predictions"]] == [last] * 10


def test_forecast_is_deterministic_and_model_is_cached(trained_synthetic):
    req = {"product_id": "ITEM_1", "store_id": "CA_1", "horizon": 14, "model": "Decision Tree"}
    a = client.post("/api/v1/forecasts", json=req).json()["predictions"]
    b = client.post("/api/v1/forecasts", json=req).json()["predictions"]
    assert a == b
    _, art = trained_synthetic
    assert main.load_model(art / "decision_tree.joblib") is main.load_model(art / "decision_tree.joblib")


# --- Health reports readiness -----------------------------------------------------------------
def test_health_reports_readiness(isolated_app):
    r = client.get("/api/v1/health").json()
    assert r["status"] == "ok" and r["data_loaded"] is False and r["models_trained"] is False


# --- BUG: dashboard inserted data-derived strings via innerHTML without escaping (XSS) -------
def test_dashboard_escapes_dynamic_values_and_has_empty_state():
    html = client.get("/").text
    assert "const esc=" in html
    assert "`<option>${x}</option>`" not in html
    assert "${esc(x.model)}" in html
    assert 'id="notice"' in html and "No processed dataset is loaded yet" in html
    assert "Historical (aggregate)" not in html
