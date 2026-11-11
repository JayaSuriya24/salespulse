"""Token authentication for POST /api/v1/datasets/m5/upload (and only that endpoint)."""
import hmac
import io
import logging

import pytest
from fastapi.testclient import TestClient

import app.main as main
from tests.conftest import AUTH, TEST_UPLOAD_TOKEN, write_synthetic_m5

client = TestClient(main.app)
URL = "/api/v1/datasets/m5/upload"


def _bad_files():
    b = lambda: io.BytesIO(b"bad,cols\n1,2\n")
    return {"sales_file": ("s.csv", b(), "text/csv"), "calendar_file": ("c.csv", b(), "text/csv"), "prices_file": ("p.csv", b(), "text/csv")}


@pytest.fixture
def handler_spy(monkeypatch):
    """Fail loudly if processing ever starts: unauthorized requests must never reach the handler."""
    calls = []
    monkeypatch.setattr(main, "prepare_m5", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(RuntimeError("reached handler")))
    return calls


# ---------------------------------------------------------------- unauthorized requests
def test_missing_token_returns_401_with_challenge(handler_spy, isolate_raw_uploads):
    r = client.post(URL, files=_bad_files())
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert "Authentication required" in r.json()["detail"]
    assert handler_spy == [] and list(isolate_raw_uploads.iterdir()) == []  # nothing written, nothing processed


@pytest.mark.parametrize("header", ["Bearer", "Bearer   ", "Basic dXNlcjpwYXNz", "Token abc", TEST_UPLOAD_TOKEN])
def test_malformed_authorization_header_returns_401(header, handler_spy):
    r = client.post(URL, files=_bad_files(), headers={"Authorization": header})
    assert r.status_code == 401 and handler_spy == []


@pytest.mark.parametrize("supplied", ["wrong-token-value-1234567890", TEST_UPLOAD_TOKEN[:-1], TEST_UPLOAD_TOKEN + "x", "A" * 5000])
def test_invalid_token_returns_403(supplied, handler_spy, isolate_raw_uploads):
    r = client.post(URL, files=_bad_files(), headers={"Authorization": f"Bearer {supplied}"})
    assert r.status_code == 403 and r.json() == {"detail": "Invalid upload token."}
    assert handler_spy == [] and list(isolate_raw_uploads.iterdir()) == []


def test_non_ascii_token_is_rejected_cleanly_not_500():
    r = client.post(URL, files=_bad_files(), headers={"Authorization": "Bearer t\xf6k\xe9n-with-latin1-chars".encode("latin-1")})
    assert r.status_code == 403


def test_token_comparison_is_constant_time(monkeypatch):
    seen = []
    real = hmac.compare_digest
    monkeypatch.setattr(main.hmac, "compare_digest", lambda a, b: seen.append((type(a), type(b))) or real(a, b))
    client.post(URL, files=_bad_files(), headers={"Authorization": "Bearer nope-nope-nope-nope"})
    assert seen == [(bytes, bytes)]


def test_unauthorized_requests_do_not_modify_live_data(tmp_path, monkeypatch, isolate_raw_uploads):
    data, art = tmp_path / "d", tmp_path / "a"
    data.mkdir(); art.mkdir()
    (data / "sales.csv").write_text("LIVE"); (art / "evaluation.json").write_text("LIVE")
    monkeypatch.setattr(main, "DATA_DIR", data); monkeypatch.setattr(main, "ARTIFACT_DIR", art)
    raw = write_synthetic_m5(tmp_path / "src")
    files = {"sales_file": ("s.csv", (raw / "sales_train_validation.csv").read_bytes(), "text/csv"),
             "calendar_file": ("c.csv", (raw / "calendar.csv").read_bytes(), "text/csv"),
             "prices_file": ("p.csv", (raw / "sell_prices.csv").read_bytes(), "text/csv")}
    assert client.post(URL, files=files).status_code == 401
    assert client.post(URL, files=files, headers={"Authorization": "Bearer wrong-wrong-wrong-wrong"}).status_code == 403
    assert (data / "sales.csv").read_text() == "LIVE" and (art / "evaluation.json").read_text() == "LIVE"
    assert list(isolate_raw_uploads.iterdir()) == []


# ---------------------------------------------------------------- authorized requests
def test_valid_token_reaches_handler():
    # Bad schema -> the handler ran and validated the files (400), i.e. authentication passed.
    r = client.post(URL, files=_bad_files(), headers=AUTH)
    assert r.status_code == 400 and "must contain" in r.json()["detail"]


def test_valid_token_full_upload_flow(tmp_path, monkeypatch, isolate_raw_uploads):
    data, art = tmp_path / "d", tmp_path / "a"
    data.mkdir(); art.mkdir()
    monkeypatch.setattr(main, "DATA_DIR", data); monkeypatch.setattr(main, "ARTIFACT_DIR", art)
    main.clear_sales_cache()
    raw = write_synthetic_m5(tmp_path / "src")
    files = {"sales_file": ("s.csv", (raw / "sales_train_validation.csv").read_bytes(), "text/csv"),
             "calendar_file": ("c.csv", (raw / "calendar.csv").read_bytes(), "text/csv"),
             "prices_file": ("p.csv", (raw / "sell_prices.csv").read_bytes(), "text/csv")}
    r = client.post(URL + "?stores=3&products=2", files=files, headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "processed_and_trained"
    main.clear_sales_cache()


def test_bearer_scheme_is_case_insensitive():
    r = client.post(URL, files=_bad_files(), headers={"Authorization": f"bearer {TEST_UPLOAD_TOKEN}"})
    assert r.status_code == 400  # authenticated, then rejected by schema validation


def test_token_can_be_rotated_without_restart(monkeypatch):
    monkeypatch.setenv("M5_UPLOAD_TOKEN", "rotated-token-" + "y" * 20)
    assert client.post(URL, files=_bad_files(), headers=AUTH).status_code == 403          # old token revoked
    new = {"Authorization": "Bearer rotated-token-" + "y" * 20}
    assert client.post(URL, files=_bad_files(), headers=new).status_code == 400            # new token accepted


# ---------------------------------------------------------------- fail closed when not configured
@pytest.mark.parametrize("value", [None, "", "   ", "short-token"])
def test_upload_disabled_when_token_unset_empty_or_too_short(monkeypatch, handler_spy, value):
    if value is None:
        monkeypatch.delenv("M5_UPLOAD_TOKEN", raising=False)
    else:
        monkeypatch.setenv("M5_UPLOAD_TOKEN", value)
    # Even a request that "matches" the weak/empty token must be refused.
    headers = {"Authorization": f"Bearer {value}"} if value and value.strip() else {}
    r = client.post(URL, files=_bad_files(), headers=headers)
    assert r.status_code == 503 and "disabled" in r.json()["detail"]
    assert handler_spy == []


# ---------------------------------------------------------------- scope: only the M5 upload is protected
def test_dashboard_and_read_endpoints_stay_public():
    for path in ["/", "/docs", "/openapi.json", "/api/v1/health", "/api/v1/datasets", "/api/v1/options",
                 "/api/v1/models", "/api/v1/experiments", "/api/v1/sales/summary", "/api/v1/sales/trends"]:
        assert client.get(path).status_code not in (401, 403), path


def test_forecast_endpoint_stays_public(monkeypatch):
    monkeypatch.delenv("M5_UPLOAD_TOKEN", raising=False)  # even with no token configured at all
    r = client.post("/api/v1/forecasts", json={"product_id": "X", "store_id": "Y"})
    assert r.status_code not in (401, 403)  # never an auth response (a 404/503 about missing data is fine)


def test_generic_csv_upload_is_not_token_protected():
    r = client.post("/api/v1/datasets/upload", files={"file": ("s.csv", io.BytesIO(b"date,sales\n2020-01-01,1\n"), "text/csv")})
    assert r.status_code == 200


def test_other_methods_on_upload_path_are_not_authenticated_routes():
    assert client.get(URL).status_code == 405  # method not allowed, not an auth response


# ---------------------------------------------------------------- the token is never exposed
def test_configured_token_never_appears_in_responses_or_logs(caplog, handler_spy):
    caplog.set_level(logging.DEBUG)
    secret = TEST_UPLOAD_TOKEN
    bodies = []
    for resp in [client.post(URL, files=_bad_files()),
                 client.post(URL, files=_bad_files(), headers={"Authorization": "Bearer " + "z" * 30}),
                 client.post(URL, files=_bad_files(), headers=AUTH),
                 client.get("/openapi.json"), client.get("/api/v1/health"), client.get("/"), client.get("/docs"),
                 client.get("/api/v1/experiments")]:
        bodies.append(resp.text + str(dict(resp.headers)))
    assert all(secret not in b for b in bodies)
    assert secret not in caplog.text


def test_supplied_wrong_token_is_not_echoed_back(handler_spy):
    supplied = "supplied-wrong-token-ABCDEF123456"
    r = client.post(URL, files=_bad_files(), headers={"Authorization": f"Bearer {supplied}"})
    assert supplied not in r.text and supplied not in str(dict(r.headers))


def test_token_not_read_from_query_string_or_form_fields(handler_spy):
    r = client.post(URL + f"?token={TEST_UPLOAD_TOKEN}&access_token={TEST_UPLOAD_TOKEN}", files=_bad_files())
    assert r.status_code == 401  # only the Authorization header is accepted (URLs end up in access logs)


# ---------------------------------------------------------------- documentation / UI
def test_openapi_documents_bearer_security_on_upload_only():
    spec = client.get("/openapi.json").json()
    assert spec["paths"][URL]["post"].get("security")
    assert "security" not in spec["paths"]["/api/v1/forecasts"]["post"]
    assert "security" not in spec["paths"]["/api/v1/health"]["get"]


def test_dashboard_sends_token_as_bearer_header_and_masks_input():
    html = client.get("/").text
    assert 'id="uploadToken"' in html and 'type="password"' in html
    assert "'Authorization':'Bearer '+tok" in html
    assert "localStorage" not in html and "sessionStorage" not in html  # token is never persisted by the page
    assert "Upload failed: could not complete the request" in html  # a network error must not leave the UI stuck on "Processing"
