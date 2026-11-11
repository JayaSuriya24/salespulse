import os, json, shutil, gc
from pathlib import Path
from uuid import uuid4
import pandas as pd
import numpy as np
import joblib
import hmac
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Security
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from ml.pipeline import prepare_m5, FEATURES
from ml.train import train_all

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data/processed")))
if not DATA_DIR.is_absolute():
    DATA_DIR = ROOT / DATA_DIR

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(ROOT / "artifacts")))
if not ARTIFACT_DIR.is_absolute():
    ARTIFACT_DIR = ROOT / ARTIFACT_DIR

RAW_DIR = Path(os.getenv("RAW_DIR", str(ROOT / "data/raw")))
if not RAW_DIR.is_absolute():
    RAW_DIR = ROOT / RAW_DIR

# Generic (preview-only) CSV uploads are capped to avoid filling the disk.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ML_MODELS = {"Linear Regression", "Decision Tree", "Random Forest"}

app = FastAPI(title="SalesPulse API", version="1.0.0", description="Walmart M5 retail sales analytics and forecasting")

# ---- M5 upload authentication --------------------------------------------------------------
# Only POST /api/v1/datasets/m5/upload (which replaces the live data and retrains models) is
# protected. Everything else, including the dashboard and forecasts, stays public.
# The token comes from the M5_UPLOAD_TOKEN environment variable (never from code or the repo).
M5_UPLOAD_PATH = "/api/v1/datasets/m5/upload"
MIN_TOKEN_LENGTH = 16


def _configured_upload_token():
    """Read the token on each request so it can be rotated without code changes; None = not usable."""
    token = os.getenv("M5_UPLOAD_TOKEN", "").strip()
    return token if len(token) >= MIN_TOKEN_LENGTH else None


@app.middleware("http")
async def m5_upload_auth(request: Request, call_next):
    # Runs BEFORE the multipart body is read, so unauthenticated clients cannot make the
    # server receive/spool hundreds of MB. Neither the configured nor the supplied token is
    # ever logged or echoed in a response.
    if request.url.path == M5_UPLOAD_PATH and request.method == "POST":
        expected = _configured_upload_token()
        if expected is None:
            return JSONResponse({"detail": "M5 upload is disabled: the server has no valid M5_UPLOAD_TOKEN configured "
                                           f"(minimum {MIN_TOKEN_LENGTH} characters)."}, status_code=503)
        scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not supplied.strip():
            return JSONResponse({"detail": "Authentication required: send 'Authorization: Bearer <token>'."},
                                status_code=401, headers={"WWW-Authenticate": "Bearer"})
        # Constant-time comparison on bytes (compare_digest rejects non-ASCII str).
        if not hmac.compare_digest(supplied.strip().encode("utf-8"), expected.encode("utf-8")):
            return JSONResponse({"detail": "Invalid upload token."}, status_code=403)
    return await call_next(request)


# Documentation only: makes Swagger UI show an "Authorize" button for the protected route.
# Enforcement is done by the middleware above.
_bearer_docs = HTTPBearer(auto_error=False, description="Value of the server's M5_UPLOAD_TOKEN")

_cached_sales = None
_cached_sales_mtime = None
_model_cache = {}


def load_model(path: Path):
    """Load a joblib model once per file version instead of on every forecast request."""
    mtime = path.stat().st_mtime
    cached = _model_cache.get(str(path))
    if cached is None or cached[0] != mtime:
        model = joblib.load(path)
        # Single-row recursive predictions are slower with a thread pool; output is identical.
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        cached = (mtime, model)
        _model_cache[str(path)] = cached
    return cached[1]


def _copy_upload(src, target: Path, max_bytes=None):
    """Stream an uploaded file to disk, optionally enforcing a size limit."""
    written = 0
    with target.open("wb") as out:
        while chunk := src.read(1024 * 1024):
            written += len(chunk)
            if max_bytes is not None and written > max_bytes:
                raise ValueError(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit")
            out.write(chunk)
    return written

def get_processed_sales(parse_dates=False):
    global _cached_sales, _cached_sales_mtime
    p = DATA_DIR / "sales.csv"
    if not p.exists():
        return None
    mtime = (str(p), p.stat().st_mtime)  # key on path too, so a changed DATA_DIR never serves stale data
    if _cached_sales is None or _cached_sales_mtime != mtime:
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"])
        _cached_sales = df
        _cached_sales_mtime = mtime
    return _cached_sales

def clear_sales_cache():
    global _cached_sales, _cached_sales_mtime
    _cached_sales = None
    _cached_sales_mtime = None

class ForecastRequest(BaseModel):
    dataset_id: str = "m5_subset"
    product_id: str
    store_id: str
    horizon: int = Field(default=7, ge=1, le=28)
    model: str = "best"

@app.get("/api/v1/health")
def health():
    # Liveness stays "ok" (used by Render's health check); readiness flags are informational.
    return {
        "status": "ok",
        "service": "SalesPulse",
        "data_loaded": (DATA_DIR / "sales.csv").exists(),
        "models_trained": (ARTIFACT_DIR / "evaluation.json").exists(),
    }

@app.get("/api/v1/datasets")
def datasets():
    d = get_processed_sales()
    if d is None:
        return {"datasets": []}
    return {"datasets": [{
        "id": "m5_subset",
        "rows": len(d),
        "products": int(d.item_id.nunique()),
        "stores": int(d.store_id.nunique()),
        "start": str(d.date.min().date()),
        "end": str(d.date.max().date())
    }]}

# Upload handlers are plain `def` so FastAPI runs the blocking pandas/sklearn work in a
# worker thread instead of freezing the event loop (and every other request) meanwhile.
@app.post("/api/v1/datasets/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / f"upload_{uuid4().hex}.csv"
    try:
        _copy_upload(file.file, target, MAX_UPLOAD_MB * 1024 * 1024)
        sample = pd.read_csv(target, nrows=5)
        if not ({"date", "sales"} <= set(sample.columns) or any(c.startswith("d_") for c in sample.columns)):
            raise ValueError("Expected long format columns (date, sales) or M5 wide format (d_1...d_n)")
        return {
            "status": "uploaded",
            "filename": file.filename,
            "columns": sample.columns.tolist(),
            "preview": sample.fillna("").to_dict(orient="records")
        }
    except Exception as e:
        target.unlink(missing_ok=True)
        raise HTTPException(400, str(e))

@app.post(M5_UPLOAD_PATH, dependencies=[Security(_bearer_docs)])
def upload_m5(sales_file: UploadFile = File(...), calendar_file: UploadFile = File(...), prices_file: UploadFile = File(...),
              stores: int = Query(3, ge=1, le=10, description="Number of M5 stores to keep (M5 has 10)"),
              products: int = Query(30, ge=1, le=3049, description="Number of M5 products to keep (M5 has 3,049)")):
    """Upload the three official M5 CSVs, validate in staging, process a subset, and train models.

    Processing and training run entirely inside the staging directory; raw files, the processed
    dataset and model artifacts are only replaced after both steps succeed, so a failed upload
    never leaves the app with mismatched data and models.
    """
    uploads = {
        "sales_train_validation.csv": sales_file,
        "calendar.csv": calendar_file,
        "sell_prices.csv": prices_file
    }
    for expected_name, f in uploads.items():
        if not f.filename.lower().endswith(".csv"):
            raise HTTPException(400, f"{f.filename} must be a CSV file")

    staging = RAW_DIR / f"_staging_{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for filename, f in uploads.items():
            _copy_upload(f.file, staging / filename)

        # Validate schemas before modifying permanent raw data
        sales_preview = pd.read_csv(staging / "sales_train_validation.csv", nrows=5)
        if not ({"item_id", "store_id"} <= set(sales_preview.columns) and any(c.startswith("d_") for c in sales_preview.columns)):
            raise ValueError("sales_train_validation.csv must contain 'item_id', 'store_id', and day sales columns (d_1, ...)")

        cal_preview = pd.read_csv(staging / "calendar.csv", nrows=5)
        if not {"d", "date", "wm_yr_wk"} <= set(cal_preview.columns):
            raise ValueError("calendar.csv must contain 'd', 'date', and 'wm_yr_wk' columns")

        prices_preview = pd.read_csv(staging / "sell_prices.csv", nrows=5)
        if not {"store_id", "item_id", "wm_yr_wk", "sell_price"} <= set(prices_preview.columns):
            raise ValueError("sell_prices.csv must contain 'store_id', 'item_id', 'wm_yr_wk', and 'sell_price' columns")

        staged_sales = staging / "processed_sales.csv"
        staged_artifacts = staging / "artifacts"
        # Memory: free the in-memory dashboard dataset and the prepared frame before training
        # (peak RSS was ~524 MB on Linux, above Render's 512 MB free tier).
        clear_sales_cache()
        _model_cache.clear()
        prepared = prepare_m5(str(staging), str(staged_sales), stores=stores, products=products)
        summary_counts = {"rows": len(prepared), "products": int(prepared.item_id.nunique()), "stores": int(prepared.store_id.nunique())}
        del prepared
        gc.collect()
        report = train_all(str(staged_sales), str(staged_artifacts), include_forest=False)

        # Commit: everything succeeded, now replace the live files.
        for filename in uploads:
            shutil.move(str(staging / filename), str(RAW_DIR / filename))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_sales), str(DATA_DIR / "sales.csv"))
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        for artifact in staged_artifacts.iterdir():
            shutil.move(str(artifact), str(ARTIFACT_DIR / artifact.name))
        clear_sales_cache()
        _model_cache.clear()
        return {
            "status": "processed_and_trained",
            **summary_counts,
            "evaluation": report
        }
    except Exception as e:
        raise HTTPException(400, f"M5 processing/training failed: {e}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

@app.get("/api/v1/sales/summary")
def summary():
    d = get_processed_sales()
    if d is None:
        raise HTTPException(404, "Dataset not processed. Run scripts/prepare_data.py")
    return {
        "total_sales": float(d.sales.sum()),
        "average_daily_sales": float(d.groupby("date").sales.sum().mean()),
        "products": int(d.item_id.nunique()),
        "stores": int(d.store_id.nunique()),
        "rows": len(d)
    }

@app.get("/api/v1/sales/trends")
def trends():
    d = get_processed_sales()
    if d is None:
        raise HTTPException(404, "Dataset not processed")
    s = d.groupby("date", as_index=False).sales.sum()
    return {"data": [{"date": r.date.strftime("%Y-%m-%d"), "sales": float(r.sales)} for r in s.itertuples()]}

@app.get("/api/v1/models")
def models():
    p = ARTIFACT_DIR / "evaluation.json"
    return json.loads(p.read_text()) if p.exists() else {"models": [], "message": "Train models first"}

@app.get("/api/v1/models/compare")
def compare():
    return models()

@app.get("/api/v1/experiments")
def experiments():
    p = ARTIFACT_DIR / "evaluation.json"
    return {
        "tracking": "Local evaluation record",
        "mlflow_available": __import__("importlib").util.find_spec("mlflow") is not None,
        "evaluation": json.loads(p.read_text()) if p.exists() else None
    }

@app.post("/api/v1/forecasts")
def forecast(req: ForecastRequest):
    d = get_processed_sales()
    if d is None:
        raise HTTPException(404, "Dataset not processed")
    report_path = ARTIFACT_DIR / "evaluation.json"
    series = d[(d.item_id == req.product_id) & (d.store_id == req.store_id)].sort_values("date")
    if series.empty:
        raise HTTPException(404, "Product/store combination not found")

    if not report_path.exists():
        raise HTTPException(503, "Models have not been trained yet. Run scripts/train.py or upload the M5 files.")
    report = json.loads(report_path.read_text())
    name = report.get("best_model") if req.model == "best" else req.model
    available_names = {row["model"] for row in report.get("results", [])}
    if name not in available_names:
        raise HTTPException(400, f"Model '{name}' is not available in the current training run")

    artifact = ARTIFACT_DIR / (name.lower().replace(" ", "_") + ".joblib")
    if name in ML_MODELS and not artifact.exists():
        raise HTTPException(503, "The selected model artifact is missing. Train the models again.")

    model = load_model(artifact) if name in ML_MODELS else None
    hist = series.sales.astype(float).tolist()
    last = series.date.max()
    out = []
    # Category codes must match training (pandas category codes = index in sorted unique values).
    store_code = sorted(d.store_id.astype(str).unique().tolist()).index(req.store_id)
    item_code = sorted(d.item_id.astype(str).unique().tolist()).index(req.product_id)

    # Recursive multi-step forecast; future actual sales are never consumed as inputs.
    for step in range(1, req.horizon + 1):
        date = last + pd.Timedelta(days=step)
        if model is None:
            if name == "Naive":
                value = float(hist[-1])
            elif name == "Seasonal Naive":
                value = float(hist[-7]) if len(hist) >= 7 else float(hist[-1])
            else:  # Moving Average
                value = float(np.mean(hist[-7:])) if len(hist) >= 7 else float(hist[-1])
        else:
            row = series.iloc[-1].copy()
            row["date"] = date
            row["dayofweek"] = date.dayofweek
            row["day"] = date.day
            row["month_num"] = date.month
            row["weekend"] = int(date.dayofweek >= 5)
            row["event_flag"] = 0
            for lag in (1, 7, 14, 28):
                row[f"lag_{lag}"] = hist[-lag] if len(hist) >= lag else hist[-1]
            for w in (7, 14, 28):
                row[f"rolling_mean_{w}"] = float(np.mean(hist[-w:]))
            row["sell_price"] = float(series.sell_price.iloc[-1]) if pd.notna(series.sell_price.iloc[-1]) else 0.0
            row["store_code"] = store_code
            row["item_code"] = item_code
            X_df = pd.DataFrame([row])[FEATURES].astype(float)
            value = max(0.0, float(model.predict(X_df)[0]))

        hist.append(value)
        out.append({"date": date.strftime("%Y-%m-%d"), "predicted_sales": round(value, 3)})

    return {
        "forecast_id": uuid4().hex,
        "dataset_id": req.dataset_id,
        "product_id": req.product_id,
        "store_id": req.store_id,
        "model": name,
        "method": "trained_model_recursive" if model is not None else "baseline_rule_recursive",
        "horizon": req.horizon,
        "predictions": out,
        # Recent observed history of the SAME store/product series (for like-for-like charts).
        "history": [{"date": r.date.strftime("%Y-%m-%d"), "sales": float(r.sales)} for r in series.tail(60).itertuples()]
    }

@app.get("/api/v1/forecasts/{forecast_id}")
def get_forecast(forecast_id: str):
    raise HTTPException(404, "Forecast IDs are returned with each request; persistent forecast history is not enabled in this MVP")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse('''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>SalesPulse</title><link rel="icon" href="data:,"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>
    :root{--navy:#101d36;--blue:#284b8f;--orange:#f29b38;--bg:#f4f6fa;--muted:#667085}*{box-sizing:border-box}body{margin:0;background:var(--bg);font:15px Inter,system-ui;color:#172033}.top{background:var(--navy);color:white;padding:22px 5%;display:flex;justify-content:space-between;align-items:center}.brand{font-size:23px;font-weight:750;letter-spacing:.2px}.brand span{color:var(--orange)}.wrap{max-width:1200px;margin:28px auto;padding:0 22px}.sub{color:var(--muted);margin:5px 0 22px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,.panel{background:white;border:1px solid #e7eaf0;border-radius:14px;padding:18px;box-shadow:0 4px 16px #15294b08}.metric{font-size:26px;font-weight:750;margin-top:8px}.label{color:var(--muted);font-size:13px}.grid{display:grid;grid-template-columns:1.5fr 1fr;gap:16px;margin-top:18px}.panel h3{margin:0 0 14px}.controls{display:flex;gap:10px;flex-wrap:wrap}select,input,button{padding:11px 12px;border-radius:8px;border:1px solid #d8deea;background:white;font:inherit}button{background:var(--blue);color:white;border:0;font-weight:650;cursor:pointer}button:hover{opacity:.92}input[type=file]{max-width:100%;min-width:0}.notice{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:10px;padding:12px 14px;margin-bottom:16px}.result{margin-top:14px;color:#344054}.foot{color:#8a93a3;text-align:center;padding:28px;font-size:12px}@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.top{padding:18px 22px}}
    </style></head><body><header class="top"><div class="brand">Sales<span>Pulse</span></div><div>Retail Intelligence · M5 Forecasting</div></header><main class="wrap"><h1>Retail sales at a glance</h1><p class="sub">Explore historical Walmart sales and generate data-driven forecasts.</p><div id="notice" class="notice" hidden></div><section class="cards"><div class="card"><div class="label">Total historical sales</div><div class="metric" id="total">—</div></div><div class="card"><div class="label">Average daily sales</div><div class="metric" id="avg">—</div></div><div class="card"><div class="label">Products analyzed</div><div class="metric" id="products">—</div></div><div class="card"><div class="label">Stores analyzed</div><div class="metric" id="stores">—</div></div></section><section class="grid"><div class="panel"><h3>Historical daily sales</h3><div id="trend" style="height:320px"></div></div><div class="panel"><h3>Model performance</h3><div id="models">Run the training script to populate measured model metrics.</div></div></section><section class="panel" style="margin-top:16px"><h3>Load Walmart M5 data</h3><p class="label">Upload the three official CSV files. Processing selects a manageable subset and trains the forecasting models.</p><div class="controls"><input type="file" id="salesFile" accept=".csv"><input type="file" id="calendarFile" accept=".csv"><input type="file" id="pricesFile" accept=".csv"><input type="password" id="uploadToken" placeholder="Upload token" autocomplete="off" aria-label="Upload token" style="min-width:0"><button onclick="uploadM5()">Process dataset & train</button></div><div id="uploadStatus" class="result"></div></section><section class="panel" style="margin-top:16px"><h3>Generate a sales forecast</h3><div class="controls"><select id="store"><option>Loading stores…</option></select><select id="product"><option>Loading products…</option></select><select id="horizon"><option value="7">Next 7 days</option><option value="28">Next 28 days</option></select><select id="model"><option value="best">Best model</option><option>Linear Regression</option><option>Random Forest</option><option>Decision Tree</option><option>Moving Average</option><option>Seasonal Naive</option><option>Naive</option></select><button onclick="forecast()">Generate forecast</button></div><div id="forecast" class="result"></div></section><div class="foot">SalesPulse · Walmart M5 subset · Forecast estimates are not guarantees.</div></main><script>
    let trendData = null;
    const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const hasPlotly=()=>typeof Plotly!=='undefined';
    function showNotice(msg){let n=document.getElementById('notice');n.textContent=msg;n.hidden=!msg}
    async function get(u){let r=await fetch(u);if(!r.ok)throw Error(await r.text());return r.json()}
    async function init(){
        // Each section loads independently so one failure (e.g. chart CDN offline) does not blank the rest.
        showNotice('');
        try{
            let s=await get('/api/v1/sales/summary');
            total.textContent=Math.round(s.total_sales).toLocaleString();
            avg.textContent=s.average_daily_sales.toFixed(1);
            products.textContent=s.products;
            stores.textContent=s.stores;
        }catch(e){console.warn(e);showNotice('No processed dataset is loaded yet. Upload the three official M5 CSV files below (or run scripts/prepare_data.py and scripts/train.py) to populate the dashboard.')}
        try{
            let t=await get('/api/v1/sales/trends');
            trendData = t.data;
            if(hasPlotly()) Plotly.newPlot('trend',[{x:t.data.map(x=>x.date),y:t.data.map(x=>x.sales),name:'Historical',type:'scatter',mode:'lines',line:{color:'#284b8f',width:2},fill:'tozeroy',fillcolor:'rgba(40,75,143,.08)'}],{margin:{l:45,r:15,t:8,b:40},paper_bgcolor:'white',plot_bgcolor:'white',yaxis:{title:'Units sold (all selected series)'},xaxis:{gridcolor:'#eef1f6'},showlegend:false},{responsive:true});
            else document.getElementById('trend').textContent='Chart library could not be loaded (offline?). Data is still available via /api/v1/sales/trends.';
        }catch(e){console.warn(e);document.getElementById('trend').textContent='No historical data available yet.'}
        try{
            let m=await get('/api/v1/models');
            if(m.results && m.results.length){
                models.innerHTML='<table width="100%"><tr><th align="left">Model</th><th>MAE</th><th>RMSE</th><th>WAPE</th></tr>'+m.results.map(x=>`<tr><td>${esc(x.model)}</td><td align="right">${x.MAE.toFixed(2)}</td><td align="right">${x.RMSE.toFixed(2)}</td><td align="right">${x.WAPE ? x.WAPE.toFixed(1)+'%' : '—'}</td></tr>`).join('')+'</table><p class="label">Chronological 28-day holdout, one-step evaluation (lags from observed history).</p>';
                let sel=document.getElementById('model'),cur=sel.value;
                sel.innerHTML='<option value="best">Best model ('+esc(m.best_model)+')</option>'+m.results.map(x=>`<option value="${esc(x.model)}">${esc(x.model)}</option>`).join('');
                if([...sel.options].some(o=>o.value===cur)) sel.value=cur;
            }
        }catch(e){console.warn(e)}
    }
    async function uploadM5(){
        let a=document.getElementById('salesFile').files[0],b=document.getElementById('calendarFile').files[0],c=document.getElementById('pricesFile').files[0];
        let status=document.getElementById('uploadStatus');
        if(!a||!b||!c){status.textContent='Select all three official M5 CSV files.';return}
        let f=new FormData();
        f.append('sales_file',a);f.append('calendar_file',b);f.append('prices_file',c);
        let tok=document.getElementById('uploadToken').value.trim();
        if(!tok){status.textContent='Enter the upload token (the M5_UPLOAD_TOKEN configured on the server).';return}
        status.textContent='Processing and training models. This may take a few minutes…';
        let r,d;
        try{
            r=await fetch('/api/v1/datasets/m5/upload',{method:'POST',body:f,headers:{'Authorization':'Bearer '+tok}});
            d=await r.json();
        }catch(e){status.textContent='Upload failed: could not complete the request to the server. Please try again.';return}
        status.textContent=r.ok?`Ready: ${d.rows.toLocaleString()} rows, ${d.products} products, ${d.stores} stores. Best model: ${d.evaluation.best_model}. Refreshing dashboard…`:d.detail;
        if(r.ok){await init();await loadOptions()}
    }
    async function forecast(){
        let product=document.getElementById('product').value,store=document.getElementById('store').value;
        let fcDiv=document.getElementById('forecast');
        if(!product||!store){fcDiv.textContent='Prepare the dataset first.';return}
        fcDiv.textContent='Generating recursive forecast…';
        let r=await fetch('/api/v1/forecasts',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({product_id:product,store_id:store,horizon:+document.getElementById('horizon').value,model:document.getElementById('model').value})
        });
        let d=await r.json();
        if(!r.ok){let msg=typeof d.detail==='string'?d.detail:'Forecast failed: invalid request';fcDiv.innerHTML=`<span style="color:#b42318">${esc(msg)}</span>`;return}
        fcDiv.innerHTML=`<b>${esc(d.model)}</b> recursive forecast for ${esc(product)} @ ${esc(store)} (${d.horizon} days):<br>`+d.predictions.map(x=>`<b>${x.date}</b>: ${x.predicted_sales.toFixed(1)} units`).join(' &nbsp; · &nbsp; ');
        if(hasPlotly() && d.predictions.length){
            // Plot the selected series' own history so forecast and history share the same scale.
            let h = d.history || [];
            let histTrace = {x:h.map(x=>x.date), y:h.map(x=>x.sales), name:`Actual (${product} @ ${store})`, type:'scatter', mode:'lines', line:{color:'#284b8f',width:2}};
            let fcTrace = {x:d.predictions.map(x=>x.date), y:d.predictions.map(x=>x.predicted_sales), name:`${d.model} forecast`, type:'scatter', mode:'lines+markers', line:{color:'#f29b38',width:2.5,dash:'dot'}};
            Plotly.newPlot('trend', [histTrace, fcTrace], {margin:{l:45,r:15,t:8,b:40},paper_bgcolor:'white',plot_bgcolor:'white',yaxis:{title:'Units sold (selected product)'},xaxis:{gridcolor:'#eef1f6'},legend:{orientation:'h',y:1.15}}, {responsive:true});
        }
    }
    async function loadOptions(){
        let r=await fetch('/api/v1/options');
        if(r.ok){
            let d=await r.json();
            store.innerHTML=d.stores.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
            product.innerHTML=d.products.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
        }
    }
    init();loadOptions();
    </script></body></html>''')

@app.get("/api/v1/options")
def options():
    d = get_processed_sales()
    if d is None:
        return {"stores": [], "products": []}
    return {
        "stores": sorted(d.store_id.unique().tolist()),
        "products": sorted(d.item_id.unique().tolist())
    }
