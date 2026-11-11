from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(divide by zero|overflow|invalid value) encountered in matmul.*")
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ml.pipeline import add_features, FEATURES

MODELS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=16, min_samples_leaf=3, n_jobs=-1, random_state=42),
}

def metrics(actual, predicted):
    a=np.asarray(actual,dtype=float); p=np.asarray(predicted,dtype=float)
    denom=np.abs(a).sum()
    return {"MAE":float(mean_absolute_error(a,p)),"RMSE":float(np.sqrt(mean_squared_error(a,p))),
            "MAPE":float(np.mean(np.abs((a-p)[a!=0]/a[a!=0]))*100) if np.any(a!=0) else None,
            "WAPE":float(np.abs(a-p).sum()/denom*100) if denom else None}

def train_all(csv_path="data/processed/sales.csv", artifact_dir="artifacts", include_forest=True):
    df=pd.read_csv(csv_path,parse_dates=["date"])
    feat=add_features(df)
    dates=sorted(feat.date.unique())
    test_start=pd.Timestamp(dates[-28])
    train=feat[feat.date < test_start]
    test=feat[feat.date >= test_start]
    if len(train)<100: raise ValueError("Not enough training rows after feature generation")
    Path(artifact_dir).mkdir(parents=True,exist_ok=True)
    results=[]; best=None; best_mae=float("inf")
    selected_models = MODELS if include_forest else {k:v for k,v in MODELS.items() if k != "Random Forest"}
    X_train = train[FEATURES].astype(float)
    X_test = test[FEATURES].astype(float)
    for name, model in selected_models.items():
        start=time.time(); model.fit(X_train,train.sales)
        pred=model.predict(X_test).clip(0)
        score=metrics(test.sales,pred); duration=time.time()-start
        row={"model":name,**score,"training_seconds":round(duration,2),"test_start":str(test_start.date()),"test_end":str(feat.date.max().date())}
        results.append(row)
        joblib.dump(model,Path(artifact_dir)/(name.lower().replace(" ","_") + ".joblib"))
        if score["MAE"]<best_mae: best_mae=score["MAE"]; best=name
    # Baselines evaluated on the same chronological holdout rows.
    for name, pred in [("Naive",test.lag_1),("Seasonal Naive",test.lag_7),("Moving Average",test.rolling_mean_7)]:
        results.append({"model":name,**metrics(test.sales,pred),"training_seconds":0.0,
                        "test_start":str(test_start.date()),"test_end":str(feat.date.max().date())})
    results=sorted(results,key=lambda r:r["MAE"])
    best_model=next((r["model"] for r in results if r["model"] in selected_models),None)
    report={"best_model":best_model,"evaluation":"Chronological holdout; final 28 days; row-wise one-step evaluation using observed lag history.",
            "training_rows":int(len(train)),"test_rows":int(len(test)),"products":int(df.item_id.nunique()),
            "stores":int(df.store_id.nunique()),"date_min":str(df.date.min().date()),"date_max":str(df.date.max().date()),"results":results}
    (Path(artifact_dir)/"evaluation.json").write_text(json.dumps(report,indent=2))
    (Path(artifact_dir)/"feature_columns.json").write_text(json.dumps(FEATURES,indent=2))
    return report
