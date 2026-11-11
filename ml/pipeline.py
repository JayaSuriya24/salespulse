from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

EXPECTED = {"sales_train_validation.csv", "calendar.csv", "sell_prices.csv"}


def extract_m5(zip_path: str, raw_dir: str = "data/raw") -> dict:
    raw = Path(raw_dir); raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        missing = EXPECTED - names
        if missing:
            raise ValueError(f"M5 archive is missing: {', '.join(sorted(missing))}")
        saved = {}
        for name in EXPECTED:
            target = raw / name
            with z.open(name) as src, target.open("wb") as dst:
                dst.write(src.read())
            saved[name] = str(target)
    return saved


def prepare_m5(raw_dir: str = "data/raw", out_path: str = "data/processed/sales.csv",
               stores: int = 3, products: int = 30) -> pd.DataFrame:
    raw = Path(raw_dir)
    missing = [f for f in EXPECTED if not (raw / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required M5 source file(s) in '{raw_dir}': {', '.join(sorted(missing))}")
    sales_path = raw / "sales_train_validation.csv"
    # Read identifier columns first, then stream the wide file in small chunks.
    identifiers = pd.read_csv(sales_path, usecols=["item_id", "store_id"])
    selected_stores = identifiers.store_id.drop_duplicates().head(stores).tolist()
    selected_items = identifiers.item_id.drop_duplicates().head(products).tolist()
    selected_parts = []
    for chunk in pd.read_csv(sales_path, chunksize=200):
        part = chunk[chunk.store_id.isin(selected_stores) & chunk.item_id.isin(selected_items)]
        if not part.empty: selected_parts.append(part)
    if not selected_parts: raise ValueError("No matching M5 store/product rows found")
    sales = pd.concat(selected_parts, ignore_index=True)
    calendar = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    price_parts = []
    for chunk in pd.read_csv(raw / "sell_prices.csv", chunksize=100000):
        part = chunk[chunk.store_id.isin(selected_stores) & chunk.item_id.isin(selected_items)]
        if not part.empty: price_parts.append(part)
    prices = pd.concat(price_parts, ignore_index=True) if price_parts else pd.DataFrame(columns=["store_id","item_id","wm_yr_wk","sell_price"])
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                      value_vars=day_cols, var_name="d", value_name="sales")
    long = long.merge(calendar[["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]], on="d", how="left")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long["sales"] = pd.to_numeric(long.sales, errors="coerce").fillna(0).clip(lower=0)
    long["sell_price"] = pd.to_numeric(long.sell_price, errors="coerce")
    long["event_flag"] = long.event_name_1.notna().astype(int)
    long["is_weekend"] = long.date.dt.dayofweek.isin([5, 6]).astype(int)
    long = long.sort_values(["store_id", "item_id", "date"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_path, index=False)
    return long


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(["store_id", "item_id", "date"]).copy()
    g = x.groupby(["store_id", "item_id"], sort=False).sales
    for lag in (1, 7, 14, 28):
        x[f"lag_{lag}"] = g.shift(lag)
    for window in (7, 14, 28):
        x[f"rolling_mean_{window}"] = g.transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
    x["dayofweek"] = pd.to_datetime(x.date).dt.dayofweek
    x["day"] = pd.to_datetime(x.date).dt.day
    x["month_num"] = pd.to_datetime(x.date).dt.month
    x["weekend"] = (x.dayofweek >= 5).astype(int)
    x["event_flag"] = x.event_flag.fillna(0)
    x["sell_price"] = x.sell_price.fillna(x.groupby(["store_id", "item_id"]).sell_price.transform("median"))
    x["sell_price"] = x.sell_price.fillna(0)
    x["store_code"] = x.store_id.astype("category").cat.codes
    x["item_code"] = x.item_id.astype("category").cat.codes
    return x.dropna(subset=[f"lag_{k}" for k in (1,7,14,28)] + [f"rolling_mean_{k}" for k in (7,14,28)])

FEATURES = ["dayofweek", "day", "month_num", "weekend", "event_flag", "sell_price", "store_code", "item_code",
            "lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"]
