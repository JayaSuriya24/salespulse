import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from ml.pipeline import extract_m5, prepare_m5

p = argparse.ArgumentParser(description="Prepare Walmart M5 dataset for SalesPulse")
p.add_argument('--zip', required=False, default=None, help="Path to m5-forecasting-accuracy.zip archive")
p.add_argument('--raw-dir', default=str(ROOT / "data/raw"), help="Path to raw CSV files directory")
p.add_argument('--out', default=str(ROOT / "data/processed/sales.csv"), help="Path for processed output CSV")
p.add_argument('--stores', type=int, default=3, help="Number of stores to extract")
p.add_argument('--products', type=int, default=30, help="Number of products to extract")

a = p.parse_args()
if a.zip:
    extract_m5(a.zip, raw_dir=a.raw_dir)
df = prepare_m5(raw_dir=a.raw_dir, out_path=a.out, stores=a.stores, products=a.products)
print(f'Prepared {len(df):,} rows | {df.item_id.nunique()} products | {df.store_id.nunique()} stores | {df.date.min()} to {df.date.max()}')
