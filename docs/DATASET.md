# Dataset documentation

## Source
Walmart M5 Forecasting Accuracy competition: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

## Files used
- `sales_train_validation.csv`: product/store daily unit sales history.
- `calendar.csv`: date, weekday, month/year, event and SNAP indicators.
- `sell_prices.csv`: weekly item/store selling prices.

## Processing
The pipeline reads only selected rows from the wide sales and price files in chunks, selects the first three stores and first thirty products in source order by default, melts `d_1...d_1913` into date-indexed rows, joins calendar and weekly prices, and keeps valid zero-sales records. The current processed subset contains 172,170 observations across 30 products and 3 stores, covering 2011-01-29 to 2016-04-24.

## Data governance
The raw M5 files are not included in this project package or Git repository. Users must obtain the files from Kaggle and comply with the competition's data terms. Processed files and trained artifacts are generated locally at runtime.
