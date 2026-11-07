"""Load the raw NASDAQ-100 price file and run basic data checks.

Usage (from the project folder):
    python -m src.load_data                 # reads data/NASDAQ100_Historical_Data.csv
    python -m src.load_data some/file.csv   # reads another path
"""
import sys

import pandas as pd

from src.config import DATA_CSV

PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]
EXPECTED_COLS = ["Ticker", "Date"] + PRICE_COLS + ["Volume"]


def load_prices(path=DATA_CSV):
    """Read the CSV, parse dates, and sort by (Ticker, Date).

    Sorting matters: every later per-ticker calculation (features, next-day
    label) assumes each ticker's rows are in date order.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    missing_cols = set(EXPECTED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing columns: {sorted(missing_cols)}")
    df = df[EXPECTED_COLS]  # fixed column order
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def check_prices(df):
    """Print basic data-quality checks. Returns a dict of the counts."""
    n_rows = len(df)
    n_tickers = df["Ticker"].nunique()
    n_dup = int(df.duplicated(["Ticker", "Date"]).sum())
    n_nonpos = int((df[PRICE_COLS] <= 0).any(axis=1).sum())  # rows with any price <= 0
    na_per_col = df.isna().sum()
    n_missing = int(na_per_col.sum())

    print(f"Total rows:        {n_rows:,}")
    print(f"Tickers:           {n_tickers}")
    print(f"Date range:        {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Duplicate (Ticker, Date) rows: {n_dup}")
    print(f"Rows with a non-positive price: {n_nonpos}")
    print(f"Missing values:    {n_missing}")
    if n_missing:
        print(na_per_col[na_per_col > 0].to_string())

    # Per-ticker first date, last date, and number of rows.
    span = df.groupby("Ticker")["Date"].agg(start="min", end="max", rows="size")
    span["start"] = span["start"].dt.date
    span["end"] = span["end"].dt.date
    print("\nPer-ticker start / end dates (sorted by start date):")
    print(span.sort_values(["start", "rows"]).to_string())

    return {"rows": n_rows, "tickers": n_tickers, "duplicates": n_dup,
            "nonpositive_price_rows": n_nonpos, "missing": n_missing}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_CSV
    print(f"Reading {path}\n")
    check_prices(load_prices(path))
