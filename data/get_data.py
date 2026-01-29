#!/usr/bin/env python3
"""
Script to download stock price data and compute technical features.

This script uses the `yfinance` library to fetch daily OHLCV data for a
user‑specified list of tickers.  It then leverages the feature
engineering utilities defined in `utils.feature_engineering` to
transform the raw data into a tabular format suitable for model
training or inference.  The resulting dataset includes momentum,
volatility and volume indicators as well as a forward 20‑day return
target.  The file is saved as a CSV in the location provided via the
`--output` argument.

Usage:
    python data/get_data.py --tickers "AAPL MSFT GOOGL" \
                            --start 2018-01-01 --end 2025-12-31 \
                            --output data/features.csv

Note: You can omit the `--end` argument to download up to the
current date.  If you run this script inside the provided
repository, the default values will build a feature set covering many
liquid US equities.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import yfinance as yf

from utils.feature_engineering import compute_features


def download_price_data(tickers: List[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily price data for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        Symbols understood by Yahoo! Finance.
    start : str
        Start date in ``YYYY-MM-DD`` format (inclusive).
    end : str or None
        End date in ``YYYY-MM-DD`` format (exclusive).  If omitted,
        defaults to today.

    Returns
    -------
    pandas.DataFrame
        A multi‑index DataFrame with tickers as the top level and OHLCV
        fields as the second level.
    """
    # yfinance supports passing a space‑separated string of tickers.  The
    # ``group_by='ticker'`` argument ensures the returned DataFrame uses
    # a multi‑index of (ticker, field).
    price_data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=True,
    )
    return price_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download stock data and compute features")
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL MSFT GOOGL AMZN META NVDA TSLA JNJ V WMT JPM PG UNH HD MA XOM PFE KO PEP",
        help="Space‑separated list of ticker symbols to download",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date for the historical download (YYYY‑MM‑DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date (YYYY‑MM‑DD) – defaults to today",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/features.csv",
        help="Output CSV path for the computed features",
    )
    args = parser.parse_args()

    tickers = args.tickers.split()
    end_date = args.end

    print(f"Downloading data for {len(tickers)} tickers from {args.start} to {end_date or 'today'}…")
    price_data = download_price_data(tickers, args.start, end_date)
    print("Computing features…")
    features_df = compute_features(price_data)

    # Ensure the output directory exists
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path} ({len(features_df)} rows)")


if __name__ == "__main__":
    main()