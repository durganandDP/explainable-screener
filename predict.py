#!/usr/bin/env python3
"""
Prediction and explanation utilities for the explainable stock screener.

This script provides a CLI for ranking a list of tickers based on
forward return predictions and generating human‑readable reasons for
each ranking.  The explanations are derived from LightGBM's built‑in
``predict_contrib`` (also aliased ``pred_contrib``) functionality
which returns per‑feature SHAP‑like contributions for each prediction.

Example usage:
    python predict.py --model models/model.txt \
                      --config models/feature_config.json \
                      --tickers "AAPL MSFT GOOGL" \
                      --start 2025-01-01 --end 2026-01-01

The script prints a table of tickers, their predicted 20‑day return
and the top three contributing features.
"""

from __future__ import annotations

import argparse
import datetime
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf

from utils.feature_engineering import compute_features


def load_model(model_path: str) -> lgb.Booster:
    """Load a trained LightGBM model from file."""
    return lgb.Booster(model_file=model_path)


def download_recent_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch recent OHLCV data for the supplied tickers.

    This helper wraps ``yfinance.download`` so the rest of the
    prediction logic can remain agnostic about the data source.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )
    return data


def prepare_latest_features(
    price_data: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """Compute features and keep only the most recent record per ticker.

    Parameters
    ----------
    price_data : pandas.DataFrame
        Multi‑indexed price data as returned by `yfinance.download`.
    feature_cols : list of str
        Columns to retain in the output.

    Returns
    -------
    pandas.DataFrame
        Latest feature snapshot per ticker with columns ``ticker``,
        feature_cols and ``date``.
    """
    features_df = compute_features(price_data)
    # Ensure date column is of datetime type
    features_df["date"] = pd.to_datetime(features_df["date"])
    # Take the last available date per ticker
    latest_df = (
        features_df.sort_values("date")
        .groupby("ticker")
        .tail(1)
        .reset_index(drop=True)
    )
    return latest_df[["ticker", "date"] + feature_cols]


def generate_reasons(
    contributions: np.ndarray, feature_cols: List[str], top_k: int = 3
) -> List[str]:
    """Produce human‑readable reason strings from SHAP contributions.

    Parameters
    ----------
    contributions : np.ndarray
        Array of shape (n_samples, n_features) with per‑feature
        contributions.  Each value represents the additive effect of
        that feature on the predicted return.
    feature_cols : list of str
        Names of the features, aligned with columns in ``contributions``.
    top_k : int, optional
        Number of top contributing features to include in each reason.

    Returns
    -------
    list of str
        Reason strings for each sample.  Each reason lists the
        feature name, whether its contribution is positive or negative
        and the magnitude rounded to four decimals.
    """
    reasons: List[str] = []
    for row in contributions:
        # Identify indices of largest absolute contributions
        order = np.argsort(np.abs(row))[::-1]
        parts = []
        for idx in order[:top_k]:
            feat = feature_cols[idx]
            value = row[idx]
            sign = "positive" if value >= 0 else "negative"
            parts.append(f"{feat.replace('_', ' ')} ({sign}, {value:.4f})")
        reasons.append("; ".join(parts))
    return reasons


def predict_and_explain(
    model: lgb.Booster,
    tickers: List[str],
    start: str,
    end: str,
    feature_cols: List[str],
    top_k: int = 3,
) -> pd.DataFrame:
    """Generate predictions and reasons for a list of tickers.

    Parameters
    ----------
    model : lightgbm.Booster
        Pre‑trained LightGBM model.
    tickers : list of str
        Ticker symbols to evaluate.
    start : str
        Start date for downloading historical data (used to compute features).
    end : str
        End date for downloading historical data.
    feature_cols : list of str
        Names of the features expected by the model.
    top_k : int, optional
        Number of top contributing features to include in the reason.

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns ``ticker``, ``pred_return`` and
        ``reasons`` sorted by descending ``pred_return``.
    """
    price_data = download_recent_data(tickers, start, end)
    if price_data.empty:
        raise ValueError("No data downloaded; please check the ticker symbols and date range.")
    latest_features = prepare_latest_features(price_data, feature_cols)
    # Ensure ordering of columns matches model
    X = latest_features[feature_cols]
    # ``pred_contrib=True`` yields one extra column with the expected value.
    contrib_matrix = model.predict(X, pred_contrib=True)
    # The last column is the expected value; we drop it to isolate feature contributions.
    contributions = contrib_matrix[:, :-1]
    # Predicted returns are simply the sum of contributions per row.
    pred_returns = contributions.sum(axis=1)
    reasons = generate_reasons(contributions, feature_cols, top_k=top_k)
    result_df = latest_features[["ticker"]].copy()
    result_df["pred_return"] = pred_returns
    result_df["reasons"] = reasons
    # Sort highest predicted return first
    result_df = result_df.sort_values("pred_return", ascending=False).reset_index(drop=True)
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stock screener on a set of tickers")
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.txt",
        help="Path to the LightGBM model file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/feature_config.json",
        help="Path to the JSON file specifying feature names",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL MSFT GOOGL AMZN META NVDA TSLA JNJ V WMT JPM PG UNH HD MA XOM PFE KO PEP",
        help="Space‑separated list of ticker symbols to evaluate",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY‑MM‑DD) for computing features; defaults to 200 days before end date",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY‑MM‑DD) for computing features; defaults to today",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top contributing features to display in reasons",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    with open(args.config) as f:
        config = json.load(f)
    feature_cols = config["features"]
    tickers = args.tickers.split()

    today = datetime.date.today()
    end_date = args.end or today.strftime("%Y-%m-%d")
    if args.start:
        start_date = args.start
    else:
        # Choose a start date that ensures we have enough history (at
        # least 120 trading days) to compute six‑month momentum.  We
        # approximate 200 calendar days as ~9.5 months to be safe.
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - datetime.timedelta(days=200)
        start_date = start_dt.strftime("%Y-%m-%d")

    result_df = predict_and_explain(
        model,
        tickers,
        start=start_date,
        end=end_date,
        feature_cols=feature_cols,
        top_k=args.top_k,
    )
    # Print the results in a readable format
    pd.set_option("display.max_colwidth", None)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()