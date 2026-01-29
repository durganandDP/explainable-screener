"""
Feature engineering utilities for the explainable stock screener.

This module defines helper functions to transform raw OHLCV stock data
into a tabular feature set suitable for training and inference. The
goal is to capture basic momentum, volatility and liquidity signals
without requiring any proprietary data sources.  The features align
with common quantitative finance practices and help the model
generalise across different market regimes.

Functions defined here deliberately avoid external state. All inputs
should be pure dataframes, and the outputs are plain dataframes that
can be persisted to disk.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Tuple

def compute_features(price_data: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for each stock in the provided price data.

    Parameters
    ----------
    price_data : pandas.DataFrame
        A multi-indexed DataFrame returned by ``yfinance.download`` with
        columns structured as a two-level MultiIndex: the top level is
        the ticker symbol and the second level contains OHLCV fields
        such as ``Open``, ``High``, ``Low``, ``Close`` and ``Volume``.

    Returns
    -------
    pandas.DataFrame
        A flat DataFrame with columns:

        - ``date``: trading date
        - ``ticker``: stock symbol
        - ``momentum_1m``: one‑month momentum (20 trading days)
        - ``momentum_3m``: three‑month momentum (60 trading days)
        - ``momentum_6m``: six‑month momentum (120 trading days)
        - ``volatility_20d``: 20‑day rolling standard deviation of daily returns
        - ``volume_ratio_20d``: ratio of current volume to its 20‑day rolling mean
        - ``target_20d``: forward 20‑day return, used as the regression target

    Notes
    -----
    The feature calculations require sufficient lookback periods. Rows
    near the start of the sample, where these rolling windows are not
    fully populated, will be dropped.
    """
    feature_frames: List[pd.DataFrame] = []
    # The top level of price_data columns contains tickers
    tickers: List[str] = list(price_data.columns.levels[0])
    for ticker in tickers:
        # Extract OHLCV data for one ticker and ensure a copy so we can
        # safely add columns without SettingWithCopy warnings.
        df = price_data[ticker].copy()
        # Drop any rows where closing prices or volume are missing.
        df = df.dropna(subset=["Close", "Volume"])

        # Compute daily log return (using natural log to approximate
        # continuously compounded returns).  We avoid division by zero
        # by ensuring positive prices.
        df["return"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))

        # Momentum features: percentage change over different horizons.
        df["momentum_1m"] = df["Close"] / df["Close"].shift(20) - 1.0
        df["momentum_3m"] = df["Close"] / df["Close"].shift(60) - 1.0
        df["momentum_6m"] = df["Close"] / df["Close"].shift(120) - 1.0

        # Volatility: realised standard deviation of daily returns over a 20‑day window.
        df["volatility_20d"] = df["return"].rolling(window=20).std()

        # Volume ratio: current volume relative to its 20‑day moving average.
        volume_ma20 = df["Volume"].rolling(window=20).mean()
        df["volume_ratio_20d"] = df["Volume"] / volume_ma20

        # Forward 20‑day return as the regression target.  We align the
        # label so that each row's target corresponds to future
        # performance; this is critical for avoiding look‑ahead bias.
        df["target_20d"] = df["Close"].shift(-20) / df["Close"] - 1.0

        # Reset the index into columns so we can unify across tickers.
        df = df.reset_index().rename(columns={"index": "date"})
        df["ticker"] = ticker

        # Select only the columns we need and drop rows with any NaNs.
        feature_frames.append(
            df[[
                "date",
                "ticker",
                "momentum_1m",
                "momentum_3m",
                "momentum_6m",
                "volatility_20d",
                "volume_ratio_20d",
                "target_20d",
            ]].dropna()
        )

    # Combine all tickers into a single DataFrame.
    combined = pd.concat(feature_frames, ignore_index=True)
    return combined