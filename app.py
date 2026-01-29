#!/usr/bin/env python3
"""
Streamlit application for the explainable stock screener.

This web app allows users to enter a list of ticker symbols, select
dates for computing technical features, and view predicted 20‑day
returns alongside explanations for each stock.  The explanations are
derived using LightGBM’s ``predict_contrib`` feature which provides
per‑feature additive contributions (akin to SHAP values).  The app
displays the results in a ranked table and enables iterative
exploration.

To run the app locally, execute ``streamlit run app.py`` from the
repository root.  Ensure you have installed the required packages
using ``pip install -r requirements.txt``.  Note that downloading
fresh price data requires an internet connection.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import streamlit as st
import pandas as pd

from predict import load_model, predict_and_explain


@st.cache_resource
def _load_model_and_config(model_path: str, config_path: str) -> tuple:
    """Load the model and feature configuration once and cache it."""
    model = load_model(model_path)
    with open(config_path) as f:
        config = json.load(f)
    return model, config["features"]


def main() -> None:
    st.set_page_config(page_title="Explainable Stock Screener", layout="wide")
    st.title("📈 Explainable Stock Screener")
    st.markdown(
        """
        This app identifies stocks likely to outperform over the next 20 trading days and
        **explains why**.  Behind the scenes a LightGBM model is trained on
        historical OHLCV data and engineered features.  Predictions are
        accompanied by feature contributions derived from LightGBM’s
        built‑in SHAP algorithm (via the ``predict_contrib`` flag).  For each
        stock, we list the top drivers (e.g., strong 3‑month momentum, low
        volatility) and whether they act positively or negatively on the
        expected return.
        """
    )

    # Load the model and feature names.  Use relative paths so the app
    # works when deployed as a package.
    model_path = Path("models/model.txt")
    config_path = Path("models/feature_config.json")
    if not model_path.exists() or not config_path.exists():
        st.error(
            "Model or configuration file missing. Please train the model first using ``python train.py``."
        )
        return
    model, feature_cols = _load_model_and_config(str(model_path), str(config_path))

    # User inputs
    default_tickers = "AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JNJ, V, WMT, JPM, PG, UNH, HD, MA, XOM, PFE, KO, PEP"
    tickers_input = st.text_input(
        "Enter ticker symbols (comma‑separated)",
        value=default_tickers,
        help="You can specify any symbols supported by Yahoo! Finance."
    )
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date for computing features",
            value=datetime.date.today() - datetime.timedelta(days=200),
            help="The model needs at least 120 days of history to compute six‑month momentum."
        )
    with col2:
        end_date = st.date_input(
            "End date (exclusive)",
            value=datetime.date.today(),
            help="Future dates are not allowed."
        )

    top_k = st.number_input(
        "Number of top features to display in reasons",
        min_value=1,
        max_value=len(feature_cols),
        value=3,
        step=1,
        help="Increasing this will include more explanatory features per stock."
    )

    run_button = st.button("Run Screener")
    if run_button:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            st.warning("Please enter at least one ticker symbol.")
        else:
            try:
                result_df = predict_and_explain(
                    model,
                    tickers,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    feature_cols=feature_cols,
                    top_k=top_k,
                )
                st.subheader("Ranked Results")
                st.dataframe(result_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error running screener: {e}")


if __name__ == "__main__":
    main()