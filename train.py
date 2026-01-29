#!/usr/bin/env python3
"""
Training script for the explainable stock screener.

This module reads a pre‑computed feature dataset (as created by
``data/get_data.py``), splits it into training and validation sets,
trains a LightGBM regression model to forecast forward 20‑day returns,
and persists both the model and associated configuration to disk.

By default the script avoids look‑ahead bias by using a time‑ordered
split: the earliest 80% of the data are used for training and the
remaining 20% for validation.  Users may modify this behaviour as
desired.  Training metrics are printed to the console.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the feature dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the features CSV.  The file must contain the
        engineered features and the target column ``target_20d``.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset as a DataFrame.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def time_train_test_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into chronological training and validation sets.

    Data are first sorted by date to simulate an expanding window.

    Parameters
    ----------
    df : pandas.DataFrame
        The feature dataframe with a ``date`` column.
    test_fraction : float, optional
        Proportion of the data reserved for validation.  Defaults to 0.2.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The training and validation dataframes.
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_index = int((1.0 - test_fraction) * len(df_sorted))
    train_df = df_sorted.iloc[:split_index]
    val_df = df_sorted.iloc[split_index:]
    return train_df, val_df


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], target_col: str) -> lgb.Booster:
    """Train a LightGBM regression model.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training set.
    val_df : pandas.DataFrame
        Validation set.
    feature_cols : list of str
        Names of the feature columns.
    target_col : str
        Name of the target column.

    Returns
    -------
    lightgbm.Booster
        Trained LightGBM model.
    """
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    # Convert to LightGBM datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # Basic parameter set.  We favour conservative settings to reduce
    # overfitting on noisy financial data.
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
    }

    # Train with early stopping on the validation set
    # Use a callback for early stopping rather than the deprecated
    # ``early_stopping_rounds`` argument to maintain compatibility
    # across LightGBM versions.  Early stopping halts training when
    # the validation metric has not improved for a specified
    # number of rounds.
    callbacks = [lgb.early_stopping(stopping_rounds=25, verbose=False)]
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["validation"],
        callbacks=callbacks,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stock screener model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/features.csv",
        help="Path to the features CSV file",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="models/model.txt",
        help="Output path for the trained LightGBM model",
    )
    parser.add_argument(
        "--output_config",
        type=str,
        default="models/feature_config.json",
        help="Output path for a JSON config with feature metadata",
    )
    args = parser.parse_args()

    df = load_dataset(args.data)

    # Define the feature set explicitly.  This list must align with
    # ``compute_features`` from utils.feature_engineering.
    feature_cols = [
        "momentum_1m",
        "momentum_3m",
        "momentum_6m",
        "volatility_20d",
        "volume_ratio_20d",
    ]
    target_col = "target_20d"

    # Drop rows with any NA values in the selected columns
    df = df.dropna(subset=feature_cols + [target_col])

    train_df, val_df = time_train_test_split(df)
    print(f"Training on {len(train_df)} samples; validating on {len(val_df)} samples.")
    model = train_model(train_df, val_df, feature_cols, target_col)

    # Persist the model and configuration
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save_model(args.output_model)
    print(f"Model saved to {args.output_model}")

    config = {"features": feature_cols, "target": target_col}
    os.makedirs(os.path.dirname(args.output_config), exist_ok=True)
    with open(args.output_config, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Feature config saved to {args.output_config}")


if __name__ == "__main__":
    main()