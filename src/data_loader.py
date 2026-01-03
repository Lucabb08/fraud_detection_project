"""Data loading and preprocessing utilities.

Functions:
- load_raw(path): load CSV file into a pandas DataFrame
- load_and_split(path, target='Class', time_col='Time', test_size=0.2): load, sort by time_col, split chronologically and return X_train, X_test, y_train, y_test
- get_num_cat_columns(X): return (num_cols, cat_cols) based on dtypes
"""

import os
from typing import Tuple, List
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV file at `path` and return a DataFrame.

    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}. Put the dataset in data/raw/ or update the path.")
    return pd.read_csv(path)


def load_and_split(path: str, target: str = 'Class', time_col: str = 'Time', test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset from `path`, sort chronologically by `time_col` and split into train/test.

    Returns: X_train, X_test, y_train, y_test
    """
    df = load_raw(path)
    if time_col not in df.columns:
        # If no time column, do a simple random chronological-safe split by index order
        df = df.reset_index(drop=True)
    else:
        df = df.sort_values(time_col)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    X = df.drop(columns=[target])
    y = df[target]

    n = len(df)
    train_end = int((1.0 - test_size) * n)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end:]
    y_test = y.iloc[train_end:]

    return X_train, X_test, y_train, y_test


def get_num_cat_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (num_cols, cat_cols) for a DataFrame X."""
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    return num_cols, cat_cols
