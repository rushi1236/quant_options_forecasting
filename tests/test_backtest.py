# tests/test_backtest.py

import pandas as pd
from pathlib import Path

def test_backtest_results_file_exists():
    path = Path("data/processed/backtest_results.pkl")
    assert path.exists(), f"{path} not found. Run backtest_signals.py first."

def test_backtest_results_equity_columns():
    df = pd.read_pickle("data/processed/backtest_results.pkl")
    assert isinstance(df, pd.DataFrame), "Expected backtest_results.pkl to be a DataFrame."
    
    expected_cols = {"signal_rf_equity", "signal_xgb_equity", "signal_lr_equity", "buy_hold"}
    assert expected_cols.issubset(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"

def test_backtest_results_no_nans():
    df = pd.read_pickle("data/processed/backtest_results.pkl")
    assert not df.isna().any().any(), "NaNs found in backtest results DataFrame."

def test_backtest_index_sorted():
    df = pd.read_pickle("data/processed/backtest_results.pkl")
    assert df.index.is_monotonic_increasing, "Index is not sorted by date."
