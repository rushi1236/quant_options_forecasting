# tests/test_phase4_signals.py

import os
import pandas as pd

SIGNALS_PATH = "data/processed/underlying_price_with_signals.pkl"
REQUIRED_SIGNAL_COLS = ["signal_rf", "signal_xgb", "signal_lr"]

def test_phase4_file_exists():
    """Check that the signals file exists."""
    assert os.path.exists(SIGNALS_PATH), f"{SIGNALS_PATH} not found. Run Phase 4 script first."

def test_phase4_required_columns_present():
    """Check that required signal columns are present."""
    df = pd.read_pickle(SIGNALS_PATH)
    missing_cols = [col for col in REQUIRED_SIGNAL_COLS if col not in df.columns]
    assert not missing_cols, f"Missing signal columns: {missing_cols}"

def test_phase4_no_nans_in_signals():
    """Check that there are no NaNs in signal columns."""
    df = pd.read_pickle(SIGNALS_PATH)
    for col in REQUIRED_SIGNAL_COLS:
        assert df[col].notna().all(), f"NaNs found in {col}"

def test_phase4_signals_valid_values():
    """Check that signals are only in the set {-1, 0, 1}."""
    df = pd.read_pickle(SIGNALS_PATH)
    valid_values = {-1, 0, 1}
    for col in REQUIRED_SIGNAL_COLS:
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset(valid_values), f"Invalid values in {col}: {unique_vals}"
