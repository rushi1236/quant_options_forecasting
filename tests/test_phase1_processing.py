# tests/test_phase1_processing.py
import os
import pandas as pd

DATA_PATH = "data/processed/underlying_price.pkl"
REQUIRED_COLS = ['Close', 'log_ret', 'hv10', 'hv20', 'hv30']

def test_processed_file_exists():
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found. Run Phase 1 notebook first."

def test_required_columns_present():
    df = pd.read_pickle(DATA_PATH)
    for col in REQUIRED_COLS:
        assert col in df.columns, f"Missing required column: {col}"

def test_no_nans_in_key_columns():
    df = pd.read_pickle(DATA_PATH)
    for col in REQUIRED_COLS:
        nan_count = df[col].isna().sum()
        assert nan_count == 0, f"Column {col} has {nan_count} NaNs."

def test_data_shape_reasonable():
    df = pd.read_pickle(DATA_PATH)
    assert len(df) > 1000, f"Unexpectedly small dataset: {len(df)} rows."
