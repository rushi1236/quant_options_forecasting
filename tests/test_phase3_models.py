# tests/test_phase3_models.py

import pytest
from pathlib import Path
import pandas as pd

PRED_PATH = Path("data/processed/underlying_price_with_preds.pkl")
REQUIRED_COLS = ['pred_rf', 'pred_xgb', 'pred_lr']

def test_phase3_file_exists():
    assert PRED_PATH.exists(), f"{PRED_PATH} not found. Run Phase 3 script first."

def test_phase3_required_columns_present():
    df = pd.read_pickle(PRED_PATH)
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    assert not missing, f"Missing prediction columns: {missing}"

def test_phase3_no_nans_in_predictions():
    df = pd.read_pickle(PRED_PATH)
    for col in REQUIRED_COLS:
        assert df[col].notna().all(), f"NaNs found in {col}"

def test_phase3_predictions_reasonable():
    df = pd.read_pickle(PRED_PATH)
    for col in REQUIRED_COLS:
        assert df[col].between(-1, 1).any(), f"Values in {col} seem out of expected range"
