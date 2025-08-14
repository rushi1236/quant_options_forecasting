import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed/underlying_price.pkl")

REQUIRED_FEATURES = [
    "abs_log_ret",
    "ret_3d",
    "ret_std_5d",
    "hv_diff",
    "day_of_week",
    "hv10_fwd"
]

def test_phase2_file_exists():
    assert PROCESSED_PATH.exists(), f"{PROCESSED_PATH} not found. Run Phase 1 and Phase 2 scripts first."

def test_phase2_required_columns_present():
    df = pd.read_pickle(PROCESSED_PATH)
    missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
    assert not missing, f"Missing columns after Phase 2: {missing}"

def test_phase2_no_nans_in_required_columns():
    df = pd.read_pickle(PROCESSED_PATH)
    for col in REQUIRED_FEATURES:
        assert df[col].notna().all(), f"NaNs found in column: {col}"

def test_phase2_shape_reasonable():
    df = pd.read_pickle(PROCESSED_PATH)
    assert len(df) > 1000, f"Dataset too small: {len(df)} rows"
