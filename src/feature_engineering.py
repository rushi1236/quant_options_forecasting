"""
src/feature_engineering.py

Feature engineering for volatility forecasting.

Adds:
- abs_log_ret
- ret_3d
- ret_std_5d
- hv_diff
- day_of_week
"""

import sys
from pathlib import Path

# --- Patch sys.path so script works standalone or as module ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_df, save_df

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to dataframe.
    """
    df = df.copy()
    df['abs_log_ret'] = df['log_ret'].abs()
    df['ret_3d'] = df['Close'].pct_change(3)
    df['ret_std_5d'] = df['log_ret'].rolling(5).std()
    df['hv_diff'] = df['hv10'] - df['hv20']
    df['day_of_week'] = df.index.dayofweek
    return df.dropna()


if __name__ == "__main__":
    in_path = "data/processed/underlying_price.pkl"
    print(f"Loading data from {in_path}...")
    df = load_df(in_path)

    print("Engineering features...")
    df = engineer_features(df)

    out_path = in_path  # overwrite
    save_df(df, out_path)
    print(f"✅ Saved updated dataset with engineered features to {out_path}")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
