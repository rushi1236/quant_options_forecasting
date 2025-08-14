# phase1_data_ingestion.py

import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import download_data, save_df, ensure_dirs

# ======================
# Config
# ======================
TICKER = "^NSEI"  # NIFTY 50
START_DATE = "2018-01-01"
END_DATE = None
OUT_PATH = Path("data/processed/underlying_price.pkl")

# ======================
# 1. Download Data
# ======================
print(f"Downloading {TICKER} from {START_DATE} to {END_DATE or 'today'}...")
df = download_data(TICKER, START_DATE, END_DATE)

# Keep only required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# ======================
# 2. Feature Engineering
# ======================
print("Engineering features...")

# Log returns
df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

# Volatility bands (rolling standard deviation of log returns)
def realized_vol(series, window):
    return series.rolling(window).std() * np.sqrt(252)

df['hv10'] = realized_vol(df['log_ret'], 10)
df['hv20'] = realized_vol(df['log_ret'], 20)
df['hv30'] = realized_vol(df['log_ret'], 30)

# Z-scores of hv10 and hv30
df['hv10_z'] = (df['hv10'] - df['hv10'].rolling(252).mean()) / df['hv10'].rolling(252).std()
df['hv30_z'] = (df['hv30'] - df['hv30'].rolling(252).mean()) / df['hv30'].rolling(252).std()

# Target: hv10 shifted forward (next day’s value)
df['hv10_fwd'] = df['hv10'].shift(-1)

# Drop rows with NaNs
df = df.dropna()

# ======================
# 3. Save processed data
# ======================
ensure_dirs([OUT_PATH])
save_df(df, OUT_PATH)

print(f"✅ Saved processed data to {OUT_PATH}")
print(f"Final shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"NaNs in hv10: {df['hv10'].isna().sum()}")
