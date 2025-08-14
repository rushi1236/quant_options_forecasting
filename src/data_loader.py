"""
src/data_loader.py

Simple utilities to download, save and load OHLCV data.

Functions:
- download_data: fetches from yfinance (returns a DataFrame)
- save_df: saves DataFrame to a pickle (creates parent dirs)
- load_df: loads a DataFrame from a pickle
- ensure_dirs: helper to create directories

Note: network calls (yfinance) are only in download_data; unit tests avoid network by using save/load.
"""
from pathlib import Path
from typing import Optional, Union, List
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def download_data(ticker: str = "^NSEI",
                  start: str = "2018-01-01",
                  end: Optional[str] = None) -> pd.DataFrame:
    """
    Download OHLCV data for `ticker` using yfinance.
    Returns a DataFrame with Date index and columns ['Open','High','Low','Close','Adj Close','Volume'] (or similar).
    """
    logger.info("Downloading %s from %s to %s", ticker, start, end)
    df = yf.download(ticker, start=start, end=end)
    # If yfinance returns MultiIndex columns (common), flatten to level 0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df


def save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save DataFrame to a pickle file. Creates parent directory if missing.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p)
    logger.info("Saved DataFrame to %s", p)


def load_df(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a DataFrame from a pickle file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    df = pd.read_pickle(p)
    return df


def ensure_dirs(paths: List[Union[str, Path]]) -> None:
    """
    Create directories for a list of paths (file or dir). If a file path is provided,
    only the parent directory is created.
    """
    for p in paths:
        path = Path(p)
        to_create = path if path.is_dir() else path.parent
        to_create.mkdir(parents=True, exist_ok=True)


# small CLI helper for dev usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and save OHLCV data using yfinance")
    parser.add_argument("--ticker", default="^NSEI")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="data/raw/underlying_raw.pkl")
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    save_df(df, args.out)
    print(f"Downloaded {len(df)} rows and saved to {args.out}")
