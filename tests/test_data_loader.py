# tests/test_data_loader.py
import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import save_df, load_df

def test_save_and_load_roundtrip(tmp_path):
    # create a tiny synthetic DataFrame
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({
        "Open": [1.0, 1.1, 1.2],
        "High": [1.2, 1.3, 1.25],
        "Low":  [0.9, 1.05, 1.1],
        "Close":[1.05, 1.2, 1.15],
        "Volume":[100, 110, 90]
    }, index=idx)
    out = tmp_path / "test.pkl"
    save_df(df, out)
    df2 = load_df(out)
    pd.testing.assert_frame_equal(df, df2)
