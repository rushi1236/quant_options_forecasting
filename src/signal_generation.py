import pandas as pd
from src.data_loader import load_df, save_df
from src.config import SIGNAL_METHOD, THRESHOLD_METHOD, THRESHOLD_QUANTILE, DATA_WITH_PREDS, DATA_WITH_SIGNALS

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for model in ["rf", "xgb", "lr"]:
        pred_col = f"pred_{model}"
        signal_col = f"signal_{model}"
        if pred_col not in df.columns:
            raise ValueError(f"Missing {pred_col} in DataFrame")
        df[signal_col] = 0
        if SIGNAL_METHOD == "diff":
            df.loc[df[pred_col] > df[pred_col].shift(1), signal_col] = 1
            df.loc[df[pred_col] < df[pred_col].shift(1), signal_col] = -1
        elif SIGNAL_METHOD == "threshold":
            if THRESHOLD_METHOD == "median":
                thresh = df[pred_col].median()
            elif THRESHOLD_METHOD == "quantile":
                thresh = df[pred_col].quantile(THRESHOLD_QUANTILE)
            else:
                thresh = float(THRESHOLD_METHOD)
            df.loc[df[pred_col] > thresh, signal_col] = 1
            df.loc[df[pred_col] < thresh, signal_col] = -1
        else:
            raise NotImplementedError(f"SIGNAL_METHOD={SIGNAL_METHOD} not implemented")
    return df

if __name__ == "__main__":
    print(f"Loading predictions from {DATA_WITH_PREDS}...")
    df = load_df(DATA_WITH_PREDS)
    print("Generating signals...")
    df_signals = generate_signals(df)
    save_df(df_signals, DATA_WITH_SIGNALS)
    print(f"✅ Saved dataset with signals to {DATA_WITH_SIGNALS}")
    print(f"Signal columns: {[c for c in df_signals.columns if c.startswith('signal_')]}")

