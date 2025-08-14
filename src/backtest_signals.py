import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_df
from src.config import DATA_WITH_SIGNALS, TRANSACTION_COST

def backtest_model(df, signal_col, tc=0.0):
    sig = df[signal_col].shift(1).fillna(0)
    ret = df["log_ret"]
    strat_ret = sig * ret
    trades = sig.diff().abs() / 2
    strat_ret -= trades * tc
    eq_curve = strat_ret.cumsum()
    total_pnl = eq_curve.iloc[-1]
    sharpe = np.sqrt(252) * strat_ret.mean() / strat_ret.std() if strat_ret.std() != 0 else 0
    win_rate = (strat_ret > 0).mean()
    max_dd = (eq_curve - eq_curve.cummax()).min()
    return {"model": signal_col, "total_pnl": total_pnl, "sharpe": sharpe, 
            "win_rate": win_rate, "max_dd": max_dd, "equity_curve": eq_curve}

def main(tc=TRANSACTION_COST):
    df = load_df(DATA_WITH_SIGNALS)
    model_cols = [c for c in df.columns if c.startswith("signal_")]
    results = []
    all_curves = pd.DataFrame(index=df.index)
    for col in model_cols:
        res = backtest_model(df, col, tc=tc)
        results.append(res)
        all_curves[col] = res["equity_curve"]
    # Buy & hold
    bh_curve = df["log_ret"].cumsum()
    bh_total_pnl = bh_curve.iloc[-1]
    bh_sharpe = np.sqrt(252) * df["log_ret"].mean() / df["log_ret"].std()
    bh_max_dd = (bh_curve - bh_curve.cummax()).min()
    summary_df = pd.DataFrame(results)[["model", "total_pnl", "sharpe", "win_rate", "max_dd"]]
    summary_df["bh_total_pnl"] = bh_total_pnl
    summary_df["bh_sharpe"] = bh_sharpe
    summary_df["bh_max_dd"] = bh_max_dd
    print("\n=== BACKTEST SUMMARY ===")
    print(summary_df)
    # Plot
    plt.figure(figsize=(10,6))
    for col in model_cols:
        plt.plot(all_curves.index, all_curves[col], label=col)
    plt.plot(bh_curve.index, bh_curve, label="buy_hold", linestyle="--")
    plt.title("Equity Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

