# generate_docs.py
from pathlib import Path

# Paths
README_PATH = Path("README.md")
REPORT_PATH = Path("final_report.md")

# README content
#readme_content = """# Quant Options Volatility Forecasting

#End-to-end volatility forecasting pipeline for NIFTY index using classical and ML models. 
#Generates trading signals and backtests strategies vs. buy-and-hold benchmark.

## Features

- Data ingestion from Yahoo Finance (OHLCV)
- Feature engineering: log returns, realized volatility, lags, moving averages
- ML model training: Random Forest, XGBoost, Linear Regression
- Signal generation based on model predictions
- Backtesting: PnL, Sharpe, max drawdown, win rate

## Modules

- `src/data_loader.py` – download/save/load OHLCV data
- `src/feature_engineering.py` – compute engineered features
- `src/model_training.py` – train models and save predictions
- `src/signal_generation.py` – generate trading signals
- `src/backtest_signals.py` – backtest signals vs benchmark

## Setup

1. Clone repo
2. Create virtual environment:
```bash
python -m venv ~/myenv311
source ~/myenv311/bin/activate
