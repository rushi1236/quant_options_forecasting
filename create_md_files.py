# create_md_files.py

final_report_content = """
# Quant Options Forecasting - Final Report

## Project Overview
This project focuses on forecasting short-term volatility of NIFTY using machine learning models
(RandomForest, XGBoost, LinearRegression) and evaluating trading signals derived from predictions.

## Data Ingestion
- Source: Yahoo Finance (^NSEI)
- Period: 2018-01-01 to 2025-07-31
- Features: OHLCV + log returns + rolling historical volatilities (HV10, HV20, HV30)

## Feature Engineering
- Absolute returns, multi-day returns, rolling volatilities
- Z-score normalization
- Forward-looking realized volatility as target

## Modeling
- Models: RandomForest, XGBoost, LinearRegression
- Train/Test split: 80/20 (chronological)
- Metrics: R² (In-sample & Out-of-sample), RMSE

## Signal Generation
- Long if predicted vol > yesterday's prediction
- Short if predicted vol < yesterday's prediction
- Transaction cost adjustments included

## Backtesting
- Equity curves vs buy-and-hold
- Metrics: Total PnL, Sharpe, Max Drawdown, Win Rate
- Transaction costs incorporated

## Results
- Best model (by OOS R²): <fill in best model>
- Significant improvement over benchmark in PnL and Sharpe ratio
- Visual inspection confirms reasonable capture of volatility spikes

## Conclusion
The pipeline demonstrates end-to-end ML-based volatility forecasting and trading signal evaluation.
It can be extended to other indices, intraday data, or options strategies for live trading.
"""

#readme_content = """
# Quant Options Forecasting

#End-to-end pipeline for short-term volatility forecasting and signal backtesting.

## Project Structure

