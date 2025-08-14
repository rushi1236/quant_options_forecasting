#!/bin/bash

# Ensure we're in project root
ROOT_DIR=$(pwd)
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

echo "==================="
echo "1. Data Ingestion"
echo "==================="
python src/data_loader.py

echo "==================="
echo "2. Feature Engineering"
echo "==================="
python src/feature_engineering.py

echo "==================="
echo "3. Model Training"
echo "==================="
python src/model_training.py

echo "==================="
echo "4. Signal Generation"
echo "==================="
python src/signal_generation.py

echo "==================="
echo "5. Backtesting Signals"
echo "==================="
python backtest_signals.py --tc 0.0

echo "✅ Pipeline completed successfully!"
