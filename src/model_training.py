"""
src/model_training.py

Phase 3: Train multiple models on engineered features and save predictions.
Models: RandomForest, XGBoost, LinearRegression.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("Please install xgboost: pip install xgboost")

from src.data_loader import load_df, save_df

# ======================
# CONFIG
# ======================
INPUT_PATH = Path("data/processed/underlying_price.pkl")
OUTPUT_PATH = Path("data/processed/underlying_price_with_preds.pkl")

FEATURES = ['abs_log_ret', 'ret_3d', 'ret_std_5d', 'hv10', 'hv_diff', 'day_of_week']
TARGET = 'hv10_fwd'


def train_and_predict():
    # 1. Load data
    print(f"Loading data from {INPUT_PATH}...")
    df = load_df(INPUT_PATH)

    # 2. Train/Test split
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    results = {}

    # 3. RandomForest
    rf = RandomForestRegressor(max_depth=7, n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    df['pred_rf'] = rf.predict(df[FEATURES])
    results['rf_r2'] = r2_score(y_test, rf.predict(X_test))
    results['rf_rmse'] = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))

    # 4. XGBoost
    xgb = XGBRegressor(max_depth=4, n_estimators=200, learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)
    df['pred_xgb'] = xgb.predict(df[FEATURES])
    results['xgb_r2'] = r2_score(y_test, xgb.predict(X_test))
    results['xgb_rmse'] = np.sqrt(mean_squared_error(y_test, xgb.predict(X_test)))

    # 5. LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    df['pred_lr'] = lr.predict(df[FEATURES])
    results['lr_r2'] = r2_score(y_test, lr.predict(X_test))
    results['lr_rmse'] = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))

    # 6. Save
    save_df(df, OUTPUT_PATH)
    print(f"✅ Saved dataset with predictions to {OUTPUT_PATH}")
    print("Model Performance (OOS):")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    train_and_predict()
