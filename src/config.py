# -----------------------------
# File paths
# -----------------------------
DATA_RAW = "data/raw/underlying_raw.pkl"
DATA_PROCESSED = "data/processed/underlying_price.pkl"
DATA_WITH_PREDS = "data/processed/underlying_price_with_preds.pkl"
DATA_WITH_SIGNALS = "data/processed/underlying_price_with_signals.pkl"

# -----------------------------
# ML Model Hyperparameters
# -----------------------------
MODELS = {
    "rf": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
    "xgb": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, 
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
    "lr": {}  # LinearRegression has no hyperparameters
}

# -----------------------------
# Signal Generation
# -----------------------------
SIGNAL_METHOD = "diff"        # 'diff' or 'threshold'
THRESHOLD_METHOD = "median"   # 'median', 'quantile', or numeric
THRESHOLD_QUANTILE = 0.6
HOLD_DAYS = 1                 # holding period

# -----------------------------
# Backtest Settings
# -----------------------------
TRANSACTION_COST = 0.0
SCALE = 1.0
DT = 1 / 252
