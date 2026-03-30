"""
models.py  –  ML model training, prediction and evaluation
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

from feature_engineering import prepare_features, LOOKBACK


# ── Linear Regression (Ridge) ─────────────────────────────────────────────────

def train_linear_regression(X: np.ndarray, y: np.ndarray):
    X_flat = X.reshape(len(X), -1)
    model  = Ridge(alpha=1.0)
    model.fit(X_flat, y)
    return model


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(X: np.ndarray, y: np.ndarray):
    X_flat = X.reshape(len(X), -1)
    model  = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=42
    )
    model.fit(X_flat, y)
    return model


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(X: np.ndarray, y: np.ndarray):
    X_flat = X.reshape(len(X), -1)
    if XGBOOST_AVAILABLE:
        model = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0
        )
    else:
        # Fallback to sklearn GradientBoosting
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
    model.fit(X_flat, y)
    return model


# ── LSTM ──────────────────────────────────────────────────────────────────────

def train_lstm(X: np.ndarray, y: np.ndarray, close_scaler):
    if not KERAS_AVAILABLE:
        # Graceful fallback to Random Forest
        return train_random_forest(X, y), []

    tf.random.set_seed(42)

    n_features = X.shape[2]
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], n_features)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="huber")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]
    history = model.fit(
        X, y,
        epochs=60, batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history.history.get("loss", [])


def predict_lstm_sequence(model, X: np.ndarray) -> np.ndarray:
    if KERAS_AVAILABLE and hasattr(model, "predict") and "keras" in str(type(model)):
        return model.predict(X, verbose=0).flatten()
    return model.predict(X.reshape(len(X), -1))


# ── Future Prediction ─────────────────────────────────────────────────────────

def predict_future(model, X: np.ndarray, n_days: int, model_name: str,
                   close_scaler, feature_scaler, data, feature_names) -> np.ndarray:
    """
    Iteratively predict n_days into the future.
    Returns array of de-scaled closing prices.
    """
    from feature_engineering import FEATURE_COLUMNS
    lookback   = X.shape[1]
    n_features = X.shape[2]
    preds_scaled = []

    current_seq = X[-1].copy()     # shape: (lookback, n_features)

    for _ in range(n_days):
        inp = current_seq[np.newaxis]          # (1, lookback, n_features)
        is_lstm = KERAS_AVAILABLE and "keras" in str(type(model))

        if is_lstm:
            pred_s = model.predict(inp, verbose=0)[0, 0]
        else:
            pred_s = model.predict(inp.reshape(1, -1))[0]

        preds_scaled.append(pred_s)

        # Roll sequence forward: append new step
        new_step = current_seq[-1].copy()
        close_idx = 0   # Close is first feature in FEATURE_COLUMNS
        new_step[close_idx] = pred_s
        current_seq = np.vstack([current_seq[1:], new_step])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds        = close_scaler.inverse_transform(preds_scaled).flatten()
    # Sanity clamp: ±50% of last price
    last_price   = data["Close"].iloc[-1]
    preds        = np.clip(preds, last_price * 0.5, last_price * 2.0)
    return preds


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str, close_scaler) -> dict:
    is_lstm = KERAS_AVAILABLE and "keras" in str(type(model))

    if is_lstm:
        y_pred_s = model.predict(X_test, verbose=0).flatten()
    else:
        y_pred_s = model.predict(X_test.reshape(len(X_test), -1))

    y_true = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = close_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2    = r2_score(y_true, y_pred)

    # Directional accuracy
    actual_dir = np.diff(y_true) > 0
    pred_dir   = np.diff(y_pred) > 0
    dir_acc    = np.mean(actual_dir == pred_dir) * 100

    return {
        "RMSE":                  round(rmse, 4),
        "MAE":                   round(mae, 4),
        "MAPE (%)":              round(mape, 2),
        "R² Score":              round(r2, 4),
        "Directional Acc (%)":   round(dir_acc, 1),
        "_model":                model,
    }
