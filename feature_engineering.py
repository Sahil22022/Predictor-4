"""
feature_engineering.py  –  Technical indicators & ML feature preparation
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Technical Indicators ──────────────────────────────────────────────────────

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # ── Moving Averages
    for window in [20, 50, 100, 200]:
        data[f"MA{window}"] = data["Close"].rolling(window).mean()

    # ── EMA
    for span in [12, 26]:
        data[f"EMA{span}"] = data["Close"].ewm(span=span, adjust=False).mean()

    # ── Bollinger Bands (20, 2σ)
    bb_mid            = data["Close"].rolling(20).mean()
    bb_std            = data["Close"].rolling(20).std()
    data["BB_Upper"]  = bb_mid + 2 * bb_std
    data["BB_Lower"]  = bb_mid - 2 * bb_std
    data["BB_Middle"] = bb_mid
    data["BB_Width"]  = (data["BB_Upper"] - data["BB_Lower"]) / bb_mid
    data["BB_Pct"]    = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"] + 1e-9)

    # ── RSI (14)
    delta      = data["Close"].diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    rs         = gain / (loss + 1e-9)
    data["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD
    data["MACD"]        = data["EMA12"] - data["EMA26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"]   = data["MACD"] - data["MACD_Signal"]

    # ── ATR (14)
    high_low     = data["High"] - data["Low"]
    high_close   = (data["High"] - data["Close"].shift()).abs()
    low_close    = (data["Low"]  - data["Close"].shift()).abs()
    true_range   = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"]  = true_range.rolling(14).mean()

    # ── Stochastic %K / %D (14)
    low14        = data["Low"].rolling(14).min()
    high14       = data["High"].rolling(14).max()
    data["Stoch_K"] = 100 * (data["Close"] - low14) / (high14 - low14 + 1e-9)
    data["Stoch_D"] = data["Stoch_K"].rolling(3).mean()

    # ── OBV
    obv          = [0]
    for i in range(1, len(data)):
        if data["Close"].iloc[i] > data["Close"].iloc[i - 1]:
            obv.append(obv[-1] + data["Volume"].iloc[i])
        elif data["Close"].iloc[i] < data["Close"].iloc[i - 1]:
            obv.append(obv[-1] - data["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    data["OBV"]  = obv

    # ── Returns & Volatility
    data["Daily_Return"]   = data["Close"].pct_change()
    data["Log_Return"]     = np.log(data["Close"] / data["Close"].shift(1))
    data["Volatility_20d"] = data["Daily_Return"].rolling(20).std()
    data["Volatility_5d"]  = data["Daily_Return"].rolling(5).std()

    # ── Price position
    data["Price_MA20_Ratio"] = data["Close"] / (data["MA20"] + 1e-9)
    data["Price_MA50_Ratio"] = data["Close"] / (data["MA50"] + 1e-9)

    # ── Volume indicators
    data["Volume_MA20"]   = data["Volume"].rolling(20).mean()
    data["Volume_Ratio"]  = data["Volume"] / (data["Volume_MA20"] + 1e-9)
    data["VWAP"]          = (data["Close"] * data["Volume"]).rolling(20).sum() / (data["Volume"].rolling(20).sum() + 1e-9)

    # ── Momentum
    for lag in [1, 3, 5, 10, 20]:
        data[f"Return_{lag}d"] = data["Close"].pct_change(lag)

    data.ffill(inplace=True)
    data.bfill(inplace=True)
    return data


# ── ML Feature Preparation ────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "Close", "High", "Low", "Open", "Volume",
    "MA20", "MA50", "EMA12", "EMA26",
    "BB_Upper", "BB_Lower", "BB_Width", "BB_Pct",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "ATR", "Stoch_K", "Stoch_D",
    "OBV", "Volatility_20d", "Volatility_5d",
    "Price_MA20_Ratio", "Price_MA50_Ratio",
    "Volume_Ratio", "VWAP",
    "Return_1d", "Return_3d", "Return_5d", "Return_10d", "Return_20d",
]

LOOKBACK = 30     # sequence length for LSTM


def prepare_features(df: pd.DataFrame, lookback: int = LOOKBACK):
    """
    Returns:
        X            – (N, lookback, n_features)
        y            – (N,) scaled close prices
        feature_names
        scaler       – fitted on all features
        close_scaler – fitted on Close only
    """
    data = df.copy()
    cols = [c for c in FEATURE_COLUMNS if c in data.columns]
    data = data[cols].dropna()

    scaler       = MinMaxScaler(feature_range=(0, 1))
    close_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_all   = scaler.fit_transform(data.values)
    scaled_close = close_scaler.fit_transform(data[["Close"]].values)

    X, y = [], []
    for i in range(lookback, len(scaled_all)):
        X.append(scaled_all[i - lookback:i])
        y.append(scaled_close[i, 0])

    return np.array(X), np.array(y), cols, scaler, close_scaler
