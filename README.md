# 📈 StockSage AI — ML Stock Price Predictor

A fully-featured Streamlit app that fetches live market data from Yahoo Finance and applies
multiple machine learning models to predict future stock closing prices.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Live Data** | Real-time OHLCV from Yahoo Finance via `yfinance` |
| **Technical Indicators** | MA(20/50/100/200), EMA, Bollinger Bands, RSI, MACD, ATR, Stochastic, OBV, VWAP |
| **ML Models** | LSTM, Random Forest, XGBoost, Ridge Regression, Ensemble |
| **Forecast Horizon** | 5 – 60 business days |
| **Model Evaluation** | RMSE, MAE, MAPE, R², Directional Accuracy |
| **Visualisations** | Interactive Plotly candlestick chart, forecast chart, correlation heatmap, feature importance |

---

## 🚀 Quick Start

### 1. Clone / download

```bash
git clone <repo>   # or just place all .py files in a folder
cd stock_predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **TensorFlow note:** If you don't want LSTM (or can't install TF), the app automatically
> falls back to Random Forest. All other models work without TensorFlow.

### 4. Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🗂 Project Structure

```
stock_predictor/
├── app.py                  # Main Streamlit application & UI
├── data_handler.py         # Live data fetching (yfinance)
├── feature_engineering.py  # 30+ technical indicators + ML feature prep
├── models.py               # LSTM, Random Forest, XGBoost, Ridge
├── utils.py                # Formatting helpers
├── requirements.txt
└── README.md
```

---

## 🤖 Model Details

### LSTM (Long Short-Term Memory)
- 3-layer LSTM with Dropout + BatchNormalization
- Sequence length: 30 trading days
- EarlyStopping + ReduceLROnPlateau callbacks
- Best for capturing temporal dependencies

### Random Forest
- 200 trees, max depth 12
- Uses all 30+ engineered features
- Robust to outliers; provides feature importance

### XGBoost
- 300 estimators, learning rate 0.05
- L1 + L2 regularisation
- Generally the best single-model performer

### Ridge Regression (Linear)
- Baseline model with L2 regularisation
- Fast inference; useful for comparison

### Ensemble
- Runs all 4 models simultaneously
- Plots each forecast on the same chart

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**.
Predictions are not financial advice. Past performance does not guarantee future results.
Always consult a qualified financial advisor before making investment decisions.
