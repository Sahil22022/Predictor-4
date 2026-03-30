"""
data_handler.py  –  Live stock data via yfinance (robust version)
"""
import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame | None:
    """Fetch OHLCV data from Yahoo Finance — handles both old and new yfinance column formats."""
    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period=period, interval=interval, auto_adjust=True)

        if df is None or df.empty:
            return None

        # ── Fix multi-level columns (yfinance ≥ 0.2.43 returns MultiIndex) ──
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ── Normalise column names ──
        df.columns = [str(c).strip().title() for c in df.columns]

        rename_map = {"Adj Close": "Close", "Adj. Close": "Close"}
        df.rename(columns=rename_map, inplace=True)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing  = [c for c in required if c not in df.columns]

        if missing:
            # Fallback: use yf.download()
            df2 = yf.download(ticker, period=period, interval=interval,
                              auto_adjust=True, progress=False)
            if isinstance(df2.columns, pd.MultiIndex):
                df2.columns = df2.columns.get_level_values(0)
            df2.columns = [str(c).strip().title() for c in df2.columns]
            df2.rename(columns=rename_map, inplace=True)
            if df2.empty or any(c not in df2.columns for c in required):
                return None
            df = df2

        df = df[required].copy()

        # ── Clean index ──
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.dropna(inplace=True)

        if len(df) < 60:
            return None

        return df

    except Exception as e:
        print(f"fetch_stock_data error ({ticker}): {e}")
        return None


@st.cache_data(ttl=3600)
def get_company_info(ticker: str) -> dict:
    """Fetch company metadata — returns empty dict on any failure."""
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_multiple_stocks(tickers: list[str], period: str = "1y") -> dict:
    """Fetch closing prices for multiple tickers (for comparison)."""
    result = {}
    for t in tickers:
        df = fetch_stock_data(t, period)
        if df is not None:
            result[t] = df["Close"]
    return result
