"""
data_handler.py  –  Live stock data via yfinance
"""
import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=300)          # cache 5 minutes
def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame | None:
    """Fetch OHLCV data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


@st.cache_data(ttl=3600)
def get_company_info(ticker: str) -> dict:
    """Fetch company metadata."""
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
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
