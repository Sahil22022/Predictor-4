import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from data_handler import fetch_stock_data, get_company_info, get_multiple_stocks
from feature_engineering import compute_technical_indicators, prepare_features
from models import (
    train_linear_regression, train_random_forest,
    train_xgboost, train_lstm,
    predict_future, evaluate_model
)
from utils import format_currency, format_percent, color_metric

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSage AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark background */
.stApp {
    background: #060a12;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d40;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2d40;
    padding-bottom: 8px;
    margin-top: 1.5rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #0f2137 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
}
.metric-delta { font-size: 0.8rem; margin-top: 4px; }
.delta-up   { color: #22c55e; }
.delta-down { color: #ef4444; }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #38bdf8;
    border-left: 3px solid #38bdf8;
    padding-left: 10px;
    margin: 2rem 0 1rem 0;
}

/* Prediction box */
.prediction-box {
    background: linear-gradient(135deg, #0f2d1f 0%, #0a1f2e 100%);
    border: 1px solid #16a34a;
    border-radius: 14px;
    padding: 24px;
    margin: 12px 0;
}
.prediction-price {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #4ade80;
}
.prediction-label { color: #86efac; font-size: 0.85rem; letter-spacing: 0.1em; }

/* Model badge */
.model-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #7dd3fc;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px;
    letter-spacing: 0.08em;
}

/* Alert box */
.info-box {
    background: #0c1a2e;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 8px 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #1e2d40;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    padding: 10px 24px;
    transition: all 0.3s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #38bdf8);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(56,189,248,0.3);
}

/* Select / Input */
.stSelectbox label, .stSlider label, .stTextInput label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* Hero */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.hero-sub {
    color: #475569;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <span style='font-family:Space Mono,monospace; font-size:1.3rem;
              background: linear-gradient(135deg,#38bdf8,#818cf8);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              font-weight:700;'>📈 StockSage AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🔍 Stock Selection")
    ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="e.g. AAPL, TSLA, MSFT").upper().strip()

    popular = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX", "BRK-B", "JPM"]
    st.markdown("<div style='margin:-8px 0 8px 0;font-size:0.7rem;color:#475569;'>Quick select:</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i, t in enumerate(popular):
        if cols[i % 5].button(t, key=f"btn_{t}"):
            ticker = t

    st.markdown("## ⚙️ Data Settings")
    period_map = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y"}
    period_label = st.selectbox("Historical Period", list(period_map.keys()), index=2)
    period = period_map[period_label]

    interval_map = {"Daily": "1d", "Weekly": "1wk"}
    interval_label = st.selectbox("Interval", list(interval_map.keys()), index=0)
    interval = interval_map[interval_label]

    st.markdown("## 🤖 Model Settings")
    model_choice = st.selectbox("Prediction Model", ["LSTM (Deep Learning)", "Random Forest", "XGBoost", "Linear Regression", "Ensemble (All)"])
    forecast_days = st.slider("Forecast Horizon (days)", min_value=5, max_value=60, value=30, step=5)

    st.markdown("## 📊 Indicators")
    show_ma     = st.checkbox("Moving Averages",     value=True)
    show_bb     = st.checkbox("Bollinger Bands",     value=True)
    show_rsi    = st.checkbox("RSI",                 value=True)
    show_macd   = st.checkbox("MACD",                value=False)
    show_volume = st.checkbox("Volume",              value=True)

    st.markdown("---")
    run_btn = st.button("🚀  RUN ANALYSIS", key="run")

    st.markdown("""
    <div class='info-box'>
    ⚠️ AI predictions are for educational purposes only. Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════════════
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.markdown(f"""
    <div class='hero-title'>StockSage AI</div>
    <div class='hero-sub'>Machine Learning · Technical Analysis · Live Market Data</div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ════════════════════════════════════════════════════════════════════
if run_btn or "stock_data" in st.session_state:

    if run_btn:
        with st.spinner(f"📡 Fetching live data for **{ticker}**..."):
            data = fetch_stock_data(ticker, period, interval)
            info = get_company_info(ticker)
        if data is None or data.empty:
            st.error(f"❌ Could not fetch data for **{ticker}**. Check the ticker symbol.")
            st.stop()
        st.session_state["stock_data"]    = data
        st.session_state["company_info"]  = info
        st.session_state["ticker"]        = ticker
        st.session_state["model_choice"]  = model_choice
        st.session_state["forecast_days"] = forecast_days
    else:
        data         = st.session_state["stock_data"]
        info         = st.session_state["company_info"]
        ticker       = st.session_state["ticker"]
        model_choice = st.session_state["model_choice"]
        forecast_days= st.session_state["forecast_days"]

    # ── Company Header ────────────────────────────────────────────────
    company_name = info.get("longName", ticker)
    sector       = info.get("sector", "N/A")
    industry     = info.get("industry", "N/A")
    market_cap   = info.get("marketCap", 0)

    st.markdown(f"""
    <div style='margin:16px 0 8px 0;'>
        <span style='font-size:1.6rem;font-weight:700;color:#f1f5f9;'>{company_name}</span>
        <span style='font-size:0.85rem;color:#38bdf8;margin-left:10px;font-family:Space Mono,monospace;'>{ticker}</span>
        <span style='font-size:0.75rem;color:#475569;margin-left:12px;'>● {sector} / {industry}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ───────────────────────────────────────────────────
    latest       = data.iloc[-1]
    prev         = data.iloc[-2]
    price        = latest["Close"]
    price_change = price - prev["Close"]
    pct_change   = (price_change / prev["Close"]) * 100
    high_52w     = data["High"].max()
    low_52w      = data["Low"].min()
    avg_vol      = int(data["Volume"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    delta_cls = "delta-up" if price_change >= 0 else "delta-down"
    delta_sym = "▲" if price_change >= 0 else "▼"

    metrics = [
        ("Current Price",   f"${price:.2f}",              f'<span class="{delta_cls}">{delta_sym} {abs(pct_change):.2f}%</span>'),
        ("52W High",        f"${high_52w:.2f}",           f'<span style="color:#64748b;">—</span>'),
        ("52W Low",         f"${low_52w:.2f}",            f'<span style="color:#64748b;">—</span>'),
        ("Avg Volume",      f"{avg_vol:,}",               f'<span style="color:#64748b;">shares/day</span>'),
        ("Market Cap",      format_currency(market_cap),  f'<span style="color:#64748b;">USD</span>'),
    ]
    for col, (lbl, val, dlt) in zip([c1,c2,c3,c4,c5], metrics):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{lbl}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-delta'>{dlt}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Compute Indicators ────────────────────────────────────────────
    with st.spinner("🔬 Computing technical indicators..."):
        data = compute_technical_indicators(data)

    # ════════════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs(["📈  PRICE & INDICATORS", "🤖  AI PREDICTION", "📊  MODEL METRICS", "🔬  FEATURE ANALYSIS"])

    # ─────────────────── TAB 1: Price Chart ──────────────────────────
    with tab1:
        rows  = 1 + (1 if show_rsi else 0) + (1 if show_macd else 0) + (1 if show_volume else 0)
        specs = [[{"secondary_y": False}]] * rows
        row_heights = []
        if rows == 1:
            row_heights = [1.0]
        else:
            row_heights = [0.55] + [0.15] * (rows - 1)

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=row_heights, specs=specs)

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"],  close=data["Close"],
            name="OHLC", increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#16a34a", decreasing_fillcolor="#dc2626"
        ), row=1, col=1)

        colors_ma = {"MA20": "#f59e0b", "MA50": "#38bdf8", "MA100": "#a78bfa", "MA200": "#fb923c"}
        if show_ma:
            for ma, clr in colors_ma.items():
                if ma in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data[ma], name=ma,
                                             line=dict(color=clr, width=1.2)), row=1, col=1)

        if show_bb and "BB_Upper" in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], name="BB Upper",
                                     line=dict(color="#818cf8", width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], name="BB Lower",
                                     line=dict(color="#818cf8", width=1, dash="dash"),
                                     fill="tonexty", fillcolor="rgba(129,140,248,0.05)"), row=1, col=1)

        current_row = 2
        if show_rsi and "RSI" in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI",
                                     line=dict(color="#f472b6", width=1.5)), row=current_row, col=1)
            fig.add_hline(y=70, line_color="#ef4444", line_dash="dot", line_width=1, row=current_row, col=1)
            fig.add_hline(y=30, line_color="#22c55e", line_dash="dot", line_width=1, row=current_row, col=1)
            fig.update_yaxes(title_text="RSI", row=current_row, col=1, color="#94a3b8", gridcolor="#1e2d40")
            current_row += 1

        if show_macd and "MACD" in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD",
                                     line=dict(color="#38bdf8", width=1.5)), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal",
                                     line=dict(color="#fb923c", width=1.5)), row=current_row, col=1)
            colors_hist = ["#22c55e" if v >= 0 else "#ef4444" for v in data["MACD_Hist"]]
            fig.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="Histogram",
                                 marker_color=colors_hist), row=current_row, col=1)
            fig.update_yaxes(title_text="MACD", row=current_row, col=1, color="#94a3b8", gridcolor="#1e2d40")
            current_row += 1

        if show_volume:
            vol_colors = ["#22c55e" if data["Close"].iloc[i] >= data["Open"].iloc[i]
                          else "#ef4444" for i in range(len(data))]
            fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume",
                                 marker_color=vol_colors, opacity=0.6), row=current_row, col=1)
            fig.update_yaxes(title_text="Volume", row=current_row, col=1, color="#94a3b8", gridcolor="#1e2d40")

        fig.update_layout(
            height=600 + (rows - 1) * 120,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a12",
            font_family="Sora", font_color="#94a3b8",
            legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#1e2d40",
                        borderwidth=1, font_size=11),
            xaxis_rangeslider_visible=False,
            margin=dict(t=20, b=10, l=10, r=10),
        )
        fig.update_xaxes(gridcolor="#0f1923", showgrid=True)
        fig.update_yaxes(gridcolor="#0f1923", showgrid=True, row=1, col=1, color="#94a3b8")
        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────── TAB 2: AI Prediction ────────────────────────
    with tab2:
        st.markdown("<div class='section-header'>AI-Powered Price Forecast</div>", unsafe_allow_html=True)

        with st.spinner("🧠 Training model & generating forecast..."):
            X, y, feature_names, scaler, close_scaler = prepare_features(data)

            if model_choice == "Ensemble (All)":
                models_to_run = ["Linear Regression", "Random Forest", "XGBoost", "LSTM (Deep Learning)"]
            else:
                models_to_run = [model_choice]

            results = {}
            for m in models_to_run:
                if "LSTM" in m:
                    model, history_loss = train_lstm(X, y, close_scaler)
                    preds = predict_future(model, X, forecast_days, m, close_scaler, scaler, data, feature_names)
                elif "Random Forest" in m:
                    model = train_random_forest(X, y)
                    preds = predict_future(model, X, forecast_days, m, close_scaler, scaler, data, feature_names)
                elif "XGBoost" in m:
                    model = train_xgboost(X, y)
                    preds = predict_future(model, X, forecast_days, m, close_scaler, scaler, data, feature_names)
                else:
                    model = train_linear_regression(X, y)
                    preds = predict_future(model, X, forecast_days, m, close_scaler, scaler, data, feature_names)
                results[m] = {"model": model, "predictions": preds}

        # Future date range
        last_date  = data.index[-1]
        biz_days   = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

        # Prediction summary
        pcol1, pcol2 = st.columns([1, 2])
        with pcol1:
            for m_name, res in results.items():
                preds      = res["predictions"]
                final_pred = preds[-1]
                chg        = ((final_pred - price) / price) * 100
                direction  = "📈" if chg > 0 else "📉"
                border_clr = "#16a34a" if chg > 0 else "#dc2626"
                st.markdown(f"""
                <div class='prediction-box' style='border-color:{border_clr};'>
                    <div class='prediction-label'>{m_name} · {forecast_days}d Forecast</div>
                    <div class='prediction-price'>${final_pred:.2f}</div>
                    <div style='margin-top:6px; color:{"#4ade80" if chg>0 else "#f87171"}; font-size:1rem; font-weight:600;'>
                        {direction} {chg:+.2f}% vs today
                    </div>
                    <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>
                        Current: ${price:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with pcol2:
            fig2 = go.Figure()

            # Historical
            hist_window = data["Close"].iloc[-90:]
            fig2.add_trace(go.Scatter(
                x=hist_window.index, y=hist_window.values,
                name="Historical", line=dict(color="#64748b", width=2)
            ))

            palette = ["#38bdf8", "#4ade80", "#f59e0b", "#c084fc"]
            for idx, (m_name, res) in enumerate(results.items()):
                preds = res["predictions"]
                clr   = palette[idx % len(palette)]

                # Confidence band
                noise = np.std(preds) * 0.15
                upper = [p + noise * (i + 1) ** 0.5 for i, p in enumerate(preds)]
                lower = [p - noise * (i + 1) ** 0.5 for i, p in enumerate(preds)]

                fig2.add_trace(go.Scatter(
                    x=list(biz_days) + list(biz_days[::-1]),
                    y=upper + lower[::-1],
                    fill="toself", fillcolor=f"rgba{tuple(int(clr.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.08,)}",
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False, name=f"{m_name} Band"
                ))
                # Bridge from last historical to first prediction
                fig2.add_trace(go.Scatter(
                    x=[hist_window.index[-1], biz_days[0]],
                    y=[hist_window.values[-1], preds[0]],
                    line=dict(color=clr, width=2, dash="dot"),
                    showlegend=False
                ))
                fig2.add_trace(go.Scatter(
                    x=biz_days, y=preds, name=m_name,
                    line=dict(color=clr, width=2.5)
                ))

            fig2.add_vline(x=str(last_date), line_color="#475569",
                           line_dash="dot", line_width=1.5,
                           annotation_text="Today", annotation_font_color="#94a3b8")

            fig2.update_layout(
                height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a12",
                font_family="Sora", font_color="#94a3b8",
                legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#1e2d40", borderwidth=1, font_size=11),
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor="#0f1923"), yaxis=dict(gridcolor="#0f1923"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Forecast table
        st.markdown("<div class='section-header'>Detailed Forecast Table</div>", unsafe_allow_html=True)
        df_forecast = pd.DataFrame({"Date": biz_days.strftime("%Y-%m-%d")})
        for m_name, res in results.items():
            df_forecast[m_name] = [f"${p:.2f}" for p in res["predictions"]]
        st.dataframe(df_forecast.set_index("Date"), use_container_width=True,
                     height=min(400, forecast_days * 35 + 40))

    # ─────────────────── TAB 3: Model Metrics ────────────────────────
    with tab3:
        st.markdown("<div class='section-header'>Model Evaluation Metrics</div>", unsafe_allow_html=True)
        with st.spinner("📐 Evaluating models..."):
            X_eval, y_eval, _, sc_eval, csc_eval = prepare_features(data)
            split   = int(len(X_eval) * 0.8)
            X_train = X_eval[:split]; X_test = X_eval[split:]
            y_train = y_eval[:split]; y_test = y_eval[split:]

            eval_results = {}
            for m_name in models_to_run:
                if "LSTM" in m_name:
                    mdl, _ = train_lstm(X_train, y_train, csc_eval)
                elif "Random Forest" in m_name:
                    mdl = train_random_forest(X_train, y_train)
                elif "XGBoost" in m_name:
                    mdl = train_xgboost(X_train, y_train)
                else:
                    mdl = train_linear_regression(X_train, y_train)
                metrics_dict = evaluate_model(mdl, X_test, y_test, m_name, csc_eval)
                eval_results[m_name] = metrics_dict

        # Metrics table
        df_metrics = pd.DataFrame(eval_results).T.reset_index()
        df_metrics.columns = ["Model", "RMSE", "MAE", "MAPE (%)", "R² Score", "Directional Acc (%)"]
        st.dataframe(df_metrics.set_index("Model").style
            .format({"RMSE": "{:.4f}", "MAE": "{:.4f}",
                     "MAPE (%)": "{:.2f}", "R² Score": "{:.4f}", "Directional Acc (%)": "{:.1f}"}),
            use_container_width=True)

        # Bar charts
        fig3 = make_subplots(rows=1, cols=2,
                             subplot_titles=["RMSE (lower = better)", "R² Score (higher = better)"])
        names = list(eval_results.keys())
        rmse_vals = [eval_results[m]["RMSE"] for m in names]
        r2_vals   = [eval_results[m]["R² Score"] for m in names]
        bar_clrs  = ["#38bdf8", "#4ade80", "#f59e0b", "#c084fc"][:len(names)]

        fig3.add_trace(go.Bar(x=names, y=rmse_vals, marker_color=bar_clrs, showlegend=False), row=1, col=1)
        fig3.add_trace(go.Bar(x=names, y=r2_vals,   marker_color=bar_clrs, showlegend=False), row=1, col=2)
        fig3.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a12",
                           font_family="Sora", font_color="#94a3b8",
                           margin=dict(t=40, b=10, l=10, r=10))
        fig3.update_xaxes(gridcolor="#0f1923"); fig3.update_yaxes(gridcolor="#0f1923")
        st.plotly_chart(fig3, use_container_width=True)

        # Actual vs Predicted
        st.markdown("<div class='section-header'>Actual vs Predicted — Test Set</div>", unsafe_allow_html=True)
        fig4 = go.Figure()
        test_dates = data.index[split + (len(data) - len(X_eval)):][:len(y_test)]
        # Actual
        actual_prices = csc_eval.inverse_transform(y_test.reshape(-1,1)).flatten()
        fig4.add_trace(go.Scatter(x=test_dates, y=actual_prices, name="Actual",
                                  line=dict(color="#64748b", width=2)))
        for idx, m_name in enumerate(models_to_run):
            mdl2 = eval_results[m_name].get("_model")
            if mdl2 is None:
                continue
            if "LSTM" in m_name:
                from models import predict_lstm_sequence
                preds_scaled = predict_lstm_sequence(mdl2, X_test)
            else:
                preds_scaled = mdl2.predict(X_test.reshape(len(X_test), -1))
            pred_prices = csc_eval.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
            fig4.add_trace(go.Scatter(x=test_dates[:len(pred_prices)], y=pred_prices, name=m_name,
                                      line=dict(color=bar_clrs[idx], width=1.8)))
        fig4.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a12",
                           font_family="Sora", font_color="#94a3b8",
                           legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#1e2d40", borderwidth=1),
                           margin=dict(t=10, b=10, l=10, r=10),
                           xaxis=dict(gridcolor="#0f1923"), yaxis=dict(gridcolor="#0f1923"))
        st.plotly_chart(fig4, use_container_width=True)

    # ─────────────────── TAB 4: Feature Analysis ─────────────────────
    with tab4:
        st.markdown("<div class='section-header'>Feature Importance & Correlation</div>", unsafe_allow_html=True)

        # Correlation heatmap
        cols_corr = ["Close", "Volume", "MA20", "MA50", "RSI", "MACD", "BB_Width",
                     "Daily_Return", "Volatility_20d", "OBV"]
        cols_corr = [c for c in cols_corr if c in data.columns]
        corr_matrix = data[cols_corr].dropna().corr()

        fig5 = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[[0,"#ef4444"],[0.5,"#0d1117"],[1,"#22c55e"]],
            zmid=0, text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}", textfont_size=10,
            hoverongaps=False
        ))
        fig5.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a12",
                           font_family="Sora", font_color="#94a3b8",
                           margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig5, use_container_width=True)

        # Feature importance (RF)
        st.markdown("<div class='section-header'>Random Forest Feature Importance</div>", unsafe_allow_html=True)
        rf_model = train_random_forest(X, y)
        importances = rf_model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=True).tail(15)

        fig6 = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"],
            orientation="h",
            marker=dict(color=feat_df["Importance"],
                        colorscale=[[0,"#1e3a5f"],[1,"#38bdf8"]])
        ))
        fig6.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a12",
                           font_family="Sora", font_color="#94a3b8",
                           margin=dict(t=10, b=10, l=10, r=20),
                           xaxis=dict(gridcolor="#0f1923"), yaxis=dict(gridcolor="#0f1923"))
        st.plotly_chart(fig6, use_container_width=True)

        # Returns distribution
        st.markdown("<div class='section-header'>Daily Returns Distribution</div>", unsafe_allow_html=True)
        if "Daily_Return" in data.columns:
            ret = data["Daily_Return"].dropna() * 100
            fig7 = go.Figure()
            fig7.add_trace(go.Histogram(x=ret, nbinsx=80, name="Returns",
                                        marker_color="#38bdf8", opacity=0.7))
            fig7.add_vline(x=0, line_color="#ef4444", line_dash="dot", line_width=1.5)
            fig7.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a12",
                               font_family="Sora", font_color="#94a3b8",
                               margin=dict(t=10, b=10, l=10, r=10),
                               xaxis=dict(title="Daily Return (%)", gridcolor="#0f1923"),
                               yaxis=dict(gridcolor="#0f1923"))
            st.plotly_chart(fig7, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align:center; padding:60px 20px;'>
        <div style='font-size:4rem; margin-bottom:16px;'>📈</div>
        <div style='font-family:Space Mono,monospace; font-size:1.2rem; color:#38bdf8; margin-bottom:12px;'>
            Ready to Analyse
        </div>
        <div style='color:#475569; max-width:500px; margin:0 auto; line-height:1.8;'>
            Enter a ticker symbol in the sidebar and click <strong style='color:#94a3b8;'>RUN ANALYSIS</strong>
            to fetch live market data, compute technical indicators, and generate AI price forecasts.
        </div>
        <div style='margin-top:32px; display:flex; justify-content:center; gap:8px; flex-wrap:wrap;'>
    """ + "".join([f"<span class='model-badge'>{t}</span>" for t in
                   ["AAPL","TSLA","MSFT","GOOGL","AMZN","NVDA","META","NFLX","BTC-USD","ETH-USD"]])
    + "</div></div>", unsafe_allow_html=True)
