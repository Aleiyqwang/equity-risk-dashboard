import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from model import predict, FEATURE_COLS
# from llm import generate_commentary
from data import download_ticker, BENCHMARK_TICKER

load_dotenv()

FEATURE_LABELS = {
    "vol_60d": "60-Day Volatility",
    "vol_20d": "20-Day Volatility",
    "vol_change": "Volatility Change",
    "return_5d": "5-Day Return",
    "return_20d": "20-Day Return",
    "return_60d": "60-Day Return",
    "drawdown_from_high": "Drawdown from 52-Week High",
    "max_drawdown_60d": "Max Drawdown (60 Days)",
    "price_vs_ma50": "Price vs 50-Day Average",
    "price_vs_ma200": "Price vs 200-Day Average",
    "beta": "Market Beta (vs S&P 500)",
    "relative_return_20d": "Relative Return vs S&P 500",
}

POPULAR_TICKERS = sorted([
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "TSM", "META", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK", "ABBV",
    "LLY", "AVGO", "PEP", "KO", "COST", "ADBE", "CSCO", "TMO", "MCD", "CRM",
    "BAC", "WFC", "INTC", "AMD", "QCOM", "TXN", "NFLX", "DIS", "PYPL", "SBUX",
    "GS", "MS", "BLK", "SPGI", "AXP", "USB", "PNC", "C", "TGT", "LOW",
    "UBER", "LYFT", "SNAP", "SPOT", "COIN", "SQ", "SHOP", "ROKU", "ZOOM", "PLTR",
    "F", "GM", "RIVN", "NIO", "LCID", "BA", "CAT", "DE", "MMM", "GE",
    "PFE", "MRNA", "BNTX", "GILD", "AMGN", "BMY", "MDT", "ABT", "DHR", "SYK",
])

st.set_page_config(page_title="Equity Drawdown Risk Dashboard", layout="wide")


st.title("Equity Drawdown Risk Scoring Dashboard")
st.caption("Enter any US-listed stock ticker to score its near-term drawdown risk.")

# --- Ticker Input ---
col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker = st.selectbox(
        "Ticker",
        options=POPULAR_TICKERS,
        index=None,
        placeholder="Search ticker (e.g. TSLA, AAPL)...",
        label_visibility="collapsed",
    )

with col_btn:
    analyse = st.button("Analyse", use_container_width=True)

if analyse:
    if not ticker:
        st.warning("Please select a ticker first.")
        st.stop()
    with st.spinner(f"Fetching data and scoring {ticker}..."):
        try:
            result = predict(ticker)
        except ValueError as e:
            st.error(f"Could not analyse **{ticker}**: {e}. This dashboard supports US-listed stocks only.")
            st.stop()

    # --- Top Section: Score, Probability, Classification ---
    col1, col2, col3 = st.columns(3)

    risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[result["classification"]]

    col1.metric("Risk Score", f"{result['risk_score']} / 100")
    col2.metric("Drawdown Probability (20d)", f"{result['probability']:.1%}")
    col3.markdown(
        f"**Risk Classification**<br>"
        f"<span style='font-size:28px; color:{risk_color}'>**{result['classification']}**</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Fetch raw price data for charts ---
    with st.spinner("Loading charts..."):
        price_data = download_ticker(ticker)
        bench_data = download_ticker(BENCHMARK_TICKER)

    # --- Charts ---
    st.subheader("Price History")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=price_data.index, y=price_data["close"],
        mode="lines", name=ticker, line=dict(color="#1f77b4")
    ))
    fig_price.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_price, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rolling Volatility (20-day)")
        daily_returns = price_data["close"].pct_change()
        vol_20d = daily_returns.rolling(20).std() * (252 ** 0.5)
        bench_returns = bench_data["close"].pct_change()
        bench_vol_20d = bench_returns.rolling(20).std() * (252 ** 0.5)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol_20d.index, y=vol_20d,
            mode="lines", name=ticker, line=dict(color="orange")
        ))
        fig_vol.add_trace(go.Scatter(
            x=bench_vol_20d.index, y=bench_vol_20d,
            mode="lines", name="S&P 500", line=dict(color="#aaaaaa", dash="dot")
        ))
        fig_vol.update_layout(
            height=250, margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(tickformat=".0%", title="Annualised Volatility"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_right:
        st.subheader("Drawdown from Rolling 52-Week High")
        rolling_high = price_data["close"].rolling(252).max()
        drawdown = (price_data["close"] - rolling_high) / rolling_high
        worst_idx = drawdown.idxmin()
        worst_val = drawdown.min()
        fig_dd = go.Figure()
        fig_dd.add_hline(y=0, line=dict(color="white", width=1, dash="dot"),
                         annotation_text="At peak", annotation_position="top right")
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown,
            mode="lines", name="Drawdown", line=dict(color="red"),
            fill="tozeroy", fillcolor="rgba(255,0,0,0.15)"
        ))
        fig_dd.add_annotation(
            x=worst_idx, y=worst_val,
            text=f"Worst: {worst_val:.0%}",
            showarrow=True, arrowhead=2, arrowcolor="white",
            font=dict(color="white", size=11), bgcolor="rgba(0,0,0,0.5)",
            ay=-30,
        )
        fig_dd.update_layout(
            height=250, margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(tickformat=".0%", title="% Below Recent Peak"),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # --- Feature Importance ---
    st.subheader("Top Risk Drivers")
    drivers_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in result["top_drivers"].keys()],
        "Importance": list(result["top_drivers"].values()),
    }).sort_values("Importance", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=drivers_df["Importance"],
        y=drivers_df["Feature"],
        orientation="h",
        marker_color="#1f77b4",
        text=[f"{v:.0%}" for v in drivers_df["Importance"]],
        textposition="outside",
    ))
    fig_imp.update_layout(
        height=300,
        margin=dict(l=0, r=60, t=20, b=0),
        xaxis=dict(tickformat=".0%", title="Contribution to Risk Score"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # --- AI Commentary ---
    # st.subheader("AI Analyst Commentary")
    # with st.spinner("Generating commentary..."):
    #     commentary = generate_commentary(result)
    # st.markdown(commentary)
