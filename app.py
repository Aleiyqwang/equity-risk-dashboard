import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from model import predict, FEATURE_COLS
from llm import generate_commentary
from data import download_ticker, BENCHMARK_TICKER

load_dotenv()

st.set_page_config(page_title="Equity Drawdown Risk Dashboard", layout="wide")
st.title("Equity Drawdown Risk Scoring Dashboard")

# --- Ticker Input ---
ticker = st.text_input("Enter a stock ticker", value="TSLA").upper().strip()

if st.button("Analyse"):
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
    st.plotly_chart(fig_price, width="stretch")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rolling Volatility (20-day)")
        daily_returns = price_data["close"].pct_change()
        vol_20d = daily_returns.rolling(20).std() * (252 ** 0.5)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol_20d.index, y=vol_20d,
            mode="lines", name="Vol 20d", line=dict(color="orange")
        ))
        fig_vol.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_vol, width="stretch")

    with col_right:
        st.subheader("Drawdown from 52-Week High")
        rolling_high = price_data["close"].rolling(252).max()
        drawdown = (price_data["close"] - rolling_high) / rolling_high
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown,
            mode="lines", name="Drawdown", line=dict(color="red"),
            fill="tozeroy", fillcolor="rgba(255,0,0,0.1)"
        ))
        fig_dd.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_dd, width="stretch")

    st.divider()

    # --- Feature Importance ---
    st.subheader("Model: Top Risk Drivers")
    drivers_df = pd.DataFrame({
        "Feature": list(result["top_drivers"].keys()),
        "Importance": list(result["top_drivers"].values()),
    }).sort_values("Importance", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=drivers_df["Importance"],
        y=drivers_df["Feature"],
        orientation="h",
        marker_color="#1f77b4",
    ))
    fig_imp.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_imp, width="stretch")

    st.divider()

    # --- AI Commentary ---
    st.subheader("AI Analyst Commentary")
    with st.spinner("Generating commentary..."):
        commentary = generate_commentary(result)
    st.markdown(commentary)
