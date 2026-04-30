import json
import os
from datetime import date
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from model import predict, FEATURE_COLS
from llm import generate_commentary
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

METRICS_PATH = os.path.join(os.path.dirname(__file__), "model", "metrics.json")
USAGE_PATH = os.path.join(os.path.dirname(__file__), "model", "daily_usage.json")
DAILY_LIMIT = 5

try:
    with open(METRICS_PATH) as f:
        MODEL_METRICS = json.load(f)
except FileNotFoundError:
    MODEL_METRICS = None


# --- Rate limiter ---

def _load_usage() -> dict:
    today = date.today().isoformat()
    try:
        with open(USAGE_PATH) as f:
            data = json.load(f)
        if data.get("date") != today:
            return {"date": today, "count": 0}
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"date": today, "count": 0}


def _save_usage(data: dict):
    with open(USAGE_PATH, "w") as f:
        json.dump(data, f)


def _try_consume_call() -> tuple[bool, int]:
    """Atomically check and increment the daily counter. Returns (allowed, calls_remaining)."""
    usage = _load_usage()
    if usage["count"] >= DAILY_LIMIT:
        return False, 0
    usage["count"] += 1
    _save_usage(usage)
    return True, DAILY_LIMIT - usage["count"]


# --- Session state init ---

for key in ("result", "price_data", "bench_data", "commentary", "analysed_ticker"):
    if key not in st.session_state:
        st.session_state[key] = None


# --- Page ---

st.set_page_config(page_title="Equity Drawdown Risk Dashboard", layout="wide")

st.title("📉 Equity Drawdown Risk Scoring Dashboard")
st.caption("Machine learning-powered downside risk scoring for US equities.")

st.markdown("""
Uses a **Gradient Boosting model** trained on 50 S&P 500 stocks (2005–2019) to estimate the probability
that a stock will fall **more than 10% within the next 20 trading days**.
For educational and research purposes — not financial advice.
""")

st.markdown("📊 Risk Score &nbsp;·&nbsp; 📈 Price & Volatility Charts &nbsp;·&nbsp; 🎯 Top Risk Drivers &nbsp;·&nbsp; 🤖 AI Analyst Commentary &nbsp;·&nbsp; 🔬 Model Performance")

st.divider()

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

    # Clear old commentary if the ticker changed
    if ticker != st.session_state.analysed_ticker:
        st.session_state.commentary = None

    with st.spinner(f"Fetching data and scoring {ticker}..."):
        try:
            st.session_state.result = predict(ticker)
        except FileNotFoundError:
            st.error("Model file not found. Please ensure `model/risk_model.pkl` exists before running the app.")
            st.stop()
        except ValueError as e:
            st.error(f"Could not analyse **{ticker}**: {e}. This dashboard supports US-listed stocks only.")
            st.stop()

    with st.spinner("Loading chart data..."):
        st.session_state.price_data = download_ticker(ticker)
        st.session_state.bench_data = download_ticker(BENCHMARK_TICKER)

    st.session_state.analysed_ticker = ticker


# --- Results (shown whenever a result is stored, survives button re-clicks) ---

if st.session_state.result is not None:
    result = st.session_state.result
    price_data = st.session_state.price_data
    bench_data = st.session_state.bench_data

    # --- Top Section: Score, Probability, Classification ---
    col1, col2, col3 = st.columns(3)

    risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[result["classification"]]

    col1.metric("Risk Score", f"{result['risk_score']} / 100",
                help="Composite score (0–100) that measures how stressed this stock's risk signals are relative to its own price history. Each of the 12 model features is ranked as a percentile within this stock's historical distribution, then weighted by the GBM's feature importances. A score of 80 means current signals look worse than 80% of this stock's own trading history.")
    col2.metric("Drawdown Probability (20d)", f"{result['probability']:.1%}",
                help="The model's predicted probability that this stock will fall more than 10% at any point in the next 20 trading days, based on current market features.")
    col3.markdown(
        f"**Risk Classification**<br>"
        f"<span style='font-size:28px; color:{risk_color}'>**{result['classification']}**</span>",
        unsafe_allow_html=True,
    )

    if MODEL_METRICS:
        base_rate = MODEL_METRICS["base_rate"]
        st.caption(
            f"Historical base rate for >10% drawdowns: **{base_rate:.1%}** — "
            f"model probability of **{result['probability']:.1%}** reflects "
            f"{'elevated' if result['probability'] > base_rate * 1.5 else 'near-average'} risk "
            f"vs the unconditional average."
        )

    st.divider()

    if MODEL_METRICS:
        st.subheader("Model Performance")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("AUC-ROC", f"{MODEL_METRICS['auc_roc']:.2f}", help="1.0 = perfect, 0.5 = random. Measures overall discrimination ability.")
        mc2.metric("Drawdown Precision", f"{MODEL_METRICS['drawdown_precision']:.0%}", help="When the model flags high risk, how often a drawdown actually occurs.")
        mc3.metric("Drawdown Recall", f"{MODEL_METRICS['drawdown_recall']:.0%}", help="Of all actual drawdowns, how many the model successfully identified.")
        mc4.metric("Historical Base Rate", f"{MODEL_METRICS['base_rate']:.1%}", help="How often a >10% drawdown occurs on any given day historically.")
        st.caption(f"Evaluated on out-of-sample test data ({MODEL_METRICS['test_period']}), never seen during training.")
        st.divider()

    # --- Charts ---
    st.subheader("Price History")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=price_data.index, y=price_data["close"],
        mode="lines", name=result["ticker"], line=dict(color="#1f77b4")
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
            mode="lines", name=result["ticker"], line=dict(color="orange")
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

    # --- SHAP Drivers ---
    st.subheader("Top Risk Drivers")
    st.caption("SHAP values show how each signal pushed this prediction up or down from the model baseline. Red = increasing drawdown risk, blue = reducing it.")

    shap_items = list(result["top_drivers"].items())
    # Sort by absolute value descending, then reverse so largest is at top of horizontal chart
    shap_items = sorted(shap_items, key=lambda x: abs(x[1]))
    features = [FEATURE_LABELS.get(f, f) for f, _ in shap_items]
    shap_vals = [v for _, v in shap_items]
    bar_colors = ["#d62728" if v > 0 else "#4878cf" for v in shap_vals]

    fig_imp = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in shap_vals],
        textposition="outside",
    ))
    fig_imp.add_vline(x=0, line=dict(color="white", width=1))
    fig_imp.update_layout(
        height=320,
        margin=dict(l=0, r=70, t=20, b=0),
        xaxis=dict(title="SHAP value (contribution to drawdown probability, log-odds)"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # --- AI Commentary ---
    st.subheader("AI Analyst Commentary")

    if st.session_state.commentary is not None:
        # Already generated — just display it
        st.markdown(st.session_state.commentary)
    else:
        usage = _load_usage()
        remaining = DAILY_LIMIT - usage["count"]

        if remaining <= 0:
            st.warning(
                "The daily AI commentary limit (5 calls/day) has been reached. "
                "Check back tomorrow."
            )
        else:
            st.caption(f"AI commentary uses the OpenAI API. Limit: {remaining} call{'s' if remaining != 1 else ''} remaining today.")
            if st.button("Generate AI Commentary", use_container_width=False):
                allowed, calls_left = _try_consume_call()
                if not allowed:
                    st.warning("Daily limit reached — no AI commentaries left today.")
                else:
                    with st.spinner("Generating commentary..."):
                        st.session_state.commentary = generate_commentary(result)
                    st.rerun()

    st.divider()

    with st.expander("Methodology"):
        m = MODEL_METRICS or {}
        st.markdown(f"""
**Model:** {m.get('algorithm', 'Gradient Boosting Classifier')} (`scikit-learn`)

**Training universe:** {m.get('n_tickers', 50)} large-cap US equities across technology, financials, healthcare, energy, and industrials

**Training period:** {m.get('train_period', '2005–2019')} · **Out-of-sample test:** {m.get('test_period', '2020–2024')} (includes COVID crash, 2022 rate shock)

**Target variable:** Binary — did the stock fall >10% at any point in the next 20 trading days?

**Features ({m.get('n_features', 12)}):** Short, medium, and long-term returns; 20-day and 60-day annualised volatility and volatility regime shift; drawdown from rolling 52-week high; max drawdown over 60 days; price vs 50-day and 200-day moving averages; rolling beta vs S&P 500; relative return vs S&P 500

**Class imbalance:** Drawdown events occur ~{m.get('base_rate', 0.138):.0%} of trading days. Addressed via sample weighting — drawdown observations were upweighted ~6× during training to prevent the model from ignoring rare but important events.

**Split method:** Strict time-based split (no random shuffling) to prevent data leakage — the model never sees future data during training.

**Risk Score (0–100):** A composite measure of how stressed the current risk signals are relative to this stock's own price history. Each of the 12 features is percentile-ranked within the stock's full historical distribution, then weighted by the GBM's learned feature importances. Distinct from the drawdown probability — the score reflects current market conditions; the probability reflects what the model predicts will happen next.

**Top Risk Drivers (SHAP):** Feature contributions are computed using SHAP (SHapley Additive exPlanations) applied to each individual prediction. Unlike global feature importances, SHAP values are instance-specific — they show how much each signal pushed this particular prediction up or down from the model's baseline. Positive values increase the predicted drawdown probability; negative values reduce it.
        """)
