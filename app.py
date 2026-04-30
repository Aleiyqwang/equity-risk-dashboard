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

# --- Constants ---

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

WATCHLISTS = {
    "Mag 7":        ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "US Tech":      ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ADBE", "CRM", "QCOM"],
    "US Financials":["JPM", "BAC", "GS", "MS", "WFC", "BLK", "V", "MA", "AXP", "SPGI"],
    "US Healthcare":["UNH", "JNJ", "ABBV", "LLY", "PFE", "MRK", "TMO", "DHR", "ABT", "AMGN"],
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
    usage = _load_usage()
    if usage["count"] >= DAILY_LIMIT:
        return False, 0
    usage["count"] += 1
    _save_usage(usage)
    return True, DAILY_LIMIT - usage["count"]


# --- Session state init ---

for key in ("result", "price_data", "bench_data", "commentary", "analysed_ticker",
            "screener_rows", "screener_full", "screener_drill",
            "screener_drill_price", "screener_drill_bench"):
    if key not in st.session_state:
        st.session_state[key] = None

if "screener_commentary" not in st.session_state:
    st.session_state.screener_commentary = {}  # ticker → commentary string


# --- Shared report renderer (used by both tabs) ---

def _render_report(result: dict, price_data: pd.DataFrame, bench_data: pd.DataFrame):
    """Renders score metrics, charts, and SHAP drivers for a given prediction."""

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
    st.caption("Risk score measures current signal stress relative to this stock's own history. Drawdown probability is the model-implied likelihood of a >10% drop in the next 20 trading days. A stock can show a low risk score but high probability if it is inherently volatile — the two metrics are intentionally distinct.")

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
        mc1.metric("AUC-ROC", f"{MODEL_METRICS['auc_roc']:.2f}", help="1.0 = perfect, 0.5 = random. Measures overall ranking ability across all thresholds.")
        mc2.metric("PR-AUC", f"{MODEL_METRICS['pr_auc']:.2f}" if "pr_auc" in MODEL_METRICS else "—", help="Precision-Recall AUC. More informative than ROC-AUC on imbalanced data — measures how well the model identifies drawdowns without excessive false alarms.")
        mc3.metric("Drawdown Recall", f"{MODEL_METRICS['drawdown_recall']:.0%}", help="Of all actual drawdowns, how many the model successfully identified.")
        mc4.metric("Historical Base Rate", f"{MODEL_METRICS['base_rate']:.1%}", help="How often a >10% drawdown occurs on any given day historically.")
        st.caption(f"Evaluated on out-of-sample test data ({MODEL_METRICS['test_period']}), never seen during training.")
        st.divider()

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

    st.subheader("Top Risk Drivers")
    st.caption("SHAP values show how each signal pushed this prediction up or down from the model baseline. Red = increasing drawdown risk, blue = reducing it.")

    shap_items = sorted(result["top_drivers"].items(), key=lambda x: abs(x[1]))
    features = [FEATURE_LABELS.get(f, f) for f, _ in shap_items]
    shap_vals = [v for _, v in shap_items]
    bar_colors = ["#d62728" if v > 0 else "#4878cf" for v in shap_vals]

    fig_imp = go.Figure(go.Bar(
        x=shap_vals, y=features, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in shap_vals],
        textposition="outside",
    ))
    fig_imp.add_vline(x=0, line=dict(color="white", width=1))
    fig_imp.update_layout(
        height=320, margin=dict(l=0, r=70, t=20, b=0),
        xaxis=dict(title="SHAP value (contribution to drawdown probability, log-odds)"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)



def _render_methodology():
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


# --- Page ---

st.set_page_config(page_title="Equity Drawdown Risk Dashboard", layout="wide")

st.title("📉 Equity Drawdown Risk Scoring Dashboard")
st.caption("Machine learning-powered downside risk scoring for US equities.")
st.markdown("""
Uses a **Gradient Boosting model** trained on 50 S&P 500 stocks (2005–2019) to estimate the probability
that a stock will fall **more than 10% within the next 20 trading days**.
For educational and research purposes — not financial advice.
""")
st.divider()

tab1, tab2 = st.tabs(["Single Stock", "Watchlist Screener"])


# ── TAB 1: Single Stock ────────────────────────────────────────────────────────

with tab1:
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

        if ticker != st.session_state.analysed_ticker:
            st.session_state.commentary = None

        with st.spinner(f"Fetching data and scoring {ticker}..."):
            try:
                st.session_state.result = predict(ticker)
            except FileNotFoundError:
                st.error("Model file not found. Please ensure `model/risk_model.pkl` exists.")
                st.stop()
            except ValueError as e:
                st.error(f"Could not analyse **{ticker}**: {e}. This dashboard supports US-listed stocks only.")
                st.stop()

        with st.spinner("Loading chart data..."):
            st.session_state.price_data = download_ticker(ticker)
            st.session_state.bench_data = download_ticker(BENCHMARK_TICKER)

        st.session_state.analysed_ticker = ticker

    if st.session_state.result is not None:
        _render_report(
            st.session_state.result,
            st.session_state.price_data,
            st.session_state.bench_data,
        )

        # AI commentary — only in single stock tab
        st.subheader("AI Analyst Commentary")
        if st.session_state.commentary is not None:
            st.markdown(st.session_state.commentary)
        else:
            usage = _load_usage()
            remaining = DAILY_LIMIT - usage["count"]
            if remaining <= 0:
                st.warning("The daily AI commentary limit (5 calls/day) has been reached. Check back tomorrow.")
            else:
                st.caption(f"AI commentary uses the OpenAI API. Limit: {remaining} call{'s' if remaining != 1 else ''} remaining today.")
                if st.button("Generate AI Commentary", use_container_width=False):
                    allowed, _ = _try_consume_call()
                    if not allowed:
                        st.warning("Daily limit reached — no AI commentaries left today.")
                    else:
                        with st.spinner("Generating commentary..."):
                            st.session_state.commentary = generate_commentary(st.session_state.result)
                        st.rerun()

        st.divider()
        _render_methodology()


# ── TAB 2: Watchlist Screener ──────────────────────────────────────────────────

with tab2:
    st.markdown("Score an entire watchlist at once and rank stocks by drawdown risk.")
    st.caption("Uses the same full SHAP analysis as the single stock view — no shortcuts.")

    # Watchlist selection
    col_wl, col_custom = st.columns([2, 3])
    with col_wl:
        preset = st.selectbox(
            "Preset watchlist",
            options=list(WATCHLISTS.keys()) + ["Custom"],
            index=0,
        )
    with col_custom:
        if preset == "Custom":
            custom_input = st.text_input(
                "Enter tickers (comma-separated)",
                placeholder="e.g. AAPL, TSLA, JPM, NVDA",
            )
            tickers_to_scan = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        else:
            tickers_to_scan = WATCHLISTS[preset]
            st.markdown(f"**Tickers:** {', '.join(tickers_to_scan)}")

    run_screener = st.button("Run Screener", use_container_width=False)

    if run_screener:
        if not tickers_to_scan:
            st.warning("Please enter at least one ticker.")
        else:
            st.session_state.screener_rows = []
            st.session_state.screener_full = {}
            st.session_state.screener_drill = None
            st.session_state.screener_commentary = {}

            failed = []
            progress_bar = st.progress(0, text="Starting...")

            for i, t in enumerate(tickers_to_scan):
                progress_bar.progress((i + 1) / len(tickers_to_scan), text=f"Scoring {t}... ({i+1}/{len(tickers_to_scan)})")
                try:
                    res = predict(t)
                    top_feature = max(res["top_drivers"], key=lambda k: abs(res["top_drivers"][k]))
                    top_val = res["top_drivers"][top_feature]

                    st.session_state.screener_rows.append({
                        "Ticker": t,
                        "Risk Score": res["risk_score"],
                        "Probability": res["probability"],
                        "Classification": res["classification"],
                        "Top Driver": FEATURE_LABELS.get(top_feature, top_feature),
                        "Direction": "Increasing risk" if top_val > 0 else "Reducing risk",
                    })
                    st.session_state.screener_full[t] = res
                except Exception:
                    failed.append(t)

            progress_bar.empty()

            if failed:
                st.warning(f"Could not score: {', '.join(failed)}. These tickers were skipped.")

    # Results table
    if st.session_state.screener_rows:
        results_df = pd.DataFrame(st.session_state.screener_rows).sort_values("Risk Score", ascending=False).reset_index(drop=True)
        results_df.insert(0, "Rank", range(1, len(results_df) + 1))

        def _style_classification(val):
            colors = {"Low": "color: #2ca02c", "Medium": "color: orange", "High": "color: #d62728"}
            return colors.get(val, "")

        def _style_direction(val):
            return "color: #d62728" if val == "Increasing risk" else "color: #4878cf"

        styled = (
            results_df.style
            .map(_style_classification, subset=["Classification"])
            .map(_style_direction, subset=["Direction"])
            .background_gradient(subset=["Risk Score"], cmap="RdYlGn_r", vmin=0, vmax=100)
            .format({"Probability": "{:.1%}", "Risk Score": "{:.0f}"})
            .set_properties(**{"text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center"), ("padding", "8px 32px"), ("background-color", "#1e1e1e"), ("color", "#fafafa"), ("font-weight", "600"), ("border-bottom", "1px solid #444")]},
                {"selector": "td", "props": [("text-align", "center !important"), ("padding", "7px 32px"), ("border-bottom", "1px solid #2a2a2a")]},
                {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse"), ("font-size", "14px")]},
                {"selector": "tr:hover td", "props": [("background-color", "#2a2a2a")]},
            ])
            .hide(axis="index")
        )

        st.subheader("Risk Ranking")
        st.caption("Ranked by model-implied downside risk. Scores and probabilities are model outputs — not financial advice. Drill into any stock below for the full report.")
        st.markdown(styled.to_html(), unsafe_allow_html=True)

        st.divider()

        # Drill-down
        st.subheader("Drill Into a Stock")
        scored_tickers = [row["Ticker"] for row in st.session_state.screener_rows]
        drill_ticker = st.selectbox(
            "Select ticker for full report",
            options=scored_tickers,
            index=None,
            placeholder="Select a ticker...",
            label_visibility="collapsed",
            key="drill_selectbox",
        )

        if drill_ticker and drill_ticker != st.session_state.screener_drill:
            st.session_state.screener_drill = drill_ticker
            with st.spinner(f"Loading chart data for {drill_ticker}..."):
                st.session_state.screener_drill_price = download_ticker(drill_ticker)
                st.session_state.screener_drill_bench = download_ticker(BENCHMARK_TICKER)

        if st.session_state.screener_drill and st.session_state.screener_drill_price is not None:
            drill_result = st.session_state.screener_full[st.session_state.screener_drill]
            drill_row = next(r for r in st.session_state.screener_rows if r["Ticker"] == st.session_state.screener_drill)
            sorted_tickers = [r["Ticker"] for r in sorted(st.session_state.screener_rows, key=lambda r: r["Risk Score"], reverse=True)]
            rank = sorted_tickers.index(st.session_state.screener_drill) + 1
            total = len(sorted_tickers)
            st.caption(
                f"**{st.session_state.screener_drill}** is ranked **#{rank} of {total}** in this watchlist, "
                f"primarily driven by **{drill_row['Top Driver']}** — {drill_row['Direction'].lower()}."
            )
            _render_report(
                drill_result,
                st.session_state.screener_drill_price,
                st.session_state.screener_drill_bench,
            )

            # AI commentary for drill-down — stored per ticker so switching stocks doesn't lose it
            st.subheader("AI Analyst Commentary")
            drill_ticker_key = st.session_state.screener_drill
            if drill_ticker_key in st.session_state.screener_commentary:
                st.markdown(st.session_state.screener_commentary[drill_ticker_key])
            else:
                usage = _load_usage()
                remaining = DAILY_LIMIT - usage["count"]
                if remaining <= 0:
                    st.warning("The daily AI commentary limit (5 calls/day) has been reached. Check back tomorrow.")
                else:
                    st.caption(f"AI commentary uses the OpenAI API. Limit: {remaining} call{'s' if remaining != 1 else ''} remaining today.")
                    if st.button("Generate AI Commentary", use_container_width=False, key="screener_commentary_btn"):
                        allowed, _ = _try_consume_call()
                        if not allowed:
                            st.warning("Daily limit reached — no AI commentaries left today.")
                        else:
                            with st.spinner("Generating commentary..."):
                                st.session_state.screener_commentary[drill_ticker_key] = generate_commentary(drill_result)
                            st.rerun()

            st.divider()
            _render_methodology()
