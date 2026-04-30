# PRD: AI-Assisted Equity Drawdown Risk Scoring Dashboard

---

## 1. Problem Statement

In equity markets, investors care not just about returns but about downside risk.

Traditional models assume stable conditions, but in reality:
- volatility changes over time
- momentum shifts quickly
- drawdowns cluster
- market conditions are regime-dependent

There is no simple, accessible tool that:
1. Quantifies short-term downside risk
2. Explains what is driving that risk — for this specific stock, today
3. Presents it in an interpretable, analyst-style format

---

## 2. Objective

Build a Python-based application that:

1. Estimates the probability that a stock experiences a >10% drawdown over the next 20 trading days
2. Uses ML to model this probability from market features
3. Uses SHAP to explain each individual prediction — not generic averages
4. Uses an LLM to generate clear, analyst-style risk commentary
5. Displays results via a Streamlit dashboard

---

## 3. Core Output

For a given ticker (e.g. TSLA), the system outputs:

- Risk score (0–100) — percentile-based composite measuring how stressed current signals are relative to this stock's own history
- Probability of >10% drawdown in next 20 days — model-implied likelihood
- Risk classification (Low / Medium / High)
- SHAP-based driver chart — instance-specific, showing direction and magnitude
- AI-generated analyst commentary (on-demand, rate-limited to 5 calls/day)
- Supporting charts: price history, rolling volatility vs S&P 500, drawdown from 52-week high

---

## 4. System Architecture

```
yfinance → feature engineering → ML model → probability
                                           → SHAP values → driver chart
                                           → risk score (percentile composite)
         → structured prompt (SHAP-informed) → LLM → commentary
         → Streamlit UI
```

Two distinct phases:
- **Training phase** (offline, run once): download multi-ticker dataset, engineer features, construct target, train and serialise model
- **Inference phase** (live app): load serialised model, fetch single ticker, compute features, score, compute SHAP values, optionally generate commentary

---

## 5. Data Pipeline

### Input:
- Ticker (user input, inference)
- Training ticker list (S&P 500 sample — 50 diverse stocks across sectors)
- Historical daily price data (yfinance, 2005–2024)

### Data retrieved:
- Close price (adjusted)
- Volume
- Market benchmark (S&P 500 — ticker `^GSPC`)

### Training data construction:
- Download historical data for each training ticker
- Compute features and target variable for each stock independently
- Concatenate all stocks into a single training dataframe
- Expected size: ~250,000+ rows (50 stocks × ~2,500 trading days)

---

## 6. Feature Engineering

### Return Features
- 5-day return
- 20-day return
- 60-day return

### Volatility Features
- 20-day realised volatility (annualised)
- 60-day realised volatility (annualised)
- change in volatility (vol_20d / vol_60d ratio)

### Drawdown Features
- current drawdown from 252-day high
- rolling max drawdown (60 days)

### Momentum Features
- price vs 50-day moving average
- price vs 200-day moving average

### Market Relative Features
- beta vs S&P 500 (rolling 60-day)
- relative return vs benchmark (20-day)

---

## 7. Target Variable

Binary classification:

```
1 = stock falls more than 10% within next 20 trading days
0 = otherwise
```

Computed by rolling forward window over historical close prices. Positive rate in training data: ~14%.

---

## 8. Model Design

### Model Type:
- Gradient Boosting Classifier (`sklearn.ensemble.GradientBoostingClassifier`)
- `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `random_state=42`

### Output:
- Probability of drawdown (0–1) via `predict_proba`

### Why this model:
- Handles non-linear relationships
- Robust on tabular financial data
- Compatible with SHAP TreeExplainer for instance-level explanations
- No feature scaling required

### Train / Test Split — TIME-BASED (CRITICAL)
Financial time-series requires a chronological split. Random splits create lookahead bias.

```
Train:      2005–2019  (~15 years, multiple market regimes)
Test:        2020–2024  (~4 years, includes COVID crash, 2022 rate shock)
Split date: 2020-01-01
```

**Never** use `train_test_split(shuffle=True)` on time-series data.

### Class Imbalance
The positive class (drawdown > 10% in 20 days) occurs ~14% of the time.
Handle with manual sample weights: minority class observations are upweighted by the ratio of class counts during training.
Note: `GradientBoostingClassifier` does not support `class_weight='balanced'` — sample weights are passed directly to `model.fit()`.

### Model Persistence
Train the model offline once. Serialise with joblib:

```python
import joblib
joblib.dump(model, 'model/risk_model.pkl')   # training
model = joblib.load('model/risk_model.pkl')   # app startup
```

The Streamlit app loads the pre-trained model at startup — it does **not** retrain on every query.

---

## 9. Model Outputs

For each prediction:
- Probability of drawdown (0–1)
- SHAP values per feature (instance-specific, via `shap.TreeExplainer`)

Derived:
- **Risk score (0–100):** percentile rank of each feature in this stock's own price history, weighted by GBM feature importances. For features where a higher value means higher risk (vol_20d, vol_60d, vol_change, beta), the raw percentile is used. For return and drawdown features, the percentile is flipped (low return = high risk). Measures how stressed current signals are relative to the stock's own past — entirely distinct from probability.
- Risk bucket:
  - Low (< 30)
  - Medium (30–70)
  - High (> 70)

### Why risk score ≠ probability
The risk score captures *how extreme* today's conditions look relative to this stock's own history. The probability captures *what the model predicts will happen* based on those conditions. A stock that is always volatile (e.g. TSLA) can have a low risk score (normal conditions for it) but high probability (because high volatility is predictive of drawdowns cross-sectionally). Both metrics are useful and intentionally distinct.

---

## 10. Model Performance (out-of-sample, 2020–2024)

Metrics saved to `model/metrics.json` and displayed in the app:
- **AUC-ROC** — overall ranking ability (1.0 = perfect, 0.5 = random)
- **PR-AUC** — Precision-Recall AUC; more informative than ROC-AUC on imbalanced data
- **Drawdown Recall** — of all actual drawdowns, how many the model caught
- **Historical Base Rate** — unconditional frequency of >10% drawdowns (~14%)

Note on precision: the model is deliberately tuned for recall (catching actual drawdowns) via sample upweighting. This reduces precision, which is an intentional tradeoff — it is better to flag a risk that doesn't materialise than to miss one that does.

---

## 11. AI Layer (LLM)

### Purpose:
Convert quantitative outputs — including SHAP direction — into human-readable insight

### LLM: OpenAI GPT-4o-mini

### Rate limiting:
Commentary is generated on-demand via a separate "Generate AI Commentary" button. Calls are limited to 5 per day across all users (shared between both tabs), tracked via `model/daily_usage.json`. `daily_usage.json` is excluded from version control via `.gitignore`.

---

### LLM Input (structured prompt)

```
Ticker: TSLA
Risk Score: 74/100 (High)
Probability of >10% drawdown in next 20 trading days: 74.0%

Top risk drivers:
- 20-day annualised volatility is 52% — increasing risk (SHAP: +0.234)
- 60-day annualised volatility is 28% — reducing risk (SHAP: -0.418)
- 20-day return is -8.0% — increasing risk (SHAP: +0.107)
```

Each driver line includes: feature description, actual value, direction (increasing/reducing risk based on SHAP sign), and SHAP value.

---

### LLM Output structure

```
**Risk Summary**
2 sentences: risk score + probability, then dominant narrative (tension between signals if mixed, single driver if one-directional).

**Key Drivers**
Increasing risk:
- Feature, actual value, what it signals, SHAP value

Reducing risk:
- Feature, actual value, why it acts as a buffer, SHAP value

**What to Watch**
- Specific quantitative forward signal tied to the dominant tension
- Momentum or price-structure signal
- Market-relative signal
```

When drivers conflict (e.g. short-term vol increasing but long-term vol low), the LLM is instructed to narrate the tension explicitly rather than averaging or ignoring it.

---

## 12. UI (Streamlit) — Single Stock Tab

### Inputs:
- Ticker (searchable selectbox from ~80 popular US tickers)

### Outputs:

#### Top Section
- Risk score (with ? tooltip explaining percentile methodology)
- Drawdown probability (with ? tooltip)
- Risk classification (colour-coded: green/orange/red)
- Caption explaining the distinction between score and probability
- Caption comparing current probability to historical base rate

#### Model Performance
- AUC-ROC, PR-AUC, Drawdown Recall, Historical Base Rate — with ? tooltips
- Caption noting these are out-of-sample test metrics

#### Charts
- Price chart (full history)
- Rolling 20-day volatility vs S&P 500
- Drawdown from rolling 52-week high (with worst-point annotation)

#### SHAP Driver Chart
- Horizontal diverging bar chart
- Red = increasing drawdown risk, blue = reducing it
- Sorted by absolute SHAP value (most impactful at top)
- Caption explaining log-odds scale

#### AI Analyst Commentary (on-demand)
- "Generate AI Commentary" button — LLM only called when clicked
- Shows calls remaining today
- Displays: Risk Summary, Key Drivers (split by direction), What to Watch
- Commentary persists in session state — re-clicking Analyse does not regenerate it

#### Methodology expander
- Model, training universe, split, target, features, risk score logic, SHAP explanation
- Appears below AI Commentary (separated by a divider)

---

## 12b. UI — Watchlist Screener Tab

### Inputs:
- Preset watchlist selector: Mag 7, US Tech, US Financials, US Healthcare
- Custom ticker input (comma-separated)

### Outputs:

#### Risk Ranking Table
- All tickers scored using the same full SHAP pipeline as the single stock view
- Sorted by risk score descending
- Columns: Rank, Ticker, Risk Score, Probability, Classification (colour-coded), Top Driver, Direction (red/blue coloured)
- Risk Score column with red-to-green background gradient (`RdYlGn_r`)
- Rendered as HTML via pandas Styler `.to_html()` — ensures consistent center-alignment across all columns; `st.dataframe` was abandoned as it overrides inline CSS alignment
- Failed tickers skipped gracefully with a warning

#### Drill-Down
- Selectbox to pick any scored ticker
- Bridge sentence: "{ticker} is ranked #X of Y in this watchlist, primarily driven by {feature} — {direction}"
- Renders the full single stock report (charts, SHAP bar, methodology, AI commentary)
- Uses stored batch results — does not re-run the model
- Commentary stored per ticker in `st.session_state.screener_commentary` dict — switching tickers does not lose generated text
- Methodology expander appears below AI Commentary (separated by a divider)

---

## 13. File Structure

```
app.py          — Streamlit dashboard (UI, session state, rate limiter)
data.py         — yfinance download and cleaning
features.py     — feature engineering functions
model.py        — training script (run offline) and inference pipeline (SHAP included)
llm.py          — structured prompt construction and OpenAI GPT-4o-mini call
requirements.txt
model/
    risk_model.pkl      — serialised trained model (joblib)
    metrics.json        — test set evaluation metrics
    daily_usage.json    — daily API call counter (gitignored)
```

### Key implementation details per file

**app.py:**
- `FEATURE_LABELS` dict maps internal feature names to display labels
- `WATCHLISTS` dict defines preset watchlists
- `_load_usage()`, `_save_usage()`, `_try_consume_call()` — file-based daily rate limiter
- `_render_report(result, price_data, bench_data)` — shared renderer used by both tabs (metrics, charts, SHAP)
- `_render_methodology()` — standalone methodology expander, called after commentary in both tabs
- Session state keys: `result`, `price_data`, `bench_data`, `commentary`, `analysed_ticker`, `screener_rows`, `screener_full`, `screener_drill`, `screener_drill_price`, `screener_drill_bench`, `screener_commentary` (dict)

**model.py:**
- `TRAINING_TICKERS` — 50 large-cap US equities across sectors
- `FEATURE_COLS` — ordered list of 12 feature names
- `build_dataset()` — downloads all tickers, engineers features and target, concatenates
- `train(data)` — time-based split, sample weights, fits GBM, saves model and metrics
- `predict(ticker)` — full inference: fetch → features → model score → SHAP → percentile risk score
- `HIGHER_IS_RISKIER = {"vol_20d", "vol_60d", "vol_change", "beta"}` — determines percentile flip direction for risk score
- SHAP: `shap.TreeExplainer(model).shap_values(latest)` returns `[class0, class1]`; take `[1][0]` for class 1, first row

**llm.py:**
- `build_prompt(result)` — formats top drivers with actual feature values, SHAP sign-derived direction, and SHAP value
- `generate_commentary(result)` — calls GPT-4o-mini with system prompt instructing analyst-style output with signal tension narration
- API key loaded from environment variable `OPENAI_API_KEY` (`.env` locally, Streamlit secrets on cloud)

---

## 14. Functional Requirements

- Fetch data from yfinance
- Compute features dynamically at inference time
- Run trained ML model (loaded once at app startup)
- Compute SHAP values per prediction via TreeExplainer
- Compute composite risk score from historical percentiles weighted by feature importances
- Generate structured LLM prompt with SHAP direction
- Rate-limit LLM calls to 5/day (shared across both tabs) via file-based counter
- Display results in Streamlit with session state persistence
- Batch-score a watchlist with progress tracking and graceful error handling
- Rank and display watchlist results in a styled HTML table
- Drill into any watchlist ticker for full report without re-running the model
- Store per-ticker commentary in session state so switching tickers preserves generated text

---

## 15. Non-Functional Requirements

- Clean, modular Python code across separate files
- Fast response time (<2–3 seconds per query, excluding LLM)
- No hardcoding of parameters or API keys
- Pinned dependencies in requirements.txt

---

## 16. Dependencies (pinned)

```
yfinance==1.3.0
pandas==3.0.2
numpy==2.3.5
scikit-learn==1.8.0
joblib==1.5.3
shap==0.51.0
openai==2.32.0
python-dotenv
streamlit==1.56.0
plotly==6.7.0
matplotlib==3.10.9
```

Note: `matplotlib` is required by pandas `background_gradient()` even though no matplotlib charts are rendered directly.
Note: pandas 3.0+ renamed `.applymap()` to `.map()` — use `.map()` for cell-level Styler functions.

---

## 17. Deployment

- Hosted on Streamlit Community Cloud
- Live app: https://equity-risk-dashboard-yuqing.streamlit.app/
- GitHub repo: https://github.com/Aleiyqwang/equity-risk-dashboard
- OpenAI API key stored as a Streamlit secret (never hardcoded or committed)
- `model/daily_usage.json` excluded from version control

---

## 18. Limitations

- Uses historical price data only — no fundamental or options data
- Model does not predict causality
- Drawdown definition is simplified (any close below -10% from entry within 20 days)
- SHAP values are in log-odds space — direction and relative magnitude are meaningful; absolute values are not directly interpretable as probability changes
- LLM output is interpretative, not predictive
- Daily AI commentary limit resets on redeploy (counter is not persisted across deployments)
- Precision is intentionally sacrificed for recall via sample upweighting

---

## 19. Shipped Features

### v1
- Single stock analysis: fetch → features → ML score → SHAP → commentary
- Percentile-based risk score (distinct from probability)
- SHAP diverging bar chart (instance-specific, red/blue)
- AI commentary behind a separate button (not auto-called)
- Rate limiting: 5 LLM calls/day via file-based counter
- Model performance metrics: AUC-ROC, PR-AUC, Recall, Base Rate
- Risk score vs probability explanatory caption
- Methodology expander

### v2
- Multi-ticker watchlist screener with batch scoring and progress bar
- Risk ranking table: Rank, Ticker, Risk Score (gradient), Probability, Classification (colour), Top Driver, Direction (colour)
- Table rendered as HTML via pandas Styler for reliable center-alignment
- Drill-down from screener into full single stock report (no model re-run)
- Bridge sentence showing rank and primary driver for drilled ticker
- AI commentary in both Single Stock and Watchlist Screener drill-down
- Per-ticker commentary persisted in session state
- Commentary above Methodology (divider between them)

---

## 20. Future Enhancements (v3)

- Daily automated risk report
- Additional features (volume, options implied vol)
- Portfolio-level risk scoring

---

## 21. CV Positioning

- Built and deployed an AI-assisted equity risk scoring tool using Python, Streamlit, and gradient boosting — estimates short-term drawdown probability from volatility, momentum, drawdown, and market-relative features, validated on out-of-sample data from 2020–2024 including COVID and the 2022 rate shock

- Implemented SHAP (SHapley Additive exPlanations) via TreeExplainer to produce instance-specific driver explanations — each prediction shows exactly which signals are increasing or reducing risk for that stock on that day, with a diverging bar chart distinguishing direction

- Designed a composite risk score that percentile-ranks each feature within the stock's own price history, weighted by GBM feature importances — making the score genuinely distinct from the model's probability output and interpretable as a measure of historical signal stress

- Integrated LLM-generated analyst commentary (GPT-4o-mini) with a structured prompt that uses SHAP direction to separate risk-increasing from risk-reducing drivers, instructing the model to identify and narrate signal tension when factors conflict

- Built a multi-ticker watchlist screener that batch-scores preset or custom watchlists (Mag 7, US Tech, US Financials, US Healthcare), ranks results by drawdown risk in a colour-coded HTML table with SHAP-derived top driver per stock, and allows drill-down into the full report for any ticker without re-running the model

---

## 22. Key Insight

> Market risk is not static — it emerges from changing volatility, momentum, and regime dynamics.
> This project demonstrates how data-driven models, SHAP explainability, and AI can be combined to quantify, explain, and communicate that risk — both at the individual stock level and across an entire watchlist.
