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

- Risk score (0–100) — composite percentile measure of current signal stress
- Probability of >10% drawdown in next 20 days
- Risk classification (Low / Medium / High)
- SHAP-based driver chart — instance-specific, showing direction and magnitude
- AI-generated analyst commentary (on-demand, rate-limited)
- Supporting charts

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
- Training ticker list (S&P 500 sample — 50–100 diverse stocks across sectors)
- Historical daily price data (yfinance, minimum 2005–2024)

### Data retrieved:
- Close price (adjusted)
- Volume
- Market benchmark (S&P 500 — ticker `^GSPC`)

### Training data construction:
- Download historical data for each training ticker
- Compute features and target variable for each stock independently
- Concatenate all stocks into a single training dataframe
- Expected size: ~250,000+ rows (100 stocks × ~2,500 trading days)

---

## 6. Feature Engineering

### Return Features
- 5-day return
- 20-day return
- 60-day return

### Volatility Features
- 20-day realised volatility
- 60-day realised volatility
- change in volatility

### Drawdown Features
- current drawdown from 252-day high
- rolling max drawdown (60 days)

### Momentum Features
- price vs 50-day moving average
- price vs 200-day moving average

### Market Relative Features
- beta vs S&P 500
- relative return vs benchmark

---

## 7. Target Variable

Binary classification:

```
1 = stock falls more than 10% within next 20 trading days
0 = otherwise
```

Computed by:
- rolling forward window over historical data

---

## 8. Model Design

### Model Type:
- Gradient Boosting Classifier (`sklearn.ensemble.GradientBoostingClassifier`)

### Inputs:
- engineered features (Section 6)

### Output:
- probability of drawdown (0–1)

### Why this model:
- handles non-linear relationships
- robust on tabular financial data
- compatible with SHAP TreeExplainer for instance-level explanations
- no feature scaling required

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
- Risk score (0–100): percentile rank of each feature in this stock's own price history, weighted by GBM feature importances. Measures how stressed current signals are relative to the stock's own past — distinct from probability.
- Risk bucket:
  - Low (<30)
  - Medium (30–70)
  - High (>70)

---

## 10. AI Layer (LLM)

### Purpose:
Convert quantitative outputs — including SHAP direction — into human-readable insight

### LLM: OpenAI GPT-4o-mini

### Rate limiting:
Commentary is generated on-demand via a separate button. Calls are limited to 5 per day across all users, tracked via `model/daily_usage.json`.

---

### LLM Input (structured prompt)

Example:

```
Ticker: TSLA
Risk Score: 74/100 (High)
Probability of >10% drawdown in next 20 trading days: 74.0%

Top risk drivers:
- 20-day annualised volatility is 52% — increasing risk (SHAP: +0.234)
- 60-day annualised volatility is 28% — reducing risk (SHAP: -0.418)
- 20-day return is -8.0% — reducing risk (SHAP: -0.107)
```

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

---

## 11. UI (Streamlit)

### Inputs:
- Ticker (searchable selectbox)

### Outputs:

#### Top Section
- Risk score (with ? tooltip explaining percentile methodology)
- Drawdown probability (with ? tooltip)
- Risk classification

#### Charts
- Price chart
- Rolling volatility vs S&P 500
- Drawdown from rolling 52-week high

#### Model Performance
- AUC-ROC, drawdown precision, drawdown recall, historical base rate

#### SHAP Driver Chart
- Horizontal diverging bar chart
- Red = increasing drawdown risk, blue = reducing it
- Sorted by absolute SHAP value
- Caption explaining log-odds scale

#### AI Commentary (on-demand)
- "Generate AI Commentary" button — LLM only called when clicked
- Shows calls remaining today
- Displays: Risk Summary, Key Drivers (split by direction), What to Watch

#### Methodology expander
- Model, training universe, split, target, features, risk score logic, SHAP explanation

---

## 12. Functional Requirements

- Fetch data from yfinance
- Compute features dynamically
- Run trained ML model
- Compute SHAP values per prediction via TreeExplainer
- Compute composite risk score from historical percentiles
- Generate structured LLM prompt with SHAP direction
- Rate-limit LLM calls to 5/day
- Display results in Streamlit with session state persistence

---

## 13. Non-Functional Requirements

- Clean, modular Python code
- Fast response time (<2–3 seconds per query, excluding LLM)
- Clear separation of: data, model, UI
- No hardcoding of parameters
- Pinned dependencies in requirements.txt

---

## 14. Development Plan

### Phase 1 — Training Data Pipeline
- Define training ticker list (50 S&P 500 stocks, diverse sectors)
- Download adjusted close + volume for each ticker via yfinance (2005–2024)
- Engineer all features per Section 6 for each ticker
- Construct target variable (rolling 20-day forward window)
- Concatenate into single training dataframe
- Apply time-based split: train 2005–2019, test 2020–2024

### Phase 2 — Model Training
- Train Gradient Boosting Classifier with manual sample weights (minority class upweighted)
- Evaluate on test set: precision, recall, AUC-ROC
- Save metrics to `model/metrics.json`
- Serialise model with joblib → `model/risk_model.pkl`

### Phase 3 — Inference Pipeline
- Build single-ticker prediction function:
  - fetch data → compute features → load model → predict probability
- Compute SHAP values via `shap.TreeExplainer` for instance-specific driver explanations
- Compute percentile-based risk score weighted by GBM feature importances
- Derive risk bucket (Low / Medium / High)

### Phase 4 — AI Commentary Layer
- Design structured prompt with SHAP direction (increasing / reducing risk per driver)
- Instruct LLM to narrate signal tension when drivers conflict
- Integrate OpenAI GPT-4o-mini
- Gate commentary behind a separate "Generate AI Commentary" button
- Rate-limit to 5 calls/day via `model/daily_usage.json`

### Phase 5 — Streamlit UI
- Build dashboard layout per Section 11
- Use session state to persist results across interactions
- Diverging SHAP bar chart (red/blue)
- Connect inference pipeline and LLM layer
- Add error handling for invalid tickers and missing model file

### Phase 6 — Polish and Deploy
- Clean, modular code (separate files: data.py, features.py, model.py, llm.py, app.py)
- Write README with setup instructions and example outputs
- Deploy to Streamlit Community Cloud
- Add API key as Streamlit secret (never hardcoded)
- Pin all dependencies in requirements.txt

---

## 15. Deliverables

- GitHub repository with clean, modular code structure:
  ```
  data.py       — yfinance download and cleaning
  features.py   — feature engineering functions
  model.py      — training script and inference (SHAP included)
  llm.py        — prompt construction and LLM call
  app.py        — Streamlit UI
  model/        — serialised model (risk_model.pkl) and metrics (metrics.json)
  README.md     — setup, usage, example outputs
  ```
- Working Streamlit app: https://equity-risk-dashboard-yuqing.streamlit.app/
- Pre-trained serialised model
- Pinned requirements.txt

---

## 16. Limitations

- Uses historical price data only
- No fundamental data
- Model does not predict causality
- Drawdown definition is simplified
- SHAP values are in log-odds space — direction and relative magnitude are meaningful, absolute values are not directly interpretable as probability changes
- LLM output is interpretative, not predictive
- Daily AI commentary limit resets on redeploy

---

## 17. Future Enhancements (v2)

- Multi-ticker batch scoring and risk ranking table
- Daily automated risk report
- Additional features (volume, options implied vol)
- Portfolio-level risk scoring

---

## 18. CV Positioning

- Built and deployed an AI-assisted equity risk scoring tool using Python, Streamlit, and gradient boosting — estimates short-term drawdown probability from volatility, momentum, drawdown, and market-relative features, validated on out-of-sample data from 2020–2024 including COVID and the 2022 rate shock

- Implemented SHAP (SHapley Additive exPlanations) via TreeExplainer to produce instance-specific driver explanations — each prediction shows exactly which signals are increasing or reducing risk for that stock on that day, with a diverging bar chart distinguishing direction

- Designed a composite risk score that percentile-ranks each feature within the stock's own price history, weighted by GBM feature importances — making the score genuinely distinct from the model's probability output and interpretable as a measure of historical stress

- Integrated LLM-generated analyst commentary (GPT-4o-mini) with a structured prompt that uses SHAP direction to separate risk-increasing from risk-reducing drivers, instructing the model to identify and narrate signal tension when factors conflict

---

## 19. Key Insight

> Market risk is not static — it emerges from changing volatility, momentum, and regime dynamics.  
> This project demonstrates how data-driven models, SHAP explainability, and AI can be combined to quantify, explain, and communicate that risk at the individual stock level.
