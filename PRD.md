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
2. Explains what is driving that risk
3. Presents it in an interpretable, analyst-style format

---

## 2. Objective

Build a Python-based application that:

1. Estimates the probability that a stock experiences a >10% drawdown over the next 20 trading days  
2. Uses ML to model this probability from market features  
3. Uses an LLM to generate clear, analyst-style risk commentary  
4. Displays results via a Streamlit dashboard  

---

## 3. Core Output

For a given ticker (e.g. TSLA), the system outputs:

- Risk score (0–100)
- Probability of >10% drawdown in next 20 days
- Risk classification (Low / Medium / High)
- Key drivers (model features)
- AI-generated analyst commentary
- Supporting charts

---

## 4. System Architecture

```
yfinance → feature engineering → ML model → probability + drivers
         → structured prompt → LLM → commentary
         → Streamlit UI
```

Two distinct phases:
- **Training phase** (offline, run once): download multi-ticker dataset, engineer features, construct target, train and serialise model
- **Inference phase** (live app): load serialised model, fetch single ticker, compute features, score, generate commentary

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
- interpretable via feature importance
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
The positive class (drawdown > 10% in 20 days) occurs ~20–25% of the time.
Handle with: `class_weight='balanced'` in the classifier constructor.
This prevents the model from always predicting "no drawdown".

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
- Probability of drawdown
- Feature importance (global or local)

Derived:
- Risk score (0–100)
- Risk bucket:
  - Low (<30)
  - Medium (30–70)
  - High (>70)

---

## 10. AI Layer (LLM)

### Purpose:
Convert quantitative outputs into human-readable insight

---

### LLM Input (structured prompt)

Example:

```
Ticker: TSLA
Risk score: 74/100 (High)
Probability of >10% drawdown in 20 days: 0.74

Top drivers:
- Volatility: elevated (38% annualised)
- Momentum: negative (-12% over 60 days)
- Drawdown: recent (-18% from peak)
- Beta: 1.6 vs S&P 500

Generate:
1. 2-sentence risk summary
2. Bullet-point key drivers
3. 2–3 forward-looking signals to monitor
```

---

### LLM Output

Example:

```
TSLA is currently exhibiting elevated downside risk, driven by a combination of high volatility and weakening momentum. The stock is trading in a less stable regime, increasing the likelihood of further short-term losses.

Key drivers:
- Elevated volatility indicates unstable price behaviour
- Negative momentum reflects sustained selling pressure
- Recent drawdown suggests weak recovery dynamics

What to watch:
- Whether volatility stabilises
- Changes in relative performance vs S&P 500
- Break of recent support levels
```

---

## 11. UI (Streamlit)

### Inputs:
- Ticker (text input)

### Outputs:

#### Top Section
- Risk score
- Probability
- Risk classification

#### Charts
- Price chart
- Rolling volatility
- Drawdown chart

#### Model Insights
- Feature importance
- Key drivers

#### AI Commentary
- Analyst-style summary
- Drivers
- Forward-looking signals

---

## 12. Functional Requirements

- Fetch data from yfinance
- Compute features dynamically
- Run trained ML model
- Generate structured LLM prompt
- Display results in Streamlit

---

## 13. Non-Functional Requirements

- Clean, modular Python code
- Fast response time (<2–3 seconds per query)
- Clear separation of:
  - data
  - model
  - UI
- No hardcoding of parameters

---

## 14. Development Plan

### Phase 1 — Training Data Pipeline
- Define training ticker list (50–100 S&P 500 stocks, diverse sectors)
- Download adjusted close + volume for each ticker via yfinance (2005–2024)
- Engineer all features per Section 6 for each ticker
- Construct target variable (rolling 20-day forward window)
- Concatenate into single training dataframe
- Apply time-based split: train 2005–2019, test 2020–2024

### Phase 2 — Model Training
- Train Gradient Boosting Classifier with `class_weight='balanced'`
- Evaluate on test set: precision, recall, AUC-ROC
- Extract feature importances
- Serialise model with joblib → `model/risk_model.pkl`

### Phase 3 — Inference Pipeline
- Build single-ticker prediction function:
  - fetch data → compute features → load model → predict probability
- Derive risk score (0–100) and risk bucket (Low / Medium / High)
- Output feature importances for top drivers

### Phase 4 — AI Commentary Layer
- Design structured prompt template (Section 10)
- Integrate LLM API (Claude Haiku or GPT-4o-mini)
- Generate: summary, key drivers, forward-looking signals
- Cache output per ticker per day (`st.cache_data`)

### Phase 5 — Streamlit UI
- Build dashboard layout per Section 11
- Connect inference pipeline and LLM layer
- Add error handling for invalid tickers

### Phase 6 — Polish and Deploy
- Clean, modular code (separate files: data.py, features.py, model.py, llm.py, app.py)
- Write README with setup instructions and example outputs
- Deploy to Streamlit Community Cloud
- Add API key as Streamlit secret (never hardcoded)

---

## 15. Deliverables

- GitHub repository with clean, modular code structure:
  ```
  data.py       — yfinance download and cleaning
  features.py   — feature engineering functions
  model.py      — training script (run once offline)
  llm.py        — prompt construction and LLM call
  app.py        — Streamlit UI
  model/        — serialised model (risk_model.pkl)
  README.md     — setup, usage, example outputs
  ```
- Working Streamlit app with live URL
- Pre-trained serialised model
- Example screenshots in README

---

## 16. Limitations

- Uses historical price data only
- No fundamental data
- Model does not predict causality
- Drawdown definition is simplified
- LLM output is interpretative, not predictive

---

## 17. Future Enhancements

- Multi-ticker batch scoring
- Daily automated risk report
- Additional features (volume, options data)
- SHAP-based explanations
- Portfolio-level risk scoring

---

## 18. CV Positioning

- Built and deployed an AI-assisted equity risk scoring tool using Python, Streamlit, and gradient boosting — estimates short-term drawdown probability from volatility, momentum, drawdown, and market-relative features, validated on out-of-sample data from 2020–2024 including COVID and the 2022 rate shock

- Integrated LLM-generated analyst commentary to translate model outputs into actionable financial risk insights, using a structured prompt architecture that separates quantitative scoring from natural language generation

---

## 19. Key Insight

> Market risk is not static — it emerges from changing volatility, momentum, and regime dynamics.  
> This project demonstrates how data-driven models and AI can be combined to quantify and interpret that risk.