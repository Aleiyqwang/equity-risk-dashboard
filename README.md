# Equity Drawdown Risk Scoring Dashboard

A machine learning dashboard that scores the near-term drawdown risk of any US-listed stock. Enter a ticker and the app predicts the probability of a >10% price drop within the next 20 trading days, explains what is driving the risk for that specific stock today, and generates AI analyst commentary.

**Live app:** https://equity-risk-dashboard-yuqing.streamlit.app/

---

## What it does

- Fetches live price data from Yahoo Finance
- Scores the stock 0–100 and classifies it as Low / Medium / High risk
- Shows the probability of a >10% drawdown in the next 20 trading days
- Visualises price history, rolling volatility vs S&P 500, and drawdown from peak
- Explains the top risk drivers using SHAP values — instance-specific, not global averages
- Generates analyst-style commentary via GPT-4o-mini (rate-limited to 5 calls/day)

## How it works

| Layer | Detail |
|---|---|
| **Data** | Adjusted close prices via `yfinance`, S&P 500 as benchmark |
| **Features** | 12 engineered features: volatility, momentum, drawdown, beta, moving averages |
| **Model** | `GradientBoostingClassifier` trained on 50 S&P 500 stocks (2005–2019), tested on 2020–2024 |
| **Imbalance handling** | Sample weights to compensate for rare drawdown events (~14% of data) |
| **Risk Score** | Percentile rank of each feature in this stock's own history, weighted by GBM feature importances — distinct from the drawdown probability |
| **Explainability** | SHAP values (via `TreeExplainer`) show how each signal pushed this specific prediction up or down |
| **LLM** | OpenAI GPT-4o-mini generates Risk Summary, Key Drivers (split by direction), and What to Watch |
| **UI** | Streamlit + Plotly |

## Project structure

```
app.py          # Streamlit dashboard
data.py         # Data download (yfinance)
features.py     # Feature engineering
model.py        # Model training and inference
llm.py          # GPT-4o-mini commentary
requirements.txt
model/          # Serialised model and metrics
```

## Run locally

**1. Clone the repo**
```bash
git clone https://github.com/Aleiyqwang/equity-risk-dashboard.git
cd equity-risk-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-key-here
```

**4. Launch the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Tech stack

- Python 3.11
- Streamlit
- scikit-learn
- shap
- yfinance
- Plotly
- OpenAI API
