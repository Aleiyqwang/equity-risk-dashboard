import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Use Streamlit secrets when deployed, fall back to .env locally
def _get_api_key() -> str:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a professional equity risk analyst. You receive structured quantitative data
about a stock's downside risk and generate concise, clear analyst-style commentary.

Your output must always follow this exact structure:

**Risk Summary**
[2 sentences summarising the overall risk level and the main reason for it]

**Key Drivers**
- [Driver 1: plain English explanation of what the feature value means]
- [Driver 2: plain English explanation]
- [Driver 3: plain English explanation]
- [Driver 4: plain English explanation]
- [Driver 5: plain English explanation]

**What to Watch**
- [Forward-looking signal 1]
- [Forward-looking signal 2]
- [Forward-looking signal 3]

Be specific, factual, and avoid generic filler. Use the actual numbers provided."""

client = OpenAI(api_key=_get_api_key())


def _format_driver_value(feature: str, value: float) -> str:
    descriptions = {
        "vol_60d": f"60-day annualised volatility is {value:.0%} (market avg ~15%)",
        "vol_20d": f"20-day annualised volatility is {value:.0%}",
        "vol_change": f"volatility has {'risen' if value > 0 else 'fallen'} by {abs(value):.0%} recently",
        "return_5d": f"5-day return is {value:+.1%}",
        "return_20d": f"20-day return is {value:+.1%}",
        "return_60d": f"60-day return is {value:+.1%}",
        "drawdown_from_high": f"stock is {abs(value):.0%} below its 52-week high",
        "max_drawdown_60d": f"max drawdown over 60 days is {abs(value):.0%}",
        "price_vs_ma50": f"price is {value:+.0%} vs its 50-day moving average",
        "price_vs_ma200": f"price is {value:+.0%} vs its 200-day moving average",
        "beta": f"beta is {value:.2f}x vs S&P 500",
        "relative_return_20d": f"20-day return vs S&P 500 is {value:+.1%}",
    }
    return descriptions.get(feature, f"{feature}: {value:.4f}")


def generate_commentary(prediction: dict) -> str:
    """
    Takes a prediction dict from model.predict() and returns LLM analyst commentary.
    """
    ticker = prediction["ticker"]
    risk_score = prediction["risk_score"]
    probability = prediction["probability"]
    classification = prediction["classification"]
    top_drivers = prediction["top_drivers"]
    latest_features = prediction["latest_features"]

    driver_lines = []
    for feature, importance in top_drivers.items():
        value = latest_features.get(feature, 0)
        description = _format_driver_value(feature, value)
        driver_lines.append(f"- {description} (model importance: {importance:.1%})")

    user_prompt = f"""Ticker: {ticker}
Risk Score: {risk_score}/100 ({classification})
Probability of >10% drawdown in next 20 trading days: {probability:.1%}

Top risk drivers:
{chr(10).join(driver_lines)}

Generate analyst commentary following the required structure."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from model import predict

    print("Running inference for TSLA...")
    prediction = predict("TSLA")

    print(f"\nRisk Score: {prediction['risk_score']}/100 ({prediction['classification']})")
    print(f"Probability: {prediction['probability']:.1%}\n")

    print("Generating AI commentary...\n")
    commentary = generate_commentary(prediction)
    print(commentary)
