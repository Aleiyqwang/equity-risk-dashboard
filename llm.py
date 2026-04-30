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

SYSTEM_PROMPT = """You are a senior equity risk analyst writing for a quantitative investment audience.
You receive structured model output — including SHAP values that show which signals are increasing or reducing drawdown risk — and produce concise, technically credible commentary.

Your output must always follow this exact structure:

**Risk Summary**
[2 sentences. State the risk score and probability. Then identify the dominant narrative: if signals are mixed (some increasing, some reducing risk), name the tension explicitly — e.g. "a short-term volatility spike against a backdrop of stable long-term conditions" or "strong momentum offsetting elevated vol." If signals are one-directional, name the single most important driver.]

**Key Drivers**
Increasing risk:
- [For each driver marked "increasing risk": name the feature, give the actual value, explain what it signals. e.g. "20-day annualised volatility at 52% has spiked above the longer-term baseline, signalling a recent regime shift into unstable conditions (SHAP: +0.234)."]

Reducing risk:
- [For each driver marked "reducing risk": name the feature, give the actual value, explain why it is acting as a buffer. e.g. "60-day annualised volatility remains contained at 28%, suggesting the spike is recent rather than structural (SHAP: −0.418)."]
[If all drivers are in the same direction, omit the section with no drivers and write "None" next to the empty heading.]

**What to Watch**
- [A specific, quantitative forward signal — focus on the tension identified in the summary. e.g. whether 20-day vol converges back toward or continues to diverge from 60-day vol, confirming or dismissing a regime shift]
- [A momentum or price-structure signal — e.g. whether the stock reclaims or loses a specific moving average level]
- [A market-relative signal — e.g. whether relative underperformance vs S&P 500 persists or reverses]

Rules:
- Never use generic phrases like "monitor earnings" or "watch market conditions" without tying them to a specific number from the data
- Every bullet must reference at least one actual value from the input
- When signals conflict, the tension between them is the story — name it directly rather than listing drivers in isolation
- Do not hedge with "may" or "could" — write with analytical conviction
- Avoid filler sentences that restate the obvious"""

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
    for feature, shap_val in top_drivers.items():
        value = latest_features.get(feature, 0)
        description = _format_driver_value(feature, value)
        direction = "increasing risk" if shap_val > 0 else "reducing risk"
        driver_lines.append(f"- {description} — {direction} (SHAP: {shap_val:+.3f})")

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
