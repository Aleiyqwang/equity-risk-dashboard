import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

from data import download_all, download_ticker, BENCHMARK_TICKER
from features import add_features_to_training_data

TRAINING_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "XOM", "CVX", "UNH", "HD", "PG", "MA", "V", "BAC", "ABBV", "MRK",
    "PFE", "KO", "PEP", "AVGO", "COST", "TMO", "MCD", "ACN", "LIN", "DHR",
    "WMT", "NEE", "TXN", "PM", "UPS", "RTX", "QCOM", "HON", "INTU", "IBM",
    "GS", "MS", "BLK", "SPGI", "AXP", "LOW", "CAT", "DE", "MMM", "GE",
]

SPLIT_DATE = "2020-01-01"
MODEL_PATH = "model/risk_model.pkl"

FEATURE_COLS = [
    "return_5d", "return_20d", "return_60d",
    "vol_20d", "vol_60d", "vol_change",
    "drawdown_from_high", "max_drawdown_60d",
    "price_vs_ma50", "price_vs_ma200",
    "beta", "relative_return_20d",
]


def build_dataset() -> pd.DataFrame:
    print(f"Downloading {len(TRAINING_TICKERS)} tickers...")
    combined = download_all(TRAINING_TICKERS)

    print("Downloading benchmark...")
    benchmark = download_ticker(BENCHMARK_TICKER)

    print("Engineering features and target...")
    data = add_features_to_training_data(combined, benchmark)
    data = data.dropna()

    print(f"Dataset shape: {data.shape}")
    print(f"Positive rate: {data['target'].mean():.1%}")
    return data


def train(data: pd.DataFrame):
    train_data = data[data.index < SPLIT_DATE]
    test_data = data[data.index >= SPLIT_DATE]

    print(f"\nTrain rows: {len(train_data)} | Test rows: {len(test_data)}")

    X_train = train_data[FEATURE_COLS]
    y_train = train_data["target"]
    X_test = test_data[FEATURE_COLS]
    y_test = test_data["target"]

    # Compute sample weights to upweight the minority class (drawdown=1)
    class_counts = y_train.value_counts()
    weight_map = {0: 1.0, 1: class_counts[0] / class_counts[1]}
    sample_weights = y_train.map(weight_map)

    print("Training model...")
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    print("\n=== Test set evaluation ===")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")

    print("\n=== Feature importances ===")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importances.to_string())

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 3),
        "drawdown_precision": round(report["1"]["precision"], 3),
        "drawdown_recall": round(report["1"]["recall"], 3),
        "base_rate": round(float(y_train.mean()), 3),
        "n_tickers": len(TRAINING_TICKERS),
        "train_period": "2005–2019",
        "test_period": "2020–2024",
        "n_features": len(FEATURE_COLS),
        "algorithm": "Gradient Boosting Classifier",
        "n_estimators": 200,
    }
    os.makedirs("model", exist_ok=True)
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Done.")


def predict(ticker: str) -> dict:
    """
    Full inference pipeline for a single ticker.
    Fetches live data, computes features, scores with the saved model.
    Returns a dict with probability, risk score, classification, and top drivers.
    """
    model = joblib.load(MODEL_PATH)

    print(f"Fetching data for {ticker}...")
    stock_data = download_ticker(ticker)
    benchmark = download_ticker(BENCHMARK_TICKER)

    if stock_data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    from features import compute_features
    features = compute_features(stock_data, benchmark)
    features = features.dropna()

    if features.empty:
        raise ValueError(f"Not enough data to compute features for: {ticker}")

    latest = features[FEATURE_COLS].iloc[[-1]]
    probability = model.predict_proba(latest)[0][1]

    # Risk score: percentile rank of each current feature in this stock's own history,
    # weighted by the GBM's feature importances.
    # Distinct from probability — measures how stressed signals are vs the stock's own past.
    HIGHER_IS_RISKIER = {"vol_20d", "vol_60d", "vol_change", "beta"}
    history = features[FEATURE_COLS]
    current = latest.iloc[0]
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)

    weighted_pct = 0.0
    for col in FEATURE_COLS:
        pct = (history[col] < current[col]).mean() * 100  # 0–100 percentile
        if col not in HIGHER_IS_RISKIER:
            pct = 100 - pct  # flip: low value = high risk for return/drawdown/momentum features
        weighted_pct += pct * importances[col]

    risk_score = round(weighted_pct)

    if risk_score < 30:
        classification = "Low"
    elif risk_score <= 70:
        classification = "Medium"
    else:
        classification = "High"

    import shap
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(latest)
    # GradientBoostingClassifier returns [class0_vals, class1_vals]; take class 1 (drawdown), first row
    shap_vals = raw_shap[1][0] if isinstance(raw_shap, list) else raw_shap[0]
    shap_series = pd.Series(shap_vals, index=FEATURE_COLS)
    top_drivers = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index).head(5)

    return {
        "ticker": ticker,
        "probability": round(probability, 3),
        "risk_score": risk_score,
        "classification": classification,
        "top_drivers": top_drivers.to_dict(),
        "latest_features": latest.iloc[0].to_dict(),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        ticker = sys.argv[2] if len(sys.argv) > 2 else "TSLA"
        result = predict(ticker)
        print(f"\n=== Risk Score: {result['ticker']} ===")
        print(f"Probability of >10% drawdown: {result['probability']}")
        print(f"Risk score: {result['risk_score']}/100")
        print(f"Classification: {result['classification']}")
        print(f"\nTop drivers:")
        for feature, importance in result["top_drivers"].items():
            value = result["latest_features"][feature]
            print(f"  {feature}: importance={importance:.3f}, current value={value:.4f}")
    else:
        data = build_dataset()
        train(data)
