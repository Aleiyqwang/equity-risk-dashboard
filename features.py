import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Takes single-ticker price data and benchmark data.
    Returns a dataframe with all engineered features.
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    bench = benchmark["close"]

    # --- Return features ---
    feat["return_5d"] = close.pct_change(5)
    feat["return_20d"] = close.pct_change(20)
    feat["return_60d"] = close.pct_change(60)

    # --- Volatility features ---
    daily_returns = close.pct_change()
    feat["vol_20d"] = daily_returns.rolling(20).std() * np.sqrt(252)
    feat["vol_60d"] = daily_returns.rolling(60).std() * np.sqrt(252)
    feat["vol_change"] = feat["vol_20d"] - feat["vol_60d"]

    # --- Drawdown features ---
    rolling_high_252 = close.rolling(252).max()
    feat["drawdown_from_high"] = (close - rolling_high_252) / rolling_high_252

    rolling_max_60 = close.rolling(60).max()
    rolling_min_60 = close.rolling(60).min()
    feat["max_drawdown_60d"] = (rolling_min_60 - rolling_max_60) / rolling_max_60

    # --- Momentum features ---
    ma_50 = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean()
    feat["price_vs_ma50"] = (close - ma_50) / ma_50
    feat["price_vs_ma200"] = (close - ma_200) / ma_200

    # --- Market relative features ---
    bench_aligned = bench.reindex(close.index).ffill()
    bench_returns = bench_aligned.pct_change()

    # Beta: covariance(stock, bench) / variance(bench) over 60 days
    feat["beta"] = (
        daily_returns.rolling(60).cov(bench_returns)
        / bench_returns.rolling(60).var()
    )

    # Relative return vs benchmark over 20 days
    feat["relative_return_20d"] = feat["return_20d"] - bench_aligned.pct_change(20)

    return feat


def add_target(df: pd.DataFrame, close: pd.Series, threshold: float = 0.10, window: int = 20) -> pd.DataFrame:
    """
    Adds a binary target column: 1 if the stock drops >10% within the next 20 trading days.
    Uses shift(-window) to look forward in time.
    """
    future_min = close.rolling(window).min().shift(-window)
    forward_return = (future_min - close) / close
    df["target"] = (forward_return < -threshold).astype(int)
    return df


def add_features_to_training_data(combined: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to each ticker in the combined training dataframe.
    Returns the combined dataframe with feature columns added.
    """
    all_frames = []

    for ticker, group in combined.groupby("ticker"):
        features = compute_features(group, benchmark)
        features = add_target(features, group["close"])
        features["ticker"] = ticker
        all_frames.append(features)

    return pd.concat(all_frames)


if __name__ == "__main__":
    from data import download_all, download_ticker, TRAINING_TICKERS, BENCHMARK_TICKER

    print("Downloading data...")
    combined = download_all(TRAINING_TICKERS)
    benchmark = download_ticker(BENCHMARK_TICKER)

    print("\nEngineering features...")
    features = add_features_to_training_data(combined, benchmark)

    print("\n=== Sample features (first 10 rows) ===")
    print(features.dropna().head(10).to_string())

    print("\n=== Feature columns ===")
    print([c for c in features.columns if c != "ticker"])

    print("\n=== Shape (after dropping NaN rows) ===")
    print(features.dropna().shape)

    print("\n=== Target distribution ===")
    counts = features["target"].value_counts()
    print(f"No drawdown (0): {counts.get(0, 0)}")
    print(f"Drawdown >10% (1): {counts.get(1, 0)}")
    print(f"Positive rate: {counts.get(1, 0) / counts.sum():.1%}")
