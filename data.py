import yfinance as yf
import pandas as pd

# Small sample of tickers to test with — we'll expand to 50-100 later
TRAINING_TICKERS = ["AAPL", "MSFT", "TSLA", "JPM", "XOM"]
BENCHMARK_TICKER = "^GSPC"

START_DATE = "2005-01-01"
END_DATE = "2024-12-31"


def download_ticker(ticker: str) -> pd.DataFrame:
    """Download adjusted close and volume for a single ticker."""
    raw = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if raw.empty:
        print(f"  WARNING: no data returned for {ticker}")
        return pd.DataFrame()

    df = raw[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df["ticker"] = ticker
    return df


def download_all(tickers: list[str]) -> pd.DataFrame:
    """Download data for all tickers and concatenate into one dataframe."""
    frames = []
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = download_ticker(ticker)
        if not df.empty:
            frames.append(df)
    combined = pd.concat(frames)
    combined.index.name = "date"
    return combined


if __name__ == "__main__":
    print("=== Downloading training tickers ===")
    data = download_all(TRAINING_TICKERS)

    print("\n=== Sample output (first 10 rows) ===")
    print(data.head(10))

    print("\n=== Shape ===")
    print(f"Rows: {len(data)}, Columns: {list(data.columns)}")

    print("\n=== Tickers present ===")
    print(data["ticker"].unique())

    print("\n=== Date range ===")
    print(f"From {data.index.min().date()} to {data.index.max().date()}")

    print("\n=== Downloading benchmark (S&P 500) ===")
    benchmark = download_ticker(BENCHMARK_TICKER)
    print(benchmark.head(5))
