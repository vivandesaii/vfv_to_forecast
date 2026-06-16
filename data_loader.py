import yfinance as yf
import pandas as pd
import os

def load_vfv(start_date="2012-01-01", end_date="2025-12-31", save_csv=True):
    """
    Download VFV.TO daily price data, compute returns and rolling
    volatility, drop NaN warmup rows, and save to data/vfv_processed.csv.
    """
    # Download
    df = yf.download("VFV.TO", start=start_date, end=end_date)

    # Fix yfinance 1.2.0 MultiIndex columns
    df.columns = df.columns.get_level_values(0)

    # Keep only Close
    df = df[["Close"]].copy()

    # Daily return
    df["daily_return"] = df["Close"].pct_change()

    # 21-day rolling volatility (raw, not annualized)
    df["rolling_vol"] = df["daily_return"].rolling(window=21).std()

    # Drop NaN warmup rows
    df.dropna(inplace=True)

    if save_csv:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/vfv_processed.csv")

    print(f"Shape:      {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\n{df[['daily_return', 'rolling_vol']].describe().round(4)}")

    return df

if __name__ == "__main__":
    load_vfv()