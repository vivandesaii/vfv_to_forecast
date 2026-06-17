import yfinance as yf
import pandas as pd
import os

def load_vfv(start_date="2012-01-01", end_date="2025-12-31", save_csv=True):
    """
    Download VFV.TO and VIX daily data, compute returns, rolling vol,
    vol_ratio, and VIX. Drop NaN warmup rows. Save to data/vfv_processed.csv.
    """
    # Download VFV.TO
    df = yf.download("VFV.TO", start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].copy()
    df["daily_return"] = df["Close"].pct_change()
    df["rolling_vol"] = df["daily_return"].rolling(window=21).std()

    # Download VIX
    vix = yf.download("^VIX", start=start_date, end=end_date)
    vix.columns = vix.columns.get_level_values(0)
    vix = vix[["Close"]].rename(columns={"Close": "vix"})

    # Merge on date index
    df = df.join(vix, how="left")

    # Forward fill VIX gaps (holidays where VIX has no data)
    df["vix"] = df["vix"].ffill()

    # Drop NaN warmup rows
    df.dropna(inplace=True)

    if save_csv:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/vfv_processed.csv")

    print(f"Shape:      {df.shape}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"\n{df[['daily_return', 'rolling_vol', 'vix']].describe().round(4)}")

    return df

if __name__ == "__main__":
    load_vfv()