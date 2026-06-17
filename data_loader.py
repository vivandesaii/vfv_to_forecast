import yfinance as yf
import pandas as pd
from arch import arch_model
import os

def add_garch_vol(df):
    """
    Fit GARCH(1,1) on daily returns and extract conditional volatility.
    Better captures volatility clustering than rolling std.
    """
    model = arch_model(df["daily_return"] * 100, vol="Garch", p=1, q=1)
    result = model.fit(disp="off")
    df["garch_vol"] = result.conditional_volatility / 100
    return df


def load_vfv(start_date="2012-01-01", end_date="2025-12-31", save_csv=True):
    """
    Download VFV.TO and VIX daily data, compute returns, rolling vol,
    GARCH conditional volatility, and VIX. Save to data/vfv_processed.csv.
    """
    # Download VFV.TO
    df = yf.download("VFV.TO", start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].copy()
    df["daily_return"] = df["Close"].pct_change()
    df["rolling_vol"] = df["daily_return"].rolling(window=21).std()

    # Download VIX
    vix_raw = yf.download("^VIX", start=start_date, end=end_date)
    if vix_raw is None or vix_raw.empty:
        raise RuntimeError("yfinance returned no data for ^VIX")
    vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix = vix_raw[["Close"]].rename(columns={"Close": "vix"})

    # Merge VIX
    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()

    # Drop NaN warmup rows before GARCH
    df.dropna(inplace=True)

    # Add GARCH conditional volatility
    df = add_garch_vol(df)

    if save_csv:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/vfv_processed.csv")

    print(f"Shape:      {df.shape}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"\n{df[['daily_return', 'rolling_vol', 'garch_vol', 'vix']].describe().round(4)}")

    return df

if __name__ == "__main__":
    load_vfv()