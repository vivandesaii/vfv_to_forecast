import pandas as pd
import numpy as np
import sys
import os
sys.path.append(".")

from hmm_regime import load_data, train_hmm, identify_calm_state
from forecasting import get_monthly_returns, run_model_competition
from strategy import get_investment_amount

import warnings
warnings.filterwarnings("ignore")


def get_first_trading_day_prices(df):
    """
    Get the first trading day Close price for each month.
    This is the price at which we invest each month.
    """
    return df["Close"].resample("MS").first()


def compute_sharpe(returns, risk_free=0.04):
    """
    Compute annualised Sharpe ratio from monthly return series.
    
    Parameters:
        returns    : pd.Series of monthly portfolio returns
        risk_free  : annual risk-free rate (default 4%)
    
    Returns:
        float — annualised Sharpe ratio
    """
    rf_monthly = risk_free / 12
    excess = returns - rf_monthly
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(12))


def compute_max_drawdown(portfolio_values):
    """
    Compute maximum peak-to-trough drawdown from portfolio value series.
    
    Returns:
        float — max drawdown as negative decimal (e.g. -0.33 = -33%)
    """
    peaks = portfolio_values.cummax()
    drawdowns = (portfolio_values - peaks) / peaks
    return float(drawdowns.min())


def compute_xirr(cashflows, dates):
    """
    Compute XIRR from list of cashflows and corresponding dates.
    Negative cashflows = money invested (outflows).
    Final positive cashflow = portfolio value at end (inflow).
    
    Uses pyxirr library.
    """
    try:
        from pyxirr import xirr
        return xirr(dates, cashflows)
    except Exception as e:
        print(f"XIRR computation failed: {e}")
        return None


def run_backtest(
    df,
    train_end="2022-12-31",
    test_start="2023-01-01",
    test_end="2025-12-31",
    base_sip=500
):
    """
    Walk-forward backtest of Smart SIP vs Vanilla SIP.

    Parameters:
        df         : processed daily DataFrame with Close, daily_return, rolling_vol
        train_end  : end of initial training window
        test_start : start of walk-forward test period
        test_end   : end of test period
        base_sip   : base monthly investment amount in CAD

    Returns:
        results_df : DataFrame with monthly backtest records
        metrics    : dict with XIRR, Sharpe, max drawdown for both strategies
    """

    monthly_prices = get_first_trading_day_prices(df)
    test_months = monthly_prices[
        (monthly_prices.index >= test_start) &
        (monthly_prices.index <= test_end)
    ].index

    print(f"Test months: {len(test_months)} ({test_months[0].date()} → {test_months[-1].date()})")

    # Storage
    records = []

    # Portfolio trackers
    smart_units = 0.0
    vanilla_units = 0.0

    # Cashflow tracking for XIRR
    smart_cashflows = []
    smart_dates = []
    vanilla_cashflows = []
    vanilla_dates = []

    for month in test_months:
        # All data available up to (not including) this month
        data_cutoff = month - pd.DateOffset(days=1)
        daily_data = df[df.index <= data_cutoff]
        monthly_returns = get_monthly_returns(daily_data)

        if len(daily_data) < 100 or len(monthly_returns) < 12:
            print(f"Skipping {month.date()} — insufficient data")
            continue

        # REGIME DETECTION
        try:
            model, states, state_probs, daily_data_enhanced = train_hmm(daily_data)
            calm_state, middle_state, volatile_state = identify_calm_state(
                model, states, daily_data_enhanced
            )
            current_state = states[-1]
            confidence = float(state_probs[-1][current_state])

            if current_state == calm_state:
                regime = "calm"
            elif current_state == volatile_state:
                regime = "volatile"
            else:
                regime = "choppy"
        except Exception as e:
            print(f"HMM failed for {month.date()}: {e}")
            regime = "choppy"
            confidence = 1.0

        # FORECASTING 
        try:
            winner, direction, rmse = run_model_competition(monthly_returns)
        except Exception as e:
            print(f"Forecast failed for {month.date()}: {e}")
            direction = "up"  # default to up on failure
            winner = "ARIMA"

        # ALLOCATION 
        amount = get_investment_amount(regime, direction, confidence=confidence, base=base_sip)
        multiplier = amount / base_sip

        # INVEST 
        price = monthly_prices.get(month)
        if price is None or np.isnan(price):
            print(f"No price for {month.date()}, skipping")
            continue

        # Buy units
        smart_units += amount / price
        vanilla_units += base_sip / price

        # Track cashflows (negative = outflow)
        smart_cashflows.append(-amount)
        smart_dates.append(month.date())
        vanilla_cashflows.append(-base_sip)
        vanilla_dates.append(month.date())

        # Portfolio values at this point
        smart_value = smart_units * price
        vanilla_value = vanilla_units * price

        records.append({
            "date":          month,
            "regime":        regime,
            "direction":     direction,
            "winner":        winner,
            "multiplier":    multiplier,
            "amount":        amount,
            "price":         price,
            "smart_units":   smart_units,
            "vanilla_units": vanilla_units,
            "smart_value":   smart_value,
            "vanilla_value": vanilla_value,
        })

        print(f"{month.date()} | {regime:<8} | {direction:<4} | {winner:<7} | "
              f"${amount:<6} | Smart: ${smart_value:>8.2f} | Vanilla: ${vanilla_value:>8.2f}")

    results_df = pd.DataFrame(records)

    if results_df.empty:
        print("No results generated.")
        return results_df, {}

    # ── FINAL PORTFOLIO VALUES ──
    last_price = results_df["price"].iloc[-1]
    final_smart = results_df["smart_value"].iloc[-1]
    final_vanilla = results_df["vanilla_value"].iloc[-1]

    # Add terminal cashflow for XIRR (positive = inflow)
    last_date = results_df["date"].iloc[-1].date()
    smart_cashflows.append(final_smart)
    smart_dates.append(last_date)
    vanilla_cashflows.append(final_vanilla)
    vanilla_dates.append(last_date)

    # ── MONTHLY RETURNS FOR SHARPE ──
    smart_monthly_returns = results_df["smart_value"].pct_change().dropna()
    vanilla_monthly_returns = results_df["vanilla_value"].pct_change().dropna()

    # ── METRICS ──
    metrics = {
        "smart": {
            "final_value":   round(final_smart, 2),
            "total_invested": round(results_df["amount"].sum(), 2),
            "total_units":   round(results_df["smart_units"].iloc[-1], 4),
            "xirr":          compute_xirr(smart_cashflows, smart_dates),
            "sharpe":        compute_sharpe(smart_monthly_returns),
            "max_drawdown":  compute_max_drawdown(results_df["smart_value"]),
        },
        "vanilla": {
            "final_value":   round(final_vanilla, 2),
            "total_invested": round(len(results_df) * base_sip, 2),
            "total_units":   round(results_df["vanilla_units"].iloc[-1], 4),
            "xirr":          compute_xirr(vanilla_cashflows, vanilla_dates),
            "sharpe":        compute_sharpe(vanilla_monthly_returns),
            "max_drawdown":  compute_max_drawdown(results_df["vanilla_value"]),
        }
    }

    return results_df, metrics


def print_metrics(metrics):
    """Pretty print comparison metrics."""
    print("\n" + "="*55)
    print(f"{'METRIC':<20} {'SMART SIP':>15} {'VANILLA SIP':>15}")
    print("="*55)
    s = metrics["smart"]
    v = metrics["vanilla"]
    print(f"{'Final Value':<20} ${s['final_value']:>14,.2f} ${v['final_value']:>14,.2f}")
    print(f"{'Total Invested':<20} ${s['total_invested']:>14,.2f} ${v['total_invested']:>14,.2f}")
    print(f"{'Total Units':<20} {s['total_units']:>15.4f} {v['total_units']:>15.4f}")
    print(f"{'XIRR':<20} {str(round(s['xirr']*100,2))+'%':>15} {str(round(v['xirr']*100,2))+'%':>15}")
    print(f"{'Sharpe Ratio':<20} {s['sharpe']:>15.4f} {v['sharpe']:>15.4f}")
    print(f"{'Max Drawdown':<20} {str(round(s['max_drawdown']*100,2))+'%':>15} {str(round(v['max_drawdown']*100,2))+'%':>15}")
    print("="*55)


if __name__ == "__main__":
    df = load_data()
    results, metrics = run_backtest(df)

    if not results.empty:
        print_metrics(metrics)

        os.makedirs("data", exist_ok=True)
        results.to_csv("data/backtest_results.csv", index=False)
        print("\nResults saved to data/backtest_results.csv")