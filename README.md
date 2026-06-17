# Smart SIP Engine — VFV.TO

A machine learning system that dynamically adjusts monthly ETF investments
based on market regime detection and return forecasting, then validates
the approach through rigorous walk-forward backtesting.

## What It Does

Instead of investing a fixed $500/month regardless of market conditions
(vanilla SIP), this engine adjusts the monthly investment between $450–$600
based on two signals:

1. **Market Regime** — detected by a 3-state Gaussian HMM trained on
   GARCH volatility, VIX, vol ratio, and daily returns
2. **Return Direction** — forecasted by a monthly competition between
   ARIMA, ETS, and Facebook Prophet

| Regime   | Forecast | Multiplier | Amount |
|----------|----------|------------|--------|
| Calm     | Up       | 1.20x      | $600   |
| Calm     | Down     | 0.85x      | $425   |
| Choppy   | Up       | 1.10x      | $550   |
| Choppy   | Down     | 0.90x      | $450   |
| Volatile | Up       | 1.00x      | $500   |
| Volatile | Down     | 1.00x      | $500   |

Multipliers scale toward 1.0x when HMM confidence is low.

## Results (V6 — Final)

Tested walk-forward: train 2012–2022, test 2023–2025.

| Metric        | Smart SIP (V6) | Vanilla SIP |
|---------------|----------------|-------------|
| Final Value   | $28,542        | $24,896     |
| Total Invested| $20,618        | $18,000     |
| XIRR          | 23.46%         | 23.46%      |
| Sharpe Ratio  | **2.5366**     | 2.3211      |
| Max Drawdown  | -3.18%         | -2.93%      |

Smart SIP matched vanilla on annualized return (XIRR) while achieving
9.3% better risk-adjusted return (Sharpe) and deploying $2,618 less capital.

## Iteration History

| Version         | Sharpe | XIRR   | Drawdown | Invested |
|-----------------|--------|--------|----------|----------|
| Vanilla         | 2.3211 | 23.46% | -2.93%   | $18,000  |
| V1 2-state HMM  | 2.2922 | 23.29% | -6.19%   | $25,875  |
| V2 confidence   | 1.8695 | 23.23% | -6.58%   | $24,850  |
| V3 3-state+VIX  | 2.3898 | 23.21% | -6.03%   | $23,531  |
| V4 capped 1.2x  | 2.3490 | 23.44% | -3.50%   | $20,968  |
| V5 6-state BIC  | 2.3231 | 23.42% | -3.53%   | $19,936  |
| **V6 GARCH ✓**  | **2.5366** | **23.46%** | **-3.18%** | **$20,618** |

## Architecture

```
data_loader.py   → VFV.TO + VIX download, GARCH(1,1) vol, preprocessing
hmm_regime.py    → 3-state GaussianHMM, StandardScaler, regime plot
forecasting.py   → ARIMA/ETS/Prophet monthly competition (rolling RMSE)
strategy.py      → Confidence-scaled multiplier lookup table
backtest.py      → Walk-forward engine, XIRR/Sharpe/drawdown metrics
notebooks/
  smart_sip.ipynb → End-to-end narrative with plots and interpretation
```

## Key Design Decisions

**GARCH over rolling vol** — GARCH(1,1) conditional volatility captures
volatility clustering (large moves follow large moves) better than
equally-weighted rolling std. Produced cleaner HMM regime separation
and 97.45% mean state confidence.

**3 states over 2** — Adding a choppy/sideways state between calm and
volatile reduced over-investment during uncertain periods. BIC analysis
suggested 6 states optimal but 3 states produced better backtest results
and simpler allocation logic.

**Prophet over LSTM** — LSTM requires thousands of sequential observations
to learn meaningful patterns. Monthly return series (~156 observations)
is insufficient. Prophet's trend + seasonality + changepoint decomposition
is more appropriate and fully interpretable.

**Walk-forward validation** — Every prediction uses only data available
at that point in time. No look-ahead bias. Initial training window:
2012–2022. Walk-forward test: 2023–2025, one month at a time.

**Confidence scaling** — HMM outputs a probability per state, not just
a binary label. Multipliers scale toward neutral (1.0x) when confidence
is below 1.0, reducing over-commitment during uncertain regime calls.

## Honest Limitations

- Single asset (VFV.TO) over one 36-month test window
- 2023–2025 was predominantly bullish — results may not generalize
- Forecasting models predicted "up" 94.4% of months, reflecting trend
- RMSE used for model competition — directional accuracy more appropriate
- Transaction costs, MER (0.09%), and CAD/USD FX impact not modelled
- HMM retrains monthly — not optimized for real-time deployment

## Setup

```bash
git clone https://github.com/vdesai17/vfv_to_forecast
cd vfv_to_forecast
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Download and process data
python src/data_loader.py

# Detect regimes and plot
python src/hmm_regime.py

# Run backtest
python src/backtest.py

# Open notebook
jupyter notebook notebooks/smart_sip.ipynb
```

## Requirements

```
yfinance==1.2.0
pandas
numpy
matplotlib
statsmodels
hmmlearn
scikit-learn
arch
prophet
pyxirr
jupyter
```

## Stack

Python 3.9 · hmmlearn · arch · statsmodels · Prophet · scikit-learn · pyxirr

---
