# Project Log

## June 16
- data_loader.py complete. 3,276 rows, 2012-12-07 to 2025-12-30.
- hmm_regime.py complete. Regimes confirmed: 2018, 2020, 2022, 2025 volatile.
- Note: volatile bands are narrow/spiky. Possible HMM sensitivity issue. 
  Document in limitations.
- forecasting.py complete. ARIMA/ETS/Prophet competition working.
- RMSE scores very close across all three models (expected).
- ARIMA won first test on 48 months of data, direction: up.

## June 17: Backtest Complete
- Smart SIP final value $35,984 vs Vanilla $24,896
- Smart SIP deployed $7,875 MORE capital (43% more)
- Vanilla wins on XIRR (23.46% vs 23.29%)
- Vanilla wins on Sharpe (2.32 vs 2.29)
- Vanilla worse on drawdown (-2.93% vs -6.19%) Smart SIP took bigger hits
- Conclusion: no meaningful alpha from dynamic allocation in 2023–2025 bull market
- Honest result: strategy did not outperform on risk-adjusted basis
## June 17: FIx
- HMM fixed with StandardScaler, features now balanced
- vol_ratio (21d/63d) added, momentum dropped (caused all-calm)
- Regime plot now correctly identifies 2015, 2018, 2020, 2022, 2025
- 2020 band still narrower than expected, document in limitations
## June 17 — V4 Locked (Final Version)

### System
- data_loader.py:  VFV.TO + VIX, 3,276 rows, 2012–2025
- hmm_regime.py:   3-state HMM (calm/choppy/volatile), StandardScaler,
                   features: daily_return, rolling_vol, vol_ratio, vix
- forecasting.py:  ARIMA, ETS, Prophet competition (rolling 3-month RMSE)
- strategy.py:     Confidence-scaled multipliers, capped at 1.2x
- backtest.py:     Walk-forward, train 2012–2022, test 2023–2025

### V4 Final Results
Sharpe:    2.3490  > vanilla 2.3211  ✓
XIRR:      23.44%  ≈ vanilla 23.46%  (gap = 0.02%)
Drawdown: -3.50%   vs vanilla -2.93%
Invested:  $20,968 vs vanilla $18,000

### Iteration History
Version          Sharpe    XIRR     Drawdown    Invested
─────────────────────────────────────────────────────────
Vanilla          2.3211    23.46%   -2.93%      $18,000
V1 2-state       2.2922    23.29%   -6.19%      $25,875
V2 confidence    1.8695    23.23%   -6.58%      $24,850
V3 3-state+VIX   2.3898    23.21%   -6.03%      $23,531
V4 capped 1.2x   2.3490    23.44%   -3.50%      $20,968  ← FINAL
V5 6-state BIC   2.3231    23.42%   -3.53%      $19,936

### Key Decisions
- Dropped LSTM → Prophet (speed + interpretability)
- Dropped momentum feature → caused all-calm HMM
- Added StandardScaler → fixed feature dominance issue
- Added VIX → stronger regime signal
- Added 3rd (choppy) state → buffer between calm and volatile
- Capped multiplier at 1.2x → reduced drawdown from -6% to -3.5%
- Stopped at V4 → V5 BIC/6-state confirmed V4 was optimal

### Limitations (for notebook)
- Single asset, 36-month test window
- 2023–2025 was predominantly bull market
- RMSE not ideal metric — directional accuracy better
- HMM detects 2020 COVID spike narrowly
- Drawdown still worse than vanilla (-3.5% vs -2.93%)
- Results not generalizable beyond this test period

## Next
- Build final comparison plot
- Write smart_sip.ipynb narrative
- Write README.md
