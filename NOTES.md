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
## June 17 (continued)
- HMM fixed with StandardScaler, features now balanced
- vol_ratio (21d/63d) added, momentum dropped (caused all-calm)
- Regime plot now correctly identifies 2015, 2018, 2020, 2022, 2025
- 2020 band still narrower than expected, document in limitations

