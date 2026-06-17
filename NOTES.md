# Project Log

## June 16
- data_loader.py complete. 3,276 rows, 2012-12-07 to 2025-12-30.
- hmm_regime.py complete. Regimes confirmed: 2018, 2020, 2022, 2025 volatile.
- Note: volatile bands are narrow/spiky. Possible HMM sensitivity issue. 
  Document in limitations.
- forecasting.py complete. ARIMA/ETS/Prophet competition working.
- RMSE scores very close across all three models (expected).
- ARIMA won first test on 48 months of data, direction: up.

## Next
- Allocation lookup table, create src/strategy.py