import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def get_monthly_returns(df):
    """
    Resample daily returns to monthly returns.
    Uses last Close price of each month to compute monthly return.
    """
    monthly = df["Close"].resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()
    return monthly_returns

def forecast_arima(series):
    """
    Fit ARIMA(1,1,0) on series and forecast one step ahead.
    Returns predicted next month return as float.
    """

    # Order (1,1,0) is a simple ARIMA that often works decently on financial returns.
    model = ARIMA(series, order=(1, 1, 0))
    result = model.fit()
    # Forecast one step ahead (next month)
    forecast = result.forecast(steps=1)
    # Return the predicted return as a float
    # iloc[0] gets the first (and only) value from the forecast Series
    return float(forecast.iloc[0])

def forecast_ets(series):
    """
    Fit Exponential Smoothing on series and forecast one step ahead.
    Returns predicted next month return as float.
    """

    # Parameters: additive trend, no seasonality, estimated initialization
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    )
    result = model.fit()
    # Forecast one step ahead (next month)
    forecast = result.forecast(steps=1)
    # Return the predicted return as a float
    return float(forecast.iloc[0])

def forecast_prophet(series):
    """
    Fit Facebook Prophet on series and forecast one step ahead.
    Returns predicted next month return as float.
    """

    # Prophet expects a DataFrame with 'ds' and 'y' columns
    df_prophet = pd.DataFrame({
        "ds": series.index,
        "y": series.values
    })

    # Parameters: tuned changepoint_prior_scale to 0.05 to avoid overfitting to crash spikes
    model = Prophet(
        changepoint_prior_scale=0.05,  # tuned: avoid overfitting to crash spikes
        yearly_seasonality=False,  # type: ignore[reportArgumentType]
        weekly_seasonality=False,  # type: ignore[reportArgumentType]
        daily_seasonality=False,  # type: ignore[reportArgumentType]
        seasonality_mode="additive"
    )
    model.fit(df_prophet)

    # Predict one month ahead (Prophet needs a future DataFrame with 'ds' column)
    future = model.make_future_dataframe(periods=1, freq="ME")
    # Forecast returns the full future DataFrame with predictions; we take the last row's 'yhat' as the next month prediction
    forecast = model.predict(future)
    # Return the predicted return as a float
    return float(forecast["yhat"].iloc[-1])

def compute_rmse(actual, predicted):
    """Compute RMSE between actual and predicted arrays."""
    return np.sqrt(mean_squared_error(actual, predicted))

def run_model_competition(series, validation_window=3):
    """
    Run rolling 3-month validation to pick best model.
    
    For each of the last `validation_window` months:
      - Train each model on data before that month
      - Predict that month
      - Score prediction vs actual
    
    Winner = lowest total RMSE across validation window.
    Returns: (winner_name, predicted_direction, rmse_dict)
    """
    # If series is too short for validation, default to ARIMA prediction on full series
    if len(series) < validation_window + 6:
        # Not enough data — default to ARIMA
        pred = forecast_arima(series)
        return "ARIMA", "up" if pred > 0 else "down", {}

    # Rolling validation
    # We will predict the last `validation_window` months one by one, each time training on all data before that month.
    actuals = []
    arima_preds, ets_preds, prophet_preds = [], [], []

    # Loop over the last `validation_window` months in reverse order (most recent month first)
    # Train on all data before the month we are trying to predict then predict that month, then slide ahead one month and repeat until we've predicted the last `validation_window` months
    # Sliding window: if validation_window=3, we predict months -3, -2, -1 in that order. For month -3, we train on all data before month -3. For month -2, we train on all data before month -2 (which includes month -3's actual value), etc.

    for i in range(validation_window, 0, -1):
        train = series.iloc[:-i]
        actual = series.iloc[-i]
        actuals.append(actual)

        try:
            arima_preds.append(forecast_arima(train))
        except:
            arima_preds.append(0.0)

        try:
            ets_preds.append(forecast_ets(train))
        except:
            ets_preds.append(0.0)

        try:
            prophet_preds.append(forecast_prophet(train))
        except:
            prophet_preds.append(0.0)

    rmse = {
        "ARIMA":   compute_rmse(actuals, arima_preds),
        "ETS":     compute_rmse(actuals, ets_preds),
        "Prophet": compute_rmse(actuals, prophet_preds),
    }

    winner = min(rmse, key=lambda name: rmse[name])

    # Winner forecasts next month on full series
    if winner == "ARIMA":
        final_pred = forecast_arima(series)
    elif winner == "ETS":
        final_pred = forecast_ets(series)
    else:
        final_pred = forecast_prophet(series)

    direction = "up" if final_pred > 0 else "down"

    return winner, direction, rmse


if __name__ == "__main__":
    # Quick sanity check
    import sys
    sys.path.append(".")
    from hmm_regime import load_data

    df = load_data()
    monthly_returns = get_monthly_returns(df)

    print(f"Monthly returns shape: {monthly_returns.shape}")
    print(f"Date range: {monthly_returns.index[0].date()} → {monthly_returns.index[-1].date()}")
    print(f"\nSample (last 5 months):\n{monthly_returns.tail()}")

    # Test competition on first 48 months of data
    test_series = monthly_returns.iloc[:48]
    winner, direction, rmse = run_model_competition(test_series)

    print(f"\nModel Competition Result:")
    print(f"  Winner:    {winner}")
    print(f"  Direction: {direction}")
    print(f"  RMSE scores: {rmse}")