import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import sys
import os

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import train_test_split, evaluate_forecast

def run_sarima(data, order=None, seasonal_order=None):
    """
    Trains a SARIMA model on the input data and forecasts the next 30 days.

    Parameters:
        data (pd.Series): Time series data with datetime index.
        order (tuple or None): (p,d,q) - non-seasonal ARIMA order. If None, it will be auto-selected.
        seasonal_order (tuple or None): (P,D,Q,s) - seasonal order. If None, auto-selected via auto_arima.

    Returns:
        tuple: (forecast, metrics)
    """
    # Ensure proper frequency for SARIMAX
    if data.index.inferred_freq is None:
        data = data.asfreq('D')

    train, test = train_test_split(data)

    # Auto-tune if no manual order provided
    if order is None or seasonal_order is None:
        print("Auto-selecting SARIMA parameters...")
        model_auto = auto_arima(
            train,
            seasonal=True,
            m=7,  # Weekly seasonality
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        order = model_auto.order
        seasonal_order = model_auto.seasonal_order
        print(f"Selected SARIMA order: {order} seasonal_order: {seasonal_order}")

    # Fit SARIMA model
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # Forecast
    forecast = model_fit.forecast(steps=len(test))
    forecast.index = test.index  # Align forecast with test period

    # Evaluate
    metrics = evaluate_forecast(test, forecast, model="SARIMA")
    return forecast, metrics
