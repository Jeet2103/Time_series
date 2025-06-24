import pandas as pd
import sys
import os

# Allow import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import train_test_split, evaluate_forecast

# Optional: Use pmdarima for better parameter selection
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


def run_arima(data, order=None):
    """
    Trains an ARIMA model and returns the forecast and evaluation metrics.

    Parameters:
        data (pd.Series): Time series data (e.g., Close prices).
        order (tuple or None): ARIMA order (p,d,q). If None, it will be auto-tuned.

    Returns:
        forecast (pd.Series): Predicted values for the test set.
        metrics (dict): Evaluation metrics (MAE, RMSE).
    """
    # Ensure datetime index with proper frequency
    if data.index.inferred_freq is None:
        data = data.asfreq('D')

    train, test = train_test_split(data)

    if order is None:
        print("Auto-selecting ARIMA parameters...")
        auto_model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
        order = auto_model.order
        print(f"Selected ARIMA order: {order}")

    # Fit model
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=len(test))
    forecast.index = test.index  # Align forecast index with actual test dates

    # Evaluate
    metrics = evaluate_forecast(test, forecast, model="ARIMA")
    return forecast, metrics
