import pandas as pd
from prophet import Prophet
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import evaluate_forecast

def run_prophet(data):
    """
    Run Prophet model on stock Close prices.
    Assumes input is a pandas Series or DataFrame with datetime index and a 'Close' column.
    """
    # Prepare DataFrame with required Prophet format
    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize Prophet
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )

    # Fit model
    model.fit(df)

    # Make future predictions
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Extract predicted values for last 30 days
    forecast_result = forecast.set_index('ds')['yhat']
    actual_series = df.set_index('ds')['y']

    # Align overlapping dates
    overlap_index = actual_series.index.intersection(forecast_result.index)
    if len(overlap_index) == 0:
        raise ValueError("No overlapping dates between actual and forecasted values.")

    actual = actual_series.loc[overlap_index]
    predicted = forecast_result.loc[overlap_index]

    # Evaluate
    metrics = evaluate_forecast(actual, predicted, model="Prophet")
    return predicted, metrics
