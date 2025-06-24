import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import evaluate_forecast

def run_lstm(data, look_back=60):
    """
    Trains an LSTM model on the given data and forecasts the next 30 days.
    
    Args:
        data (pd.Series): Time series (e.g., Close prices).
        look_back (int): Sequence length for LSTM input.
    
    Returns:
        tuple: (forecast, metrics)
    """
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    # Split into train/test (train to N-30, predict 30 future days)
    X, y = [], []
    for i in range(look_back, len(scaled_data) - 30):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, time steps, features)

    # LSTM Model architecture
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    # Forecast next 30 days
    last_input = scaled_data[-look_back:]
    preds = []
    for _ in range(30):
        x_input = last_input.reshape((1, look_back, 1))
        yhat = model.predict(x_input, verbose=0)[0][0]
        preds.append(yhat)
        last_input = np.append(last_input[1:], yhat)

    # Inverse transform predictions
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    true_values = data[-30:].values

    # Evaluate
    metrics = evaluate_forecast(true_values, preds, model="LSTM")
    return preds, metrics
