import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_test_split(data, test_size=0.2):
    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]


def evaluate_forecast(true, pred, model="Model"):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)

    print(f"{model} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    return {
        "Model": model,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

