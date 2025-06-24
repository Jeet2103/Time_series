import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_forecast(actual, predicted, title, filename):
    """Plot actual vs predicted values and save the figure."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='Actual', color='blue', linewidth=2)
    plt.plot(predicted.index, predicted.values, label='Forecast', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join("outputs", "plots", filename)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Plot saved: {output_path}")

def main():
    # Ensure output folders exist
    os.makedirs("outputs/plots", exist_ok=True)

    # Load actual cleaned data
    df = pd.read_csv("data/cleaned_data.csv", parse_dates=["Date"], index_col="Date")
    close_prices = df["Close"].interpolate()
    actual = close_prices[-30:]

    # List of models and their forecast files
    models = ["ARIMA", "SARIMA", "Prophet", "LSTM"]

    for name in models:
        forecast_path = f"outputs/forecasts/{name.lower()}_forecast.csv"
        try:
            # Load forecast data and convert to Series
            forecast_df = pd.read_csv(forecast_path, index_col=0)
            forecast_df.index = pd.to_datetime(forecast_df.index, errors='coerce')
            forecast_series = forecast_df.squeeze()

            # Take only the last 30 values and align with actual
            forecast_series = forecast_series[-30:]
            forecast_series.index = actual.index

            # Plot and save
            plot_forecast(actual, forecast_series, f"{name} Forecast vs Actual", f"{name.lower()}_forecast.png")

        except Exception as e:
            print(f"❌ Failed to plot {name}: {e}")

def generate_plots_from_forecasts():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    def plot_forecast(actual, predicted, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual.values, label='Actual', color='blue', linewidth=2)
        plt.plot(predicted.index, predicted.values, label='Forecast', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_path = os.path.join("outputs", "plots", filename)
        plt.savefig(output_path)
        plt.close()

    df = pd.read_csv("data/cleaned_data.csv", parse_dates=["Date"], index_col="Date")
    actual = df["Close"].interpolate()[-30:]

    models = ["ARIMA", "SARIMA", "Prophet", "LSTM"]
    for name in models:
        forecast_path = f"outputs/forecasts/{name.lower()}_forecast.csv"
        try:
            forecast_df = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
            forecast_series = forecast_df.squeeze()[-30:]
            forecast_series.index = actual.index
            plot_forecast(actual, forecast_series, f"{name} Forecast vs Actual", f"{name.lower()}_forecast.png")
        except Exception as e:
            print(f"Failed to plot {name}: {e}")


if __name__ == "__main__":
    main()
