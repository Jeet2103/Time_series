import pandas as pd
import sys
import os
import traceback

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model functions
from models.arima_model import run_arima
from models.sarima_model import run_sarima
from models.prophet_model import run_prophet
from models.lstm_model import run_lstm

def main():
    # Load data
    df = pd.read_csv("data/cleaned_data.csv", parse_dates=['Date'], index_col='Date')
    close_prices = df['Close'].interpolate()

    # Define models
    models = [
        ("ARIMA", run_arima),
        ("SARIMA", run_sarima),
        ("Prophet", run_prophet),
        ("LSTM", run_lstm)
    ]

    results = []
    os.makedirs("outputs/forecasts", exist_ok=True)

    for name, model_func in models:
        try:
            print(f"\nRunning {name} model...")
            forecast, metrics = model_func(close_prices)

            # Save forecast
            forecast_path = f"outputs/forecasts/{name.lower()}_forecast.csv"
            pd.Series(forecast).to_csv(forecast_path, header=['Forecast'])

            # Add results
            results.append(metrics)
            print(f"{name} completed.")
            print(f"    MAE:  {metrics['MAE']:.2f}")
            print(f"    MSE:  {metrics['MSE']:.2f}")
            print(f"    RMSE: {metrics['RMSE']:.2f}")

        except Exception as e:
            print(f"{name} failed due to: {e}")
            traceback.print_exc()

    # Save evaluation metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/forecasts/model_metrics.csv", index=False)
    print("\nEvaluation summary saved to outputs/forecasts/model_metrics.csv")

    # Print best model by RMSE
    if not results_df.empty:
        best_model = results_df.sort_values(by='RMSE').iloc[0]
        print(f"\nBest Model by RMSE: {best_model['Model']} → RMSE: {best_model['RMSE']:.2f}")

if __name__ == "__main__":
    main()
