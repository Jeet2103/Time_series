# Stock Market Time Series Forecasting

This project performs advanced time series forecasting on historical stock market data using **ARIMA**, **SARIMA**, **Prophet**, and **LSTM** models. It provides a clean, modular pipeline for preprocessing, modeling, evaluation, and visualization, with a fully interactive **Streamlit dashboard** for live forecasting and performance comparison.

---

## Overview

Forecasting stock prices is a vital challenge in the financial world. This project tackles it by leveraging both statistical and deep learning techniques to predict stock closing prices based on historical trends.

Key highlights:
- Cleans and preprocesses real-world Yahoo Finance stock data.
- Trains and evaluates four powerful models on time series data.
- Compares model performance using MAE, MSE, and RMSE.
- Visualizes forecasts and metrics in a beautiful, user-friendly Streamlit web app.

---

## Approach

The forecasting workflow follows a clean and consistent modular pipeline:

### 1. **Data Cleaning**
- Parses dates, removes commas from numerics, converts to float.
- Handles missing data and applies linear interpolation.
- Converts irregular stock trading days to a daily time series.

### 2. **Modeling**
- Uses four core forecasting models:
  - **ARIMA**: AutoRegressive Integrated Moving Average.
  - **SARIMA**: Seasonal ARIMA with weekly seasonality.
  - **Prophet**: Facebook’s time series model for trend/seasonality decomposition.
  - **LSTM**: Recurrent Neural Network for learning temporal patterns.

### 3. **Evaluation**
- All models are evaluated on the last 30 days of data.
- Metrics computed: **MAE**, **MSE**, **RMSE**.
- Forecasts and scores are saved for reproducibility.

### 4. **Visualization**
- Individual forecasts are plotted and saved.
- Forecast vs actual is displayed in the Streamlit app.
- Best model is automatically selected based on RMSE.

### 5. **Interactive Web App**
- Upload CSV directly in the app.
- View historical prices, model forecasts, and all metrics.
- Compare models and view best performer.

---

## Tech Stack

| Layer             | Technology              |
|------------------|-------------------------|
| Language          | Python 3.10+            |
| Modeling          | ARIMA, SARIMA, Prophet, LSTM |
| Data              | Pandas, Numpy           |
| Evaluation        | Scikit-learn            |
| Deep Learning     | Keras, TensorFlow       |
| Visualization     | Matplotlib, Streamlit   |
| Project Structure | Modular Python Scripts  |

---

## Codebase Structure

```
Time_Series_Forecasting/
├── data/
│ └── cleaned_data.csv # Preprocessed stock data
├── evaluation/
│ ├── evaluate_models.py # Runs all models and saves metrics/forecasts
│ └── plot_models.py # Plots forecast results from CSV
├── models/
│ ├── arima_model.py # ARIMA model script
│ ├── sarima_model.py # SARIMA model script
│ ├── prophet_model.py # Prophet model script
│ └── lstm_model.py # LSTM model script
├── outputs/
│ ├── forecasts/ # Forecast results (.csv)
│ └── plots/ # Forecast plots (.png)
├── utils/
│ └── helpers.py # Data splitting, evaluation metrics
└── streamlit_app.py # Streamlit dashboard interface
├── requirements.txt
└── README.md
```
---

## Setup & Installation

Follow these steps to set up the project environment from scratch using **Conda**:

### 1. Clone the Repository

```
git clone https://github.com/yourusername/stock-time-series-forecasting.git
cd stock-time-series-forecasting
```
### 2. Create a Virtual Environment with Conda
```
conda create -n stock-forecast python=3.10 -y
conda activate stock-forecast

```
### 3. Install Dependencies
```
pip install -r requirements.txt

```
### 4. Run Forecast Pipeline
Run the following to evaluate all models and generate plots:
```
python evaluation/evaluate_models.py
python evaluation/plot_models.py
```

### 5. Launch Streamlit App
```
streamlit run streamlit_app.py

```
### 6. Upload Your Stock CSV
Upload a Yahoo Finance style CSV (with `Date`, `Open`, `Close`, `Volume`, etc.) to view results.

## Example Forecast
The app will generate:

- Individual model forecasts

- `MAE`, `MSE`, `RMSE` scores

- Best model based on `RMSE`

---
## Full Project Report
For an in-depth explanation of the project including:

- Introduction & goals

- Step-by-step methodology

- Evaluation metrics

- Forecast graphs

- Final conclusion and future work

Please see `report.md` for the complete project report.