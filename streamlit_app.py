import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import subprocess

# Add root directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import forecast plot generator
from evaluation.plot_models import generate_plots_from_forecasts

# --------------------- Settings ---------------------
st.set_page_config(page_title="📈 Stock Forecasting", layout="wide")
st.markdown("<h1 style='text-align: center;'>📊 Stock Price Forecasting Dashboard</h1>", unsafe_allow_html=True)

# --------------------- Helper Functions ---------------------
def clean_data(df):
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')
    df['Close'] = df['Close'].interpolate(method='linear')
    return df

# --------------------- Upload CSV ---------------------
st.sidebar.header("📁 Upload Data")
uploaded = st.sidebar.file_uploader("Upload Yahoo Finance CSV", type=["csv"])

if uploaded:
    raw_df = pd.read_csv(uploaded)
    df = clean_data(raw_df)
    close_prices = df['Close']

    # Save cleaned version
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cleaned_data.csv")

    st.success("✅ Data cleaned and ready for forecasting!")

    # --------------------- Historical Plot ---------------------
    st.subheader("📈 Historical Closing Price")
    st.line_chart(close_prices)

    # --------------------- Model Evaluation ---------------------
    with st.spinner("🔄 Running forecasting models..."):
        subprocess.run(["python", "evaluation/evaluate_models.py"], check=True)

    with st.spinner("🖼️ Generating plots..."):
        generate_plots_from_forecasts()

    st.success("✅ Models evaluated and plots generated.")

    # --------------------- Metrics Summary ---------------------
    st.subheader("📊 Model Performance Metrics")

    metrics_df = pd.read_csv("outputs/forecasts/model_metrics.csv")
    st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}"}), use_container_width=True)

    # --------------------- Metrics Cards ---------------------
    st.markdown("### 📌 Summary (Last 30 Days)")
    cols = st.columns(3)

    for i, metric in enumerate(['MAE', 'MSE', 'RMSE']):
        with cols[i]:
            for _, row in metrics_df.iterrows():
                st.metric(label=f"{row['Model']} {metric}", value=f"{row[metric]:.2f}")

    # --------------------- Forecast Plots ---------------------
    st.subheader("📉 Forecast Plots")

    models = ["ARIMA", "SARIMA", "Prophet", "LSTM"]
    for model in models:
        plot_path = f"outputs/plots/{model.lower()}_forecast.png"
        if os.path.exists(plot_path):
            with st.expander(f"🔮 {model} Forecast", expanded=False):
                st.image(plot_path, use_column_width=True)
        else:
            st.warning(f"{model} forecast plot not found.")

    # --------------------- Best Model ---------------------
    best = metrics_df.sort_values('RMSE').iloc[0]
    st.subheader("🏆 Best Model (Lowest RMSE)")
    st.success(f"**{best['Model']}** with RMSE = {best['RMSE']:.2f}")
