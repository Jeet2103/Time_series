import pandas as pd
import os

def preprocess_yahoo_data(input_path="data/yahoo_data.csv", output_path="data/cleaned_data.csv"):
    # Load the CSV file
    df = pd.read_csv(input_path)

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y")

    # Sort by date ascending
    df.sort_values('Date', inplace=True)

    # Remove commas and convert numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        df[col] = pd.to_numeric(df[col].str.replace(',', '', regex=False), errors='coerce')

    # Convert Volume to float (from Indian-format strings)
    df['Volume'] = pd.to_numeric(df['Volume'].str.replace(',', '', regex=False), errors='coerce')

    # Set Date as index and convert to daily frequency
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')

    # Interpolate missing 'Close' values linearly
    df['Close'] = df['Close'].interpolate(method='linear')

    # Drop rows where 'Close' is still missing (beginning or end of data)
    df.dropna(subset=['Close'], inplace=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned data
    df.to_csv(output_path)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    preprocess_yahoo_data()
