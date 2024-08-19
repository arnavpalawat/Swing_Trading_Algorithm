import datetime
import itertools

import yfinance as yf
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def fetchData(ticker):
    """Get the historical data from Yahoo Finance"""
    start = (datetime.datetime.today() - datetime.timedelta(days=120)).strftime('%Y-%m-%d')
    end = datetime.datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)
    print("Fetched raw data head:", data.head())
    return data

def preprocessData(data):
    """Preprocess the data to remove all but close and NaN values"""
    df = data[["Close"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('B')  # Assuming business days for stock data

    # Fill or drop remaining NaNs (if any exist after asfreq)
    df = df.fillna(method='ffill').fillna(method='bfill')  # Forward-fill or backward-fill

    print("Preprocessed data head:", df.head())
    return df

def splitData(data):
    """Split the data into training and test sets"""
    n = int(len(data) * 0.8)
    train = data[:n]
    test = data[n:]
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    return train, test

def ad_test(dataset):
    """Perform ADF test and return True if the series is stationary, otherwise False."""
    dftest = adfuller(dataset, autolag='AIC')
    return dftest[1] <= 0.05

def difference(data):
    """Plot differencing of the data"""
    plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(data)
    ax1.set_title('Original Series')
    ax1.axes.xaxis.set_visible(False)

    ax2.plot(data.diff())
    ax2.set_title('1st Order Differencing')
    ax2.axes.xaxis.set_visible(False)

    ax3.plot(data.diff().diff())
    ax3.set_title('2nd Order Differencing')
    plt.show()

def pacf_and_autocorrection(data):
    """Plot Partial Autocorrelation Function"""
    plot_pacf(data.dropna())
    plt.show()

def acf(data):
    """Plot Autocorrelation Function"""
    plot_acf(data.dropna())
    plt.show()

def optimize_ARIMA(train):
    """Optimize ARIMA parameters using AIC"""
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = float("inf")
    best_pdq = None

    for param in pdq:
        try:
            model = ARIMA(train, order=param)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
        except Exception as e:
            print(f"Error with parameters {param}: {e}")
            continue

    print(f"Best ARIMA order: {best_pdq}, AIC: {best_aic}")
    return best_pdq

def execute(ticker):
    # STEP 1 --> Get Data
    data = fetchData(ticker)

    # STEP 2 --> Preprocess Data
    df = preprocessData(data)

    # Check for NaN or inf values after preprocessing
    if df.isnull().values.any():
        print("Warning: NaN values found in the data after preprocessing.")
    if df.isin([float('inf'), float('-inf')]).values.any():
        print("Warning: Infinite values found in the data after preprocessing.")

    # STEP 2.5 --> ADF Test
    is_stationary = ad_test(df)
    print(f"Is the series stationary? {'Yes' if is_stationary else 'No'}")

    plt.plot(df)
    plt.title('Preprocessed Data')
    plt.show()

    # STEP 2.5 --> Optimize ARIMA parameters
    best_order = optimize_ARIMA(df)

    # STEP 3 --> Build and plot ARIMA model with optimized parameters
    model = ARIMA(df, order=best_order)
    model_fit = model.fit()
    print(model_fit.summary())

    # Forecasting for the next 14 days
    forecast_steps = 14
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create future date range
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')  # Business days

    # Prepare forecast DataFrame
    forecast_df = pd.DataFrame(data={'Forecast': forecast}, index=future_dates)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df, label='Historical Data')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
    plt.title('Forecast vs Historical Data')
    plt.legend()
    plt.show()

    print("Forecast for the next 14 days:")
    print(forecast_df)
