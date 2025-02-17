import streamlit as st
import pandas as pd
import requests
import time
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from dotenv import load_dotenv, find_dotenv
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import csv


from statsmodels.tsa.arima.model import ARIMA

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set page configuration
st.set_page_config(page_title="Stock Market Dashboard", layout="wide", page_icon="📈")

# Obtain Alpha Vantage API key from environment variable
API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')

if not API_KEY:
    st.error("Alpha Vantage API Key not found. Please set the 'ALPHA_VANTAGE_API_KEY' environment variable in your .env file.")
    st.stop()

# DuckDB connection
conn = duckdb.connect('data/stocks.duckdb')

# Ensure metadata table exists
def ensure_metadata_table():
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_metadata (
            symbol TEXT PRIMARY KEY,
            last_fetched TIMESTAMP
        )
    """)

def ensure_us_listings_table():
    conn.execute("""
        CREATE TABLE IF NOT EXISTS us_stock_listings (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            exchange TEXT,
            asset_type TEXT,
            ipo_date DATE,
            delisting_date DATE,
            status TEXT
        )
    """)


# Function to fetch and cache stock data
def fetch_stock_data(symbol, fetch_latest=False):
    table_name = symbol.replace('.', '_')
    ensure_metadata_table()
    if fetch_latest:
        st.session_state.fetch_latest = False
        return fetch_from_api(symbol)

    # Check if data is cached
    try:
        df = conn.execute(f"SELECT * FROM '{table_name}'").fetchdf()
    except:
        df = pd.DataFrame()
    # Get last_fetched date from metadata
    res = conn.execute("SELECT last_fetched FROM stock_metadata WHERE symbol = ?", [symbol]).fetchone()
    if res:
        last_fetched = res[0]
    else:
        last_fetched = None
    if not df.empty:
        message = st.empty()
        message.success(f"Loaded cached data for {symbol}.")
        time.sleep(1)
        message.empty()
    else:
        df, last_fetched = fetch_from_api(symbol)
    return df, last_fetched


def fetch_from_api(symbol):
    table_name = symbol.replace('.', '_')
    ensure_metadata_table()
    URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}"
    r = requests.get(URL)
    data = r.json()
    if 'Time Series (Daily)' in data:
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index', dtype=float)
        df.index = pd.to_datetime(df.index)
        df.rename(columns=lambda x: x.split(' ')[1].capitalize(), inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        conn.execute(f"CREATE OR REPLACE TABLE '{table_name}' AS SELECT * FROM df")
        # Update metadata
        last_fetched = datetime.now()
        conn.execute("INSERT OR REPLACE INTO stock_metadata (symbol, last_fetched) VALUES (?, ?)", [symbol, last_fetched])
        st.success(f"Data for {symbol} fetched and cached.")
    else:
        st.error(f"Error fetching data for {symbol}. Please check the symbol and API key.")
        df = pd.DataFrame()
        last_fetched = None
    return df, last_fetched

def fetch_and_store_us_listings():
    ensure_us_listings_table()
    # Check if listings are already in the database
    existing_listings = conn.execute("SELECT COUNT(*) FROM us_stock_listings").fetchone()[0]
    if existing_listings > 0:
        return
    # Fetch the CSV data
    CSV_URL = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={API_KEY}'
    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        data = [row for row in csv.reader(decoded_content.splitlines(), delimiter=',')]
        # Convert to DataFrame
        listings_df = pd.DataFrame(data[1:], columns=data[0])
    # Filter for active listings
    listings_df = listings_df[listings_df['status'] == 'Active'].copy()
    # Prepare DataFrame for insertion
    listings_df_db = listings_df[['symbol', 'name', 'exchange', 'assetType', 'ipoDate', 'delistingDate', 'status']].copy()
    # Rename columns to match database table
    listings_df_db.columns = ['symbol', 'name', 'exchange', 'asset_type', 'ipo_date', 'delisting_date', 'status']
    # Convert date columns to proper format
    listings_df_db['ipo_date'] = pd.to_datetime(listings_df_db['ipo_date'], errors='coerce')
    listings_df_db['delisting_date'] = pd.to_datetime(listings_df_db['delisting_date'], errors='coerce')
    # Insert data into the database
    conn.execute("DELETE FROM us_stock_listings")  # Clear existing data
    conn.execute("INSERT INTO us_stock_listings SELECT * FROM listings_df_db")
    st.success("Stock listings have been stored in the database.")

def search_us_stocks(query):
    query = f"%{query.lower()}%"
    results = conn.execute("""
        SELECT * FROM us_stock_listings
        WHERE LOWER(name) LIKE ? OR LOWER(symbol) LIKE ?
        ORDER BY symbol
        LIMIT 100
    """, [query, query]).fetchdf()
    return results

# Visualization functions
def plot_closing_price(df, symbol):
    fig = px.line(
        df,
        x='Date',
        y='Close',
        title=f'{symbol} Closing Prices',
        labels={'Close': 'Closing Price (USD)', 'Date': 'Date'},
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        title_xanchor='center',
        xaxis_title='Date',
        yaxis_title='Closing Price (USD)',
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_volume(df, symbol):
    fig = px.bar(
        df,
        x='Date',
        y='Volume',
        title=f'{symbol} Trading Volume',
        labels={'Volume': 'Volume', 'Date': 'Date'},
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        title_xanchor='center',
        xaxis_title='Date',
        yaxis_title='Volume',
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_candlestick(df, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick',
    )])
    fig.update_layout(
        title=f'{symbol} Candlestick Chart',
        template='plotly_white',
        title_x=0.5,
        title_xanchor='center',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_moving_average(df, symbol, window):
    df_ma = df.copy()
    df_ma[f'MA_{window}'] = df_ma['Close'].rolling(window=window).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ma['Date'], y=df_ma['Close'], mode='lines', name='Close Price'
    ))
    fig.add_trace(go.Scatter(
        x=df_ma['Date'], y=df_ma[f'MA_{window}'], mode='lines', name=f'{window}-Day MA'
    ))
    fig.update_layout(
        title=f'{symbol} {window}-Day Moving Average',
        template='plotly_white',
        title_x=0.5,
        title_xanchor='center',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_high_low_prices(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['High'], mode='lines', name='High Price'
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Low'], mode='lines', name='Low Price'
    ))
    fig.update_layout(
        title=f'{symbol} High and Low Prices',
        template='plotly_white',
        title_x=0.5,
        title_xanchor='center',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_daily_returns(df, symbol):
    df_returns = df.copy()
    df_returns['Daily Return'] = df_returns['Close'].pct_change() * 100  # Percentage
    fig = px.bar(
        df_returns,
        x='Date',
        y='Daily Return',
        title=f'{symbol} Daily Returns',
        labels={'Daily Return': 'Daily Return (%)', 'Date': 'Date'},
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        title_xanchor='center',
        xaxis_title='Date',
        yaxis_title='Daily Return (%)',
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_cumulative_returns(df, symbol):
    df_cum_returns = df.copy()
    df_cum_returns['Cumulative Return'] = (1 + df_cum_returns['Close'].pct_change()).cumprod() - 1
    df_cum_returns['Cumulative Return'] *= 100  # Percentage
    fig = px.line(
        df_cum_returns,
        x='Date',
        y='Cumulative Return',
        title=f'{symbol} Cumulative Returns',
        labels={'Cumulative Return': 'Cumulative Return (%)', 'Date': 'Date'},
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        title_xanchor='center',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_bollinger_bands(df, symbol, window=20):
    df_bollinger = df.copy()
    df_bollinger['MA'] = df_bollinger['Close'].rolling(window=window).mean()
    df_bollinger['STD'] = df_bollinger['Close'].rolling(window=window).std()
    df_bollinger['Upper Band'] = df_bollinger['MA'] + (df_bollinger['STD'] * 2)
    df_bollinger['Lower Band'] = df_bollinger['MA'] - (df_bollinger['STD'] * 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_bollinger['Date'], y=df_bollinger['Close'], name='Close Price',
        mode='lines',
    ))
    fig.add_trace(go.Scatter(
        x=df_bollinger['Date'], y=df_bollinger['Upper Band'], name='Upper Band',
        line=dict(color='rgba(255, 0, 0, 0.5)'), mode='lines',
    ))
    fig.add_trace(go.Scatter(
        x=df_bollinger['Date'], y=df_bollinger['Lower Band'], name='Lower Band',
        line=dict(color='rgba(0, 0, 255, 0.5)'), mode='lines',
    ))
    fig.add_trace(go.Scatter(
        x=df_bollinger['Date'], y=df_bollinger['MA'], name=f'{window}-Day MA',
        line=dict(color='green'), mode='lines',
    ))
    fig.update_layout(
        title=f'{symbol} Bollinger Bands',
        template='plotly_white',
        title_x=0.5,
        title_xanchor='center',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)
def train_model(df):
    df_ml = df[['Date', 'Close']].copy()
    df_ml.sort_values('Date', inplace=True)
    df_ml.reset_index(drop=True, inplace=True)

    # Prepare data for forecasting
    df_ml['Date'] = pd.to_datetime(df_ml['Date'])
    df_ml.set_index('Date', inplace=True)
    df_ml = df_ml.asfreq('B')  # 'B' stands for business day frequency

    # Fill any missing values (if any)
    df_ml['Close'].fillna(method='ffill', inplace=True)

    # Split data into training and test sets
    split_date = df_ml.index[-30]  # Last 30 days for testing
    train = df_ml[:split_date]
    test = df_ml[split_date:]

    # Prepare features
    train['Time'] = np.arange(len(train))
    test['Time'] = np.arange(len(train), len(train) + len(test))

    X_train = train[['Time']]
    y_train = train['Close']
    X_test = test[['Time']]
    y_test = test['Close']

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred_test = model.predict(X_test)

    # Calculate error metrics
    rmse = root_mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # Print model coefficients
    print("\nModel Coefficients:")
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficient: {model.coef_[0]}")

    # Print error metrics
    print("\nError Metrics on Test Set:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Print actual vs predicted values on test set
    print("\nActual vs. Predicted Close Prices on Test Set:")
    for actual, predicted in zip(y_test, y_pred_test):
        print(f"Actual: ${actual:.2f}, Predicted: ${predicted:.2f}, Actual: ${actual - predicted:.2f}")

    # Predict the next 30 business days
    future_time_index = np.arange(len(train) + len(test), len(train) + len(test) + 30)
    future_dates = pd.date_range(start=df_ml.index[-1] + pd.Timedelta(days=1), periods=45, freq='B')[:30]
    X_future = pd.DataFrame({'Time': future_time_index})
    y_future_pred = model.predict(X_future)

    # Print future predictions
    print("\nFuture Predictions for the Next 30 Business Days:")
    for date, prediction in zip(future_dates, y_future_pred):
        print(f"Date: {date.strftime('%Y-%m-%d')}, Predicted Close: ${prediction:.2f}")

    future_predictions = pd.DataFrame({'Date': future_dates, 'Predicted_Close': y_future_pred})
    future_predictions.reset_index(drop=True, inplace=True)

    return model, y_pred_test, y_test, future_predictions, rmse, mae


def train_lstm_model(df):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.utils import shuffle
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    import streamlit as st

    # Preprocess Data
    df_lstm = df.copy()
    df_lstm['Date'] = pd.to_datetime(df_lstm['Date'], unit='ms')
    df_lstm.set_index('Date', inplace=True)
    df_lstm.sort_index(inplace=True)

    # Use only the 'Close' price for prediction
    close_prices = df_lstm[['Close']]

    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_prices)

    # Create sliding window sequences
    def create_sequences(data, window_size):
        sequences = []
        targets = []
        for i in range(len(data) - window_size):
            seq = data[i:i + window_size]
            target = data[i + window_size]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    # Define window size (e.g., 10 days)
    window_size = 50

    # Create sequences and targets
    X, y = create_sequences(scaled_close, window_size)

    # Split into training and testing sets (e.g., 80% train, 20% test)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape input data to [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=0  # Suppress output in Streamlit
    )

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)
    y_pred_test = y_pred_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Inverse transform predictions and actual values
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    y_test_inv = scaler.inverse_transform(y_test)

    # Calculate RMSE and MAE
    rmse = root_mean_squared_error(y_test_inv, y_pred_test_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_test_inv)

    # Prepare y_pred_test and y_test as pandas Series with dates for plotting
    test_dates = df_lstm.index[split + window_size:]
    y_test_series = pd.Series(y_test_inv.flatten(), index=test_dates)
    y_pred_series = pd.Series(y_pred_test_inv.flatten(), index=test_dates)

    # Define the prediction function
    def predict_future(model, recent_data, future_days, scaler):
        predictions = []
        input_seq = recent_data[-window_size:].reshape(1, window_size, 1)
        for _ in range(future_days):
            pred = model.predict(input_seq)
            pred_value = pred[0, 0]
            predictions.append(pred_value)
            new_step = np.array([[[pred_value]]])
            input_seq = np.concatenate((input_seq[:, 1:, :], new_step), axis=1)
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions

    # Number of days to predict into the future
    future_days = 10

    # Get future predictions
    future_predictions = predict_future(model, scaled_close, future_days, scaler)

    # Create future dates
    last_date = df_lstm.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

    # Create a DataFrame for future predictions
    future_predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_predictions.flatten()
    })

    return model, y_pred_series, y_test_series, future_predictions_df, rmse, mae


# Main application function
def main():
    # Custom CSS styling
    st.markdown("""
        <style>
            /* Reduce top padding */
            .main .block-container {
                padding-top: 0rem;
            }
            /* Custom CSS for the top bar heading */
            h1 {
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-bottom: 10px;
            }
            /* Adjust radio button styling */
            .stRadio > div {
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-bottom: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

    # Title
    st.title("Stock Market Analysis Dashboard")

    fetch_and_store_us_listings()

    st.sidebar.header("User Input")


    st.sidebar.subheader("Stock Selection")
    search_query = st.sidebar.text_input("Search for a stock", "")
    if search_query:
        search_results = search_us_stocks(search_query)
        if not search_results.empty:
            symbol_option = st.sidebar.selectbox("Select Stock Symbol", options=search_results['symbol'] + ' - ' + search_results['name'])
            symbol = symbol_option.split(' - ')[0]  # Extract symbol
        else:
            st.sidebar.warning("No matching stocks found.")
            return
    else:
        stock_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA']
        symbol = st.sidebar.selectbox("Select Stock Symbol", options=stock_list)

    # Fetch Latest Data Button
    st.sidebar.subheader("Data Handling")
    fetch_latest = st.sidebar.button("Fetch Latest Data")

    # Date range selection
    min_date = datetime(2022, 1, 1)
    max_date = datetime.today()
    start_date = st.sidebar.date_input("Start Date", value=datetime(2022, 1, 1), min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.sidebar.error("Error: End date must fall after start date.")

    # Visualizations selection
    st.sidebar.subheader("Select Visualizations")
    show_all = st.sidebar.checkbox("Show All Visualizations", value=False)
    show_closing_price = st.sidebar.checkbox("Closing Price Over Time", value=True)
    show_volume = st.sidebar.checkbox("Trading Volume Over Time", value=True)
    show_candlestick = st.sidebar.checkbox("Candlestick Chart", value=show_all)
    show_moving_average_20 = st.sidebar.checkbox("20-Day Moving Average", value=show_all)
    show_moving_average_50 = st.sidebar.checkbox("50-Day Moving Average", value=show_all)
    show_high_low_prices = st.sidebar.checkbox("High and Low Prices", value=show_all)
    show_daily_returns = st.sidebar.checkbox("Daily Returns", value=show_all)
    show_cumulative_returns = st.sidebar.checkbox("Cumulative Returns", value=show_all)
    show_bollinger_bands = st.sidebar.checkbox("Bollinger Bands", value=show_all)
    show_ml_prediction = st.sidebar.checkbox("Machine Learning Prediction", value=show_all)
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)

    # Fetch data
    if 'fetch_latest' not in st.session_state:
        st.session_state.fetch_latest = False
    if fetch_latest:
        st.session_state.fetch_latest = True
    df, last_fetched = fetch_stock_data(symbol, fetch_latest=st.session_state.fetch_latest)

    if not df.empty:
        # Filter data by date range
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
        df = df.sort_values(by='Date')  # Ensure data is sorted by date

        if df.empty:
            st.warning("No data available for the selected date range.")
            return

        # Display headers under stock name
        latest_data = df.iloc[-1]

        colhdr, coldate = st.columns([2, 1])
        with colhdr:
            st.header(f"{symbol}")
        with coldate:
            if last_fetched:
                st.markdown(f"**Last Data Fetched on:** {last_fetched.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.markdown("**Last Data Fetched on:** N/A")

        # Use columns to display metrics
        col2, col3, col4, col5, col6 = st.columns(5)

        with col2:
            st.metric("Current Price", f"${latest_data['Close']:.2f}")
        with col3:
            st.metric("Open", f"${latest_data['Open']:.2f}")
        with col4:
            st.metric("Low", f"${latest_data['Low']:.2f}")
        with col5:
            st.metric("High", f"${latest_data['High']:.2f}")
        with col6:
            st.metric("Volume", f"{int(latest_data['Volume']):,}")

        # Visualizations
        st.subheader(f"{symbol} Stock Data Visualizations ({start_date} to {end_date})")

        if show_all:
            show_closing_price = True
            show_volume = True
            show_candlestick = True
            show_moving_average_20 = True
            show_moving_average_50 = True
            show_high_low_prices = True
            show_daily_returns = True
            show_cumulative_returns = True
            show_bollinger_bands = True
            show_ml_prediction = True
            

        if show_closing_price:
            plot_closing_price(df, symbol)
            st.markdown("**Closing Price Over Time:** This line chart displays the closing price of the stock over the selected date range.")
        if show_volume:
            plot_volume(df, symbol)
            st.markdown("**Trading Volume Over Time:** This bar chart shows the volume of shares traded each day over the selected date range.")
        if show_candlestick:
            plot_candlestick(df, symbol)
            st.markdown("**Candlestick Chart:** This chart provides a visual representation of the stock's open, high, low, and close prices for each day.")
        if show_moving_average_20:
            plot_moving_average(df, symbol, window=20)
            st.markdown("**20-Day Moving Average:** This plot displays the stock's closing price along with the 20-day moving average, helping to identify trends.")
        if show_moving_average_50:
            plot_moving_average(df, symbol, window=50)
            st.markdown("**50-Day Moving Average:** This plot shows the closing price and the 50-day moving average, providing insights into longer-term trends.")
        if show_high_low_prices:
            plot_high_low_prices(df, symbol)
            st.markdown("**High and Low Prices:** This chart illustrates the highest and lowest prices of the stock for each day over the selected period.")
        if show_daily_returns:
            plot_daily_returns(df, symbol)
            st.markdown("**Daily Returns:** This bar chart represents the daily percentage change in the stock's closing price, indicating volatility.")
        if show_cumulative_returns:
            plot_cumulative_returns(df, symbol)
            st.markdown("**Cumulative Returns:** This line chart shows the cumulative return of the stock over the selected date range, highlighting overall performance.")
        if show_bollinger_bands:
            plot_bollinger_bands(df, symbol, window=20)
            st.markdown("**Bollinger Bands:** This chart displays the stock's price along with Bollinger Bands, which are volatility indicators based on standard deviation.")

      # Machine Learning Prediction
        if show_ml_prediction:
            st.header("Machine Learning Prediction")

            # Tabs for models
            model_option = st.radio("Select Model", ('Linear Regression', 'LSTM'))

            if model_option == 'Linear Regression':
                st.markdown("**Using Linear Regression Model**")
                # Assuming train_model is already defined and returns the necessary outputs
                model, y_pred_test, y_test, future_predictions, rmse, mae = train_model(df)
            elif model_option == 'LSTM':
                st.markdown("**Using LSTM Model**")
                model, y_pred_test, y_test, future_predictions, rmse, mae = train_lstm_model(df)

            st.markdown(f"**Test Set Root Mean Squared Error (RMSE):** {rmse:.2f}")
            st.markdown(f"**Test Set Mean Absolute Error (MAE):** {mae:.2f}")

            # Plot the actual vs predicted on test set
            test_dates = y_test.index
            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(
                x=test_dates, y=y_test,
                mode='lines', name='Actual Close (Test Set)'
            ))
            fig_test.add_trace(go.Scatter(
                x=test_dates, y=y_pred_test,
                mode='lines', name='Predicted Close (Test Set)'
            ))
            fig_test.update_layout(
                title='Actual vs Predicted Close Prices on Test Set',
                template='plotly_white',
                title_x=0.5,
                title_xanchor='center',
                xaxis_title='Date',
                yaxis_title='Closing Price (USD)',
                hovermode='x unified',
            )
            st.plotly_chart(fig_test, use_container_width=True)

            # Plot future predictions
            fig_future = go.Figure()
            # Plot last 60 days of actual close prices for context
            df_ml = df[['Date', 'Close']].copy()
            df_ml['Date'] = pd.to_datetime(df_ml['Date'], unit='ms')
            df_ml.set_index('Date', inplace=True)
            recent_actual = df_ml['Close'][-30:]
            fig_future.add_trace(go.Scatter(
                x=recent_actual.index, y=recent_actual,
                mode='lines', name='Actual Close'
            ))
            fig_future.add_trace(go.Scatter(
                x=future_predictions['Date'], y=future_predictions['Predicted_Close'],
                mode='lines', name='Predicted Close (Next Month)'
            ))
            fig_future.update_layout(
                title='Future Predicted Close Prices for the Next Month',
                template='plotly_white',
                title_x=0.5,
                title_xanchor='center',
                xaxis_title='Date',
                yaxis_title='Closing Price (USD)',
                hovermode='x unified',
            )
            st.plotly_chart(fig_future, use_container_width=True)

            # Display future predictions
            st.subheader("Predicted Prices for the Next Month")
            st.write(future_predictions)

        # Show raw data
        if show_raw_data:
            st.subheader('Raw Data')
            st.write(df)
    else:
        st.error("No data available.")

if __name__ == "__main__":
    main()