import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import pmdarima as pm
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import base64
import numpy as np
import datetime as dt
from pandas_datareader import data as wb
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.graph_objects as go



# Function to get or create session state
def get_session_state():
    return st.session_state

# Function to scrape data from the web
@st.cache_data()
def webscrap():
    url = 'https://www.moneycontrol.com/news/photos/business/stocks/'
    agent = {"User-Agent": "Mozilla/5.0"}
    page = requests.get(url, headers=agent)
    soup = BeautifulSoup(page.text, 'html.parser')
    section = soup.find_all('a', attrs={"title": re.compile("^Buzzing Stocks")})[0]
    link = section["href"]
    url2 = link
    page = requests.get(url2, headers=agent)
    soup2 = BeautifulSoup(page.text, 'html.parser')
    l1 = soup2.find_all('strong')
    l2 = [data.contents[0].strip(":").replace('&', '') for data in l1]
    return l2

# Function to get ticker symbol
@st.cache_data()
def get_ticker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "India"}
    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    if 'quotes' in data and data['quotes']:
        company_code = data['quotes'][0]['symbol']
        return company_code
    else:
        return ''

# Function to get name and ticker
@st.cache_data()
def get_name_ticker():
    l2 = webscrap()
    l3 = {}
    for name in l2:
        ticker_symbol = get_ticker(name)
        if ticker_symbol != '' and enough_historical_data(ticker_symbol):
            l3[name] = ticker_symbol
    return l3

@st.cache_data()
def enough_historical_data(ticker_symbol):
    min_data_points_threshold = 50

    try:
        # Download historical data for the specified stock
        historical_data = yf.download(ticker_symbol, start="2010-01-01", end="2023-12-31", progress=False)
        return len(historical_data) >= min_data_points_threshold
    except:
        # Handle any exceptions, such as the stock not being found or insufficient data
        return False


selected = option_menu(
    menu_title="Watchlist.io",
    options=["Home", "Downloads", "Portfolio Optimization"],
    orientation="horizontal",
    default_index=0
)

# Function for Exploratory Data Analysis
def perform_eda(data):
    st.header("Exploratory Data Analysis (EDA)")

    # 1. Overview of Historical Stock Price Data
    st.subheader("Overview of Historical Stock Price Data")
    st.write(data.head())
    st.write(data.describe())

    # 2. Time Series Visualization
    st.subheader("Time Series Visualization")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data['Date'], data['Close'], label="Close Price")
    ax.set(title="Time Series Analysis", xlabel="Date", ylabel="Close Price")
    ax.legend()
    st.pyplot(fig)

    # 3. Distribution of Stock Metrics
    st.subheader("Distribution of Stock Metrics")
    selected_metric = st.radio("Select a metric to visualize", ['Close', 'Volume'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[selected_metric], kde=True, bins=30, color='skyblue', ax=ax)
    ax.set(title=f"Distribution of {selected_metric}", xlabel=selected_metric, ylabel="Frequency")
    st.pyplot(fig)

# Main function
if selected == "Home":
    # Initialize session state
    session_state = get_session_state()
    st.header('Trending Stocks of the Day :money_with_wings:', divider='green')
    
    l3 = get_name_ticker()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    n = len(l3)
    
    c = 0
    for key in l3:
        if c % 3 == 0:
            with col1:
                if st.button(key):
                    # Store the selected stock in session state
                    session_state.selected_stock = l3[key]
                    # Navigate to another page (you can replace this with your Bollinger Bands page logic)
                    st.experimental_rerun()
            c += 1
        elif c % 3 == 1:
            with col2:
                if st.button(key):
                    session_state.selected_stock = l3[key]
                    st.experimental_rerun()
            c += 1
        else:
            with col3:
                if st.button(key):
                    session_state.selected_stock = l3[key]
                    st.experimental_rerun()
            c += 1
    if hasattr(session_state, "selected_stock"):
        t_symbol = session_state.selected_stock

        START = "2015-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365

        @st.cache_data
        def load_data(ticker):
            data = yf.download(t_symbol, START, TODAY)
            data.reset_index(inplace=True)
            return data
        data_load_state = st.text('Loading data...')
        data = load_data(t_symbol)
        data_load_state.text('Loading data... done!')

        #perform_eda(data)

        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())
            
        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)
       


# Downloads section
if selected == "Downloads":
    st.title(f"You have selected {selected}")
    l3 = get_name_ticker()
    # Downloadable CSV button
    st.markdown("""
    ### Download Stock Ticker Data
    Click the button below to download the stock ticker data as a CSV file.
    """)
    download_button = st.button("Download CSV")

    # Check if the button is clicked
    if download_button:
        # Create a DataFrame from the dictionary
        df_ticker = pd.DataFrame(list(l3.items()), columns=['Company', 'Ticker Symbol'])

        # Save DataFrame to CSV
        csv_file = df_ticker.to_csv(index=False)

        # Create a download link
        b64 = base64.b64encode(csv_file.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="stock_tickers.csv">Download CSV File</a>', unsafe_allow_html=True)

if selected == "Portfolio Optimization":
        # Title and sidebar
    st.title('Stock Portfolio Optimization')

    # User input for stock symbols
    st.sidebar.header('Enter Stock Symbols')
    stocks = st.sidebar.text_input('Enter stock symbols separated by space (e.g., TSLA AAPL):', value= "SBIN AMBUJACEM  ONGC WIPRO HEROMOTOCO LT ITC COALINDIA RELIANCE NCC").split()
    asset = [stock.upper() + '.NS' for stock in stocks]

    # User input for number of years
    years = st.sidebar.number_input('Number of years data to analyze:', min_value=1, max_value=10, step=1, value=1)
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365 * years)

    # Importing Data
    st.write('## Stock Data')
    yf.pdr_override()
    pf_data = wb.get_data_yahoo(asset, start=startDate, end=endDate)['Adj Close']
    st.write(pf_data)

    # Prediction With LSTM
    def Scale_data_set(data):
        dataset = data.values
        training_data_len = int(np.ceil(len(dataset) * .95))
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        return training_data_len, scaled_data, dataset, scaler

    def Creating_training_data(training_data_len, scaled_data):
        train_data = scaled_data[0:int(training_data_len), :]
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train

    def LSTM_model(x_train, y_train):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        return model

    def miscellaneous(model, training_data_len, scaled_data, dataset, scaler):
        test_data = scaled_data[training_data_len - 60:, :]
        x_test = []
        y_test = dataset[training_data_len:, :]

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

        return predictions, rmse

    def plot_graph_plotly(predictions, training_data_len, data, title):
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        train_data = go.Scatter(
            name='Training data',
            x=data.index,
            y=train['Close'],
            marker=dict(color='blue',size=5,)
        )

        Val_data = go.Scatter(
            name='Actual Value',
            x=valid.index,
            y=valid['Close'],
            marker=dict(color='green',size=5,)
        )

        Prediction_data = go.Scatter(
            name='Predicted Data',
            x=valid.index,
            y=valid['Predictions'],
            marker=dict(color='red',size=5,)
        )

        data = [train_data, Val_data, Prediction_data]

        layout = go.Layout(
            title = title,
            yaxis = dict(title='Close Price in Rs'),
            xaxis = dict(title='Date'),
            showlegend = True,
            legend = dict(
                x = .83, y = 0, traceorder='normal',
                bgcolor='#E2E2E2',
                bordercolor='black',
                borderwidth=2),
            width=980,
            height=500)

        fig = go.Figure(data=data, layout=layout)
        return fig


    @st.cache_data
    def Stock_prediction(data, title):
        training_data_len, scaled_data, dataset, scaler = Scale_data_set(data)
        x_train, y_train = Creating_training_data(training_data_len, scaled_data)
        model = LSTM_model(x_train, y_train)
        predictions, rmse = miscellaneous(model, training_data_len, scaled_data, dataset, scaler)
        return training_data_len, rmse, predictions

    # Sending Data for Prediction
    data = {}
    for stock in asset:
        data['{}'.format(stock)] = pf_data.filter([stock])

    for i in asset:
        DATA = data[i]
        DATA = DATA.rename(columns = {i:"Close"})
        training_data_len, rmse, predictions = Stock_prediction(DATA, i)
        fig = plot_graph_plotly(predictions, training_data_len, DATA, i)
        st.plotly_chart(fig)  # Moved outside the Stock_prediction function
        st.write("'root mean squared error' of {} = ".format(i) + str(rmse))

    # Picking best performing stock Returns
    retrn = (pf_data / pf_data.shift(1)) - 1
    annual_returns = retrn.mean() * 250 *100
    annual_returns.values[::-1].sort()
    sorted_annual_return = annual_returns[0:5]
    assets = list(sorted_annual_return.index)
    pf_data = pf_data.filter(assets)

    # Calculate returns of the stocks
    returns = (pf_data / pf_data.shift(1)) - 1

    # Markovitz Portfolio Theory
    num_assets = len(assets)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns = []
    pfolio_volatilities = []
    pfolio_weights = []

    for x in range(5000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        pfolio_returns.append(np.dot(weights, returns.mean()) * 250)
        pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 250, weights))))
        pfolio_weights.append(weights)

    pfolio_returns = np.array(pfolio_returns)
    pfolio_volatilities = np.array(pfolio_volatilities)

    portfolios = pd.DataFrame({'Return': pfolio_returns, 'Volatility': pfolio_volatilities})


    st.write('## Portfolio Optimization')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolios['Volatility'], y=portfolios['Return'], mode='markers', name='Portfolios'))
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Expected Volatility',
        yaxis_title='Expected Return')
    st.plotly_chart(fig)

    max_return = max(pfolio_returns)
    max_rindex = np.where(pfolio_returns == max_return)
    max_rvolatility = pfolio_volatilities[max_rindex[0][0]]
    max_rweights = pfolio_weights[max_rindex[0][0]]
    min_volatility = pfolio_volatilities.min()
    min_vindex = np.where(pfolio_volatilities == min_volatility)
    min_vreturns = pfolio_returns[min_vindex[0][0]]
    min_vweights = pfolio_weights[min_vindex[0][0]]

    max_sr = []
    max_sr = [str(round(max_return * 100,3)) + '%', str(round(max_rvolatility * 100,3)) + '%']
    for i in range(num_assets):
        max_sr.append(str(round(max_rweights[i] * 100,3)) + '%')

    min_vol = []
    min_vol = [str(round(min_vreturns * 100,3)) + '%', str(round(min_volatility * 100,3)) + '%']
    for i in range(num_assets):
        min_vol.append(str(round(min_vweights[i] * 100,3)) + '%')

    col = []
    for i in range(num_assets):
        col.append(pf_data.columns[i])

    result_table = pd.DataFrame(columns=[col], index=['maximum Return', 'minimum risk'])
    result_table.iloc[0] = max_sr[:num_assets]
    result_table.iloc[1] = min_vol[:num_assets]


    st.write('### Portfolio Allocation Results')
    st.write(result_table)