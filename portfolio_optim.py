import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data as wb
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Title and sidebar
st.title('Stock Prediction and Portfolio Optimization App')

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
