import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM

# Disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Download data
start = '2012-01-01'
end = '2024-1-10'
stocks = ['GOOG', 'TSLA', 'AAPL', 'CSCO', 'MSFT']
selected_stock = st.sidebar.selectbox('Select Stock', stocks)
data = yf.download(selected_stock, start, end)
data.reset_index(inplace=True)

# Prepare data
data.dropna(inplace=True)
data_train = data.Close[0: int(len(data)*0.80)]
data_test = data.Close[int(len(data)*0.80):]

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train.values.reshape(-1, 1))

x_train, y_train = [], []
for i in range(100, len(data_train_scaled)):
    x_train.append(data_train_scaled[i-100:i, 0])
    y_train.append(data_train_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define and train LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

# Save the trained LSTM model
model.save("lstm_model.h5")

# Load the saved model
loaded_model = load_model("lstm_model.h5")

# Streamlit app
st.title('Stock Forecast App')

# Sidebar for user input
st.sidebar.title('Options')
n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Plot moving averages
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

fig_ma_100, ax_ma_100 = plt.subplots(figsize=(10, 6))
ax_ma_100.plot(ma_100_days, 'r', label='MA 100 days')
ax_ma_100.plot(data.Close, 'g', label='Close')
ax_ma_100.set_xlabel('Date')
ax_ma_100.set_ylabel('Price')
ax_ma_100.set_title('Stock Prices Over Time (100-Day MA)')
ax_ma_100.legend()
st.pyplot(fig_ma_100)

fig_ma_200, ax_ma_200 = plt.subplots(figsize=(10, 6))
ax_ma_200.plot(ma_200_days, 'b', label='MA 200 days')
ax_ma_200.plot(data.Close, 'g', label='Close')
ax_ma_200.set_xlabel('Date')
ax_ma_200.set_ylabel('Price')
ax_ma_200.set_title('Stock Prices Over Time (200-Day MA)')
ax_ma_200.legend()
st.pyplot(fig_ma_200)

# Plot predicted values
st.subheader('Predicted Values')
fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
ax_pred.plot(data['Date'][-period:], data['Close'][-period:], label='Actual')
ax_pred.set_xlabel('Date')
ax_pred.set_ylabel('Price')
ax_pred.set_title('Predicted Stock Prices')
ax_pred.legend()
st.pyplot(fig_pred)

# Display contact information
st.sidebar.subheader('Contact Information')
st.sidebar.text('Developer: Your Name')