import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
model = load_model('your_model_path.h5')

# Load the dataset
# Assuming you have a CSV file containing historical stock prices
data = pd.read_csv('your_dataset.csv')

# Function to prepare data for prediction
def prepare_data(df, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    x_test = []
    for i in range(look_back, len(df)):
        x_test.append(scaled_data[i-look_back:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, scaler

# Function to predict future stock prices
def predict_stock_price(model, data, scaler, days=30):
    predictions = []
    last_sequence = data[-60:]
    for _ in range(days):
        prediction = model.predict(np.array([last_sequence]))
        prediction = scaler.inverse_transform(prediction)[0][0]
        predictions.append(prediction)
        last_sequence = np.append(last_sequence[1:], prediction)
    return predictions

# Streamlit app
def main():
    st.title('Stock Price Predictor')

    # Sidebar for user input
    st.sidebar.title('Options')
    selected_stock = st.sidebar.selectbox('Select Stock Symbol', data['Symbol'].unique())
    prediction_days = st.sidebar.slider('Number of Days for Prediction', min_value=1, max_value=365, value=30)

    # Filter data for selected stock
    stock_data = data[data['Symbol'] == selected_stock].copy()

    # Prepare data for prediction
    x_test, scaler = prepare_data(stock_data)

    # Button to trigger prediction
    if st.button('Predict'):
        # Predict future stock prices
        predictions = predict_stock_price(model, x_test, scaler, prediction_days)
        
        # Display predictions
        st.write(f'Predicted Stock Prices for the next {prediction_days} days:')
        st.write(predictions)

if __name__ == '__main__':
    main()
