import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import streamlit as st

st.title('Cryptoprice Predictor')

# Input for cryptocurrency symbol
use_input = st.text_input('Enter a currency symbol (e.g., BTC-USD) to predict:')

# Slider to select the number of years for data display
n_years = st.slider('Select the number of years for data display:', 1, 4)

if st.button('Predict'):
    # Download data
    df = yf.download(use_input, period=f'{n_years}y')

    # Check if data is empty
    if df.empty:
        st.error("No data found for the entered currency symbol. Please try again.")
    else:
        # Display data description
        st.subheader(f'Data for the last {n_years} years')
        st.write(df.describe())

        # Plot Closing Price vs Time
        st.subheader('Closing Price vs Time Chart')
        fig = plt.figure(figsize=(10, 5))
        plt.plot(df.Close, color='yellow', label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        # Plot Closing Price with 100-day Moving Average
        st.subheader('Closing Price vs Time Chart with 100-day Moving Average')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize=(10, 5))
        plt.plot(ma100, color='red', label='100-day MA')
        plt.plot(df.Close, color='yellow', label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        # Plot with 100-day and 200-day Moving Averages
        st.subheader('Closing Price vs Time Chart with 100-day and 200-day Moving Averages')
        ma200 = df.Close.rolling(200).mean()
        fig = plt.figure(figsize=(10, 5))
        plt.plot(ma100, color='red', label='100-day MA')
        plt.plot(ma200, color='green', label='200-day MA')
        plt.plot(df.Close, color='yellow', label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        # Splitting data into training and testing sets
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        # Ensure non-empty training data
        if data_training.empty or data_testing.empty:
            st.error("Insufficient data for training or testing. Please try a different currency or timeframe.")
        else:
            # Scaling data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Linear Regression Model
            model = LinearRegression()

            # Preparing training data
            x_train, y_train = [], []
            for i in range(100, len(data_training_array)):
                x_train.append(data_training_array[i-100:i, 0])
                y_train.append(data_training_array[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Training the model
            model.fit(x_train, y_train)

            # Preparing testing data
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test, y_test = [], []
            for i in range(100, len(input_data)):
                x_test.append(input_data[i-100:i, 0])
                y_test.append(input_data[i, 0])
            x_test, y_test = np.array(x_test), np.array(y_test)

            # Making predictions
            y_predicted = model.predict(x_test)

            # Scaling back predictions and original values
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Final graph
            st.subheader('Prediction vs Original')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.style.use('dark_background')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)
