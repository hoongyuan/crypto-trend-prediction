# crypto_price_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
# import tensorflow as tf
# import sklearn

# import matplotlib

# #for modeling
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#for saving model
import pickle

# #for plotting
# import matplotlib.pyplot as plt

# Load your trained deep learning model
def load_model(data_rows, future_candles):
    try:
        # for large dataset
        if data_rows > 10000:
          if future_candles == 1:
            with open('big_data_model_fc1_test.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            return model

          elif future_candles == 2:
            with open('big_data_model_fc2.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            return model

          elif future_candles == 5:
            with open('big_data_model_fc5.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            return model

          elif future_candles == 10:
            with open('big_data_model_fc10.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            return model

        # for small dataset
        else:
          if future_candles == 1:
            with open('small_data_model_fc1_without_scale.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            return model

          elif future_candles == 2:
            with open('small_data_model_fc2.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            return model

          elif future_candles == 5:
            with open('small_data_model_fc5.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            return model

          elif future_candles == 10:
            with open('small_data_model_fc10.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            return model

        # If none of the conditions match
        st.error("Model file not found for the specified configuration.")
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
    return None

# Load target_scaler model
def load_scaler():
    try:
      with open('target_scaler.pkl', 'rb') as model_file:
                y_scaler = pickle.load(model_file)
      return y_scaler
    except Exception as e:
      st.error(f"Error loading the scaler: {str(e)}")
    return None

# Load cryptocurrency data
def load_data(user_uploaded_data):
    if user_uploaded_data is not None:
        try:
            # Load user-uploaded data into a DataFrame
            crypto_data = pd.read_csv(user_uploaded_data)
            return crypto_data
        except Exception as e:
            st.error("Error loading data. Please check the file format and content.")
    return None

# Preprocess cryptocurrency data
def preprocess_data(data,future_candle):
  df = data
  # Convert the "time" column to datetime format
  df['time'] = pd.to_datetime(df['time'])

  # Extract date and time components into separate columns
  df['date'] = df['time'].dt.date
  df['time_of_day'] = df['time'].dt.time

  date_column = df.pop('date')
  time_column = df.pop('time_of_day')

  df.insert(0,'time_of_day',time_column)
  df.insert(0,'date',date_column)
  del df['time']

  future_candles = future_candle;
  target_col = 'Close_' + str(future_candles) + 'th';

  # Loop through all rows in the DataFrame
  for i in range(len(df)):
      # Check if there are at least N more rows after the current row
      if i + future_candles < len(df):
          # Add the 'close' data of the i+Nth row to the i-th row as a new column
          df.loc[i, target_col] = df.loc[i + future_candles, 'close']
      else:
          # If there are not enough rows left, fill the 'Close_nth' column with NaN
          df.loc[i, target_col] = None


  # Drop the rows by index labels
  df = df.drop(df.index[-future_candles:])

  # Use fillna() to replace NaN values with 0
  df = df.fillna(0)

  # Convert 'date' column to datetime type
  df['date'] = pd.to_datetime(df['date'])

  # Convert 'time_of_day' column to timedelta
  df['time_of_day'] = pd.to_timedelta(df['time_of_day'].astype(str))

  # Combine 'date' and 'time_of_day' columns to create a timestamp
  df['timestamp'] = df['date'] + df['time_of_day']

  # Convert datetime to timestamps (datetime64[ns])
  df['timestamp'] = df['timestamp'].astype('int64') // 10**9  # Convert to seconds

  df = df.drop(columns=['date'])
  df = df.drop(columns=['time_of_day'])
  return df

def extract_features(data_rows,future_candle,data):
    feature_columns = ['timestamp', 'Up Trend', 'Down Trend', 'Tenkan', 'Kijun', 'Chikou',
              'SenkouA', 'SenkouB', 'Basis', 'Upper', 'Lower', 'Volume',
              'Volume MA', '%K', '%D', 'Aroon Up', 'Aroon Down', 'RSI', 'RSI-based MA', 'Upper Bollinger Band',
              'Lower Bollinger Band', 'OnBalanceVolume', 'Smoothing Line', 'Histogram', 'MACD', 'Signal']
    features = data[feature_columns].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    if data_rows > 10000:
      sequence_length = 20
    else:
      if future_candle == 1 or future_candle == 2:
        sequence_length = 10
      elif future_candle == 5:
        sequence_length = 15
      elif future_candle == 10:
        sequence_length = 20

    X_sequences = []

    for i in range(len(X_scaled) - sequence_length + 1):
        X_sequences.append(X_scaled[i : i + sequence_length])

    X_sequences = np.array(X_sequences)

    return X_sequences

def show_dashboard(data):
    df = data

    # Show dataset start and end timestamp
    time_start = datetime.datetime.fromtimestamp(df['timestamp'].iloc[0])
    time_end = datetime.datetime.fromtimestamp(df['timestamp'].iloc[-1])

    sentence = "Dataset Period: " + str(time_start) + " - " + str(time_end)
    st.write(sentence)

    # Show total data rows
    st.write("Total Rows: ", len(df))

    # Show dataset EDA on each column
    st.dataframe(df.describe())

def make_prediction(model,input):

    # Make predictions
    prediction = model.predict(input)

    return prediction

# Create a Streamlit app
def main():
    st.title("Cryptocurrency Price Prediction")

    # User uploads data
    user_uploaded_data = st.file_uploader("Upload your cryptocurrency data (CSV file):", type=["csv"])

    # Define a list of options for the dropdown
    options = ["1", "2", "5", "10"]
    # Create a dropdown select box
    selected_option = st.selectbox("Select the n-th future you want to predict:", options)

    # Button to perform modelling with the input
    submit_button = st.button("Train Model")

    if user_uploaded_data is not None and submit_button and selected_option is not None:
        
        crypto_data = load_data(user_uploaded_data)
        future_candle = int(selected_option)

        # Display user-uploaded data
        st.write("Preview of uploaded Crypto Data")
        st.dataframe(crypto_data, height=400)

        # get data size
        data_rows = len(crypto_data)

        # Load model
        model = load_model(data_rows, future_candle)

        if crypto_data is not None:
            try:
                target_col = 'Close_' + str(future_candle) + 'th';

                # Preprocess user data
                preprocessed_data = preprocess_data(crypto_data,future_candle)
                st.write("Preview of preprocessed Crypto Data")
                st.dataframe(preprocessed_data, height=400)

                # Display dashboard of uploaded data
                show_dashboard(preprocessed_data)

                # Extract features and scale input from preprocessed data
                input = extract_features(data_rows,future_candle,preprocessed_data)

                prediction = make_prediction(model, input)

                st.write("Predicted Result:", prediction)

            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main()
