# crypto_price_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
def load_model():
    try:
        with open('crypto_prediction_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
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
def preprocess_data(data):
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

  future_candles = 5;
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

def extract_features(data):
    feature_columns = ['timestamp', 'Up Trend', 'Down Trend', 'Tenkan', 'Kijun', 'Chikou',
              'SenkouA', 'SenkouB', 'Basis', 'Upper', 'Lower', 'Volume',
              'Volume MA', '%K', '%D', 'Aroon Up', 'Aroon Down', 'RSI', 'RSI-based MA', 'Upper Bollinger Band',
              'Lower Bollinger Band', 'OnBalanceVolume', 'Smoothing Line', 'Histogram', 'MACD', 'Signal']
    features = data[feature_columns].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return features


# Create a Streamlit app
def main():
    st.title("Cryptocurrency Price Prediction")

    # Load model
    model = load_model()
    
    # Print the loaded model to verify it's not None
    print("Model:", model)
    
    # User uploads data
    user_uploaded_data = st.file_uploader("Upload your cryptocurrency data (CSV file):", type=["csv"])

    if user_uploaded_data is not None:
        # Display user-uploaded data
        st.write("User-uploaded data:")
        crypto_data = load_data(user_uploaded_data)
        st.write("Preview of uploaded Crypto Data")
        st.dataframe(crypto_data, height=400)

        if crypto_data is not None:
            try:
                # Preprocess user data
                preprocessed_data = preprocess_data(crypto_data)
                st.write("Preview of preprocessed Crypto Data")
                st.dataframe(preprocessed_data, height=400)

                # Extract features from preprocessed data
                input = extract_features(preprocessed_data)
                st.write(input)
                
                # Make predictions
                prediction = model.predict(input)
                st.write("Predicted Price:", prediction)

            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main()
