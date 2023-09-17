# crypto_price_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import tensorflow as tf
import sklearn
import time
import threading

#for modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#for saving model
import pickle

# #for plotting
import matplotlib.pyplot as plt

# Load your trained deep learning model
def load_model(data_rows, future_candles):
    try:
        with open('model_ep100_bs30_2.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model

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
  target_col = 'future_candle';

  # Loop through all rows in the DataFrame
  for i in range(len(df)):
      # Check if there are at least N more rows after the current row
      if i + future_candles < len(df):
          # Add the 'close' data of the i+Nth row to the i-th row as a new column
          df.loc[i, target_col] = df.loc[i + future_candles, 'close']
      else:
          # If there are not enough rows left, fill the 'Close_nth' column with NaN
          df.loc[i, target_col] = None

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

def extract_features(target_col,future_candle,data,sequence_length_in):
    feature_columns = ['open', 'high', 'low', 'close', 'Plot', 'Up Trend', 'Down Trend', 'Tenkan', 'Kijun', 'Chikou', 'SenkouA', 'SenkouB', 
                       'Basis', 'Upper', 'Lower', 'Plot.1', 'Plot.2', 'Plot.3', 'Plot.4', 'Volume', 'Volume MA', '%K', '%D', 'Aroon Up', 'Aroon Down', 
                       'RSI', 'RSI-based MA', 'Upper Bollinger Band', 'Lower Bollinger Band', 'Plot.5', 'OnBalanceVolume', 'Smoothing Line', 'Histogram', 'MACD', 'Signal', target_col]
    X = data[feature_columns].values
    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)
    y = data[target_col].values
    y = np.where(y <= 0, 1e-10, y)
    y = np.log10(y)

    sequence_length = sequence_length_in

    X_sequences = []
    y_sequences = []

    for i in range(len(X_scaled) - sequence_length + 1):
        X_sequences.append(X_scaled[i : i + sequence_length])
        y_sequences.append(y[i + sequence_length - 1])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # Split into train and test sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(X_sequences))

    X_train = X_sequences[:split_index]
    y_train = y_sequences[:split_index]

    X_test = X_sequences[split_index:]
    y_test = y_sequences[split_index:]

    return X_train, y_train, X_test, y_test, feature_columns

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
    st.write("Summary Statistics on the uploaded dataset")
    st.dataframe(df.describe())

def make_prediction(model,input):
    # Make predictions
    prediction = model.predict(input)
    return prediction

def train_model(X_train,y_train,epoch_in,batch_size_in,sequence_length_in,feature_columns):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length_in, len(feature_columns))))
    model.add(Dense(1, activation='linear'))  # Output layer
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Perform one training epoch
    model.fit(X_train, y_train, epochs=epoch_in, batch_size=batch_size_in, validation_split=0.1)

    # Notify when training is complete
    st.success("Model training is complete!")

    return model

def permutation_feature_importance(model, X, y_true, feature_names):
    perm_importance = {}
    y_pred = model.predict(X)
    baseline_error = mean_squared_error(y_true, y_pred)

    for feature_idx in range(X.shape[1]):
        X_permuted = X.copy()
        X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
        y_pred_permuted = model.predict(X_permuted)
        permuted_error = mean_squared_error(y_true, y_pred_permuted)
        perm_importance[feature_names[feature_idx]] = baseline_error - permuted_error

    return perm_importance

def main():

    st.title("Cryptocurrency Price Prediction")
    st.write("User Guide")
    st.write("1. Select a Cryptocurrecny that you would like to predict from TradingView")
    st.write("2. Apply the following technical indicators to the chart before exporting")
    st.write("   (EMA 20/50/100/200)")
    st.write("   (Bollinger Bands)")
    st.write("   (Volume)")
    st.write("   (Ichimoku Cloud)")
    st.write("   (RSI)")
    st.write("   (MACD)")
    st.write("   (Stochastic)")
    st.write("   (OnBalanceVolume)")
    st.write("   (SuperTrend))")
    st.write("3. Export the chart in ISO timeframe")
    st.write("4. Upload the CSV file here")
    st.write("5. Select the number of future candle you want to predict")
    st.write("6. Evaluate the prediction result")
    st.write("You may re-train the model if you are not satisfied with the result")
    st.write("")

    st.write("Please note that this cryptocurrency price prediction tool is for informational purposes only and should not be considered financial advice. Trading cryptocurrencies involves inherent risks, and any losses incurred are the sole responsibility of the user. It is essential to conduct thorough research and consult with a financial advisor before making any trading decisions.")

    with st.sidebar:
        st.title("User Input")

        # User uploads data
        user_uploaded_data = st.file_uploader("Upload your cryptocurrency data (CSV file):", type=["csv"])

        # Define a slider for selecting the number of future candles
        selected_option = st.slider("Select the n-th future you want to predict:", min_value=1, max_value=20, value=1)

        # Button to perform modelling with the input
        submit_button = st.button("Train Model")

    if user_uploaded_data is not None and submit_button and selected_option is not None:

        crypto_data = load_data(user_uploaded_data)
        future_candle = int(selected_option)
        target_col = 'future_candle';

        # Display user-uploaded data
        st.write("Preview of uploaded Crypto Data")
        st.dataframe(crypto_data, height=400)

        # get data size
        data_rows = len(crypto_data)

        # Load model
        model = load_model(data_rows, future_candle)

        if crypto_data is not None:
            try:
                sequence_length = 20
                epoch = 50
                batch_size = 30

                # Preprocess user data
                preprocessed_data = preprocess_data(crypto_data,future_candle)
                st.write("Preview of preprocessed Crypto Data")
                st.dataframe(preprocessed_data, height=400)

                st.subheader("Exploratory Data Analysis")

                # Create a list to store EDA charts and subheaders
                eda_data = []

                # Iterate through columns and create EDA charts
                for column in preprocessed_data.columns:
                    # Create a dictionary to store chart and subheader
                    chart_data = {"subheader": f"EDA for {column}", "chart": None}

                    # Create your EDA chart (you can replace this with your actual EDA code)
                    # For example, let's create a histogram for numeric columns
                    if preprocessed_data[column].dtype in [np.float64, np.int64]:
                        fig, ax = plt.subplots()
                        ax.hist(preprocessed_data[column], bins=20)
                        ax.set_xlabel(column)
                        ax.set_ylabel("Frequency")
                        ax.set_title(column, fontsize=20)

                        # Save the chart and subheader in the dictionary
                        chart_data["chart"] = fig

                    # Append the chart data to the list
                    eda_data.append(chart_data)

                # Create columns for displaying charts side by side
                columns = st.columns(3)  # Adjust the number of columns as needed

                # Display the charts and subheaders side by side
                for i, chart_data in enumerate(eda_data):
                    with columns[i % 3]:  # Switch to the next column after every 2 charts
                        # st.subheader(chart_data["subheader"])
                        st.pyplot(chart_data["chart"], use_container_width=True)

                # Extract features and scale input from preprocessed data
                X_train, y_train, X_test, y_test, feature_columns = extract_features(target_col,future_candle,preprocessed_data,sequence_length)

                # Train model and predict
                with st.spinner("Training the model..."):
                    lstm_model = train_model(X_train,y_train,epoch,batch_size,sequence_length,feature_columns)
                    prediction = lstm_model.predict(X_test)
                
                # Inverse scale to get the actual price
                prediction = 10**prediction
                y_test = 10**y_test

                y_test_filtered = y_test[:-future_candle]
                future_prediction = prediction[-future_candle:]
                pred_without_fc = prediction[:-future_candle]

                # Show prediction of n future candle
                st.subheader("Predicted Future Price")
                st.write(future_prediction)
                last_pred_price = np.atleast_1d(prediction[-1]).item()
                last_row_price = np.atleast_1d(y_test_filtered[-1]).item()
                price_diff = np.round(last_pred_price - last_row_price,4)

                if price_diff > 0:
                  percentage = "+" + str(price_diff/last_pred_price * 100) + "%"
                  trend = "Up Trend"
                  font_color = 'green'
                else:
                  percentage = str(price_diff/last_pred_price * 100) + "%"
                  trend = "Down Trend"
                  font_color = 'red'


                st.subheader(f"Price difference" )
                st.write(f"{str(last_row_price)} -> {str(last_pred_price)}")
                st.write(str(price_diff))
                st.write(percentage)


                st.subheader("Trend")
                # Create a subheader with a specified font color
                st.markdown(f'<p style="color: {font_color};">{trend}</p>', unsafe_allow_html=True)

                # Calculate evaluation metrics
                mae = mean_absolute_error(y_test_filtered.flatten(), pred_without_fc.flatten())
                mse = mean_squared_error(y_test_filtered.flatten(), pred_without_fc.flatten())
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_filtered.flatten(), pred_without_fc.flatten())

                # Display evaluation metrics
                st.subheader("Evaluation Metrics")
                st.write(f'MAE: {mae:.5f}')
                st.write(f'MSE: {mse:.5f}')
                st.write(f'RMSE: {rmse:.5f}')
                st.write(f'R2 Score: {r2:.2f}')

                # Evaluate model
                st.subheader("Actual vs Predicted Price")
                results_df = pd.DataFrame({
                    "Actual Result": y_test.flatten(),
                    "Predicted Result": prediction.flatten()
                })
                st.write(results_df)

                # Create a plot
                fig, ax = plt.subplots()
                ax.plot(y_test_filtered, label='Actual Data', color='red')
                ax.plot(prediction, label='Predicted Data', color='blue')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
                st.pyplot(fig)

                # Visualize feature importance
                # Calculate permutation feature importance
                st.subheader("Permutation Feature Importance")
                perm_importance = permutation_feature_importance(lstm_model, X_test, y_test, feature_columns)

                # Sort feature importance in descending order
                sorted_importance = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)

                # Display the results using Streamlit
                # st.write("Permutation Feature Importance:")
                # for feature, importance in sorted_importance:
                #     st.write(f"{feature}: {importance}")

                 # Apply log transformation to importance values
                importance_df = pd.DataFrame(sorted_importance, columns=["Feature", "Importance"])
                importance_df["Log Importance"] = np.log1p(importance_df["Importance"])

                # Create a bar chart
                st.bar_chart(importance_df.set_index("Feature")["Log Importance"])


            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main()
