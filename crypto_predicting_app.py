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
import random
import altair as alt
import calendar

#for modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# #for plotting
import matplotlib.pyplot as plt

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

  df.drop(columns=['Upper Bollinger Band','Lower Bollinger Band'], inplace=True)

  # Replace NaN values in the 'Up Trend' and 'Down Trend' columns with 0
  df['Up Trend'].fillna(0, inplace=True)
  df['Down Trend'].fillna(0, inplace=True)

  # Apply backward fill to fill missing values
  df['Chikou'].fillna(method='ffill', inplace=True)

  # Convert the 'time' column to a datetime object
  df['time'] = pd.to_datetime(df['time'])

  # Convert the "time" column to datetime format
  df['timestamp'] = pd.to_datetime(df['time'])

  # Extract date and time components into separate columns
  df['day'] = df['time'].dt.dayofweek + 1  # Adding 1 to make Monday start from 1
  df['hour'] = df['time'].dt.hour  # Adding 1 to make 01:00 start from 1

    # Convert datetime to timestamps (datetime64[ns])
  df['timestamp'] = df['timestamp'].astype('int64') // 10**9  # Convert to seconds

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

  # Apply backward fill to fill missing values
  df['future_candle'].fillna(method='ffill', inplace=True)

  column_mapping = {
    'Plot': 'Super Trend Plot',
    'Plot.1': 'EMA20',
    'Plot.2': 'EMA50',
    'Plot.3': 'EMA100',
    'Plot.4': 'EMA200',
    'Plot.5': 'RSI Plot'
  }

  # Rename the columns using the mapping
  df.rename(columns=column_mapping, inplace=True)

  return df

def extract_features(target_col,future_candle,data,sequence_length_in):
    feature_columns = ['open', 'high', 'low', 'close', 'Super Trend Plot', 'Up Trend', 'Down Trend', 'Tenkan', 'Kijun', 'Chikou', 'SenkouA', 'SenkouB',
                       'Basis', 'Upper', 'Lower', 'EMA20', 'EMA50', 'EMA100', 'EMA200', 'Volume', 'Volume MA', '%K', '%D', 'Aroon Up', 'Aroon Down',
                       'RSI', 'RSI-based MA', 'RSI Plot', 'OnBalanceVolume', 'Smoothing Line', 'Histogram',
                      'MACD', 'Signal', 'DI+', 'DI-', 'ADX', 'day', 'hour']

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

    # Calculate the date difference
    date_difference = time_end - time_start

    # Extract the number of days and hours
    days = date_difference.days
    hours = date_difference.seconds // 3600

    # Create a formatted string to display the date difference
    date_difference_str = f"{days} days, {hours} hours"

    # Display dataset start and end timestamp along with the date difference
    st.write(f"**Dataset Period:** {time_start} - {time_end} ({date_difference_str})")

    # Show total data rows
    st.write("**Total Rows:** ", len(df))

    # Count number of volume
    total_volume = 0
    for row in data['Volume']:
      total_volume += row

    st.write("**Total Volume:** ",total_volume)

    # Count number of uptrend
    found_zero = False
    uptrend_count = 0

    for row in data['Up Trend']:
      if row != 0 and found_zero == False:
        uptrend_count += 1
        found_zero = True
      elif row == 0:
        found_zero = False

    # Count number of downtrend
    found_value = False
    downtrend_count = 0

    for row in data['Down Trend']:
      if row == 0 and found_value == False:
        downtrend_count += 1
        found_value = True
      elif row != 0:
        found_value = False

    st.write("**Number of trends based on SuperTrend Indicator:**")
    st.write("**Total uptrend:** ", uptrend_count)
    st.write("**Total downtrend:** ", downtrend_count)

    # Convert numeric days to day names
    df['day_str'] = df['day'].apply(lambda x: calendar.day_name[x - 1])

    # Calculate the total 'close' for each 'day'
    total_close_by_day = df.groupby('day_str')['close'].sum().reset_index()

    # Calculate the total 'volume' for each 'day'
    total_volume_by_day = data.groupby('day_str')['Volume'].sum().reset_index()

    # Specify the desired order of days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Define a color scale for your charts (you can customize this)
    color_scale = alt.Scale(domain=day_order,
                            range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])

    # Create a colorful Altair bar chart for 'close' values
    chart_a = alt.Chart(total_close_by_day).mark_bar().encode(
        x=alt.X('day_str:N', sort=day_order),
        y='close:Q',
        color=alt.Color('day_str:N', scale=color_scale, legend=None)
    ).properties(
        title='Total Close by Day',
        width=600,
        height=300
    )

    # Create a colorful Altair bar chart for 'Volume' values
    chart_b = alt.Chart(total_volume_by_day).mark_bar().encode(
        x=alt.X('day_str:N', sort=day_order),
        y='Volume:Q',
        color=alt.Color('day_str:N', scale=color_scale, legend=None)
    ).properties(
        title='Total Volume by Day',
        width=600,
        height=300
    )

    # Display the Altair charts using Streamlit
    st.altair_chart(chart_a)
    st.altair_chart(chart_b)

    # Show dataset EDA on each column
    st.subheader("Statistics Report")
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

    st.title("Cryptocurrency Trend Prediction")
    st.subheader("Please follow the instructions below to start using the system")

    # Create an expander to hold the instructions
    with st.expander("Instructions"):
        st.write("Follow these steps to predict cryptocurrency prices using our tool:")

        st.markdown("1. **Select a Cryptocurrency**: Choose the cryptocurrency you want to predict from TradingView.")
        st.markdown("2. **Apply Technical Indicators**: Apply the following technical indicators to the chart before exporting:")
        st.write("- Volume")
        st.write("- EMA 20/50/100/200")
        st.write("- Bollinger Bands")
        st.write("- Ichimoku Cloud")
        st.write("- RSI")
        st.write("- MACD")
        st.write("- Stochastic")
        st.write("- Aroon")
        st.write("- ADX and DI")
        st.write("- OnBalanceVolume")
        st.write("- SuperTrend")

        st.markdown("3. **Export the Chart**: Export the chart in ISO timeframe.")

        st.markdown("4. **Upload the CSV File**: Upload the exported CSV file here.")

        st.markdown("5. **Select the Number of Future Candles**: Choose the number of future candles you want to predict.")

        st.markdown("6. **Evaluate the Prediction Result**: Review and evaluate the prediction result.")


    with st.sidebar:
        st.title("User Input")

        # User uploads data
        user_uploaded_data = st.file_uploader("Upload your cryptocurrency data (CSV file):", type=["csv"])

        # Define a slider for selecting the number of future candles
        selected_option = st.slider("Select the n-th future you want to predict:", min_value=1, max_value=20, value=1)

        # Button to perform modelling with the input
        submit_button = st.button("Predict Market Trend")

        # Add an HTML element for the disclaimer
        st.markdown(
            """<p style="font-size: 12px; text-align: left; margin-top: 20px;">
            Please note that this cryptocurrency price prediction tool is for informational purposes only and should not be considered financial advice. Trading cryptocurrencies involves inherent risks, and any losses incurred are the sole responsibility of the user. It is essential to conduct thorough research and consult with a financial advisor before making any trading decisions.
            </p>""",
            unsafe_allow_html=True,
        )

    if user_uploaded_data is not None and submit_button and selected_option is not None:

        crypto_data = load_data(user_uploaded_data)
        future_candle = int(selected_option)
        target_col = 'future_candle';

        # Display user-uploaded data
        st.subheader("Preview of uploaded Crypto Data")
        st.dataframe(crypto_data, height=400)

        # get data size
        data_rows = len(crypto_data)

        if crypto_data is not None:
            try:
                sequence_length = 20
                epoch = 100
                batch_size = 32

                # Preprocess user data
                preprocessed_data = preprocess_data(crypto_data,future_candle)
                st.subheader("Preview of preprocessed Crypto Data")
                st.dataframe(preprocessed_data, height=400)

                # Show dashboard
                show_dashboard(preprocessed_data)

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

                # Create a copy of future_prediction (optional)
                future_prediction_copy = future_prediction.copy()

                # Prepare the data for the table
                future_prediction_data = []

                for i, prediction in enumerate(future_prediction_copy):
                    future_prediction_data.append([f'Candle {i + 1}', prediction])

                # Create a DataFrame from the data
                future_prediction_df = pd.DataFrame(future_prediction_data, columns=['Candle', 'Predicted Price'])

                # Display the table
                st.subheader("Predicted Future Price")
                st.table(future_prediction_df)

                last_pred_price = np.atleast_1d(prediction[-1]).item()
                last_row_price = np.atleast_1d(y_test_filtered[-1]).item()
                price_diff = last_pred_price - last_row_price

                if price_diff > 0:
                  percentage = price_diff/last_pred_price
                  trend = "Up Trend"
                  font_color = 'green'
                else:
                  percentage = price_diff/last_pred_price
                  trend = "Down Trend"
                  font_color = 'red'

                # Show price difference
                formatted_last_row_price = "${:,.2f}".format(last_row_price)
                formatted_last_pred_price = "${:,.2f}".format(last_pred_price)
                formatted_price_diff = "${:,.2f}".format(price_diff)
                formatted_percentage = "{:.2%}".format(percentage)
                price_direction_symbol = "📈" if last_pred_price > last_row_price else "📉"
                color = "green" if last_pred_price > last_row_price else "red"
                st.subheader("Price Difference")
                st.markdown(f"<p style='color: {color}; font-size: 18px; font-weight: bold;'>{price_direction_symbol} Actual Price:{formatted_last_row_price} ⇒ Predicted Price:{formatted_last_pred_price}</p>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: {color}; font-weight: bold;'>Price Difference: {formatted_price_diff}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: {color}; font-weight: bold;'>Percentage Change: {formatted_percentage}</span>", unsafe_allow_html=True)

                # Show market trend
                st.subheader("Trend")
                st.markdown(f'<p style="color: {font_color}; font-weight: bold;">{trend}</p>', unsafe_allow_html=True)

                # Calculate evaluation metrics
                mae = mean_absolute_error(y_test_filtered.flatten(), pred_without_fc.flatten())
                mse = mean_squared_error(y_test_filtered.flatten(), pred_without_fc.flatten())
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_filtered.flatten(), pred_without_fc.flatten())

                # Display evaluation metrics
                st.subheader("Evaluation Metrics")
                st.write(f'**Mean Absolute Error (MAE):**: {mae:.5f}')
                st.write("MAE measures the average absolute difference between the predicted and actual values. Lower values indicate better accuracy. It tells you, on average, how far off the predictions are from the actual values.")
                st.write(f'**Mean Squared Error (MSE):** {mse:.5f}')
                st.write("MSE measures the average squared difference between the predicted and actual values. Lower values indicate better accuracy. It penalizes larger errors more than MAE.")
                st.write(f'**Root Mean Squared Error (RMSE):** {rmse:.5f}')
                st.write("RMSE is the square root of MSE. Like MSE, lower RMSE values indicate better accuracy. It's in the same unit as the target variable, making it easier to interpret.")
                st.write(f'**R-squared (R2) Score:** {r2:.2f}')
                st.write("R2 measures the proportion of the variance in the dependent variable (actual values) that is predictable from the independent variables (predictions). An R2 score of 1 indicates a perfect fit, while 0 means the model doesn't explain any variance. Higher values are better.")

                # Create a plot
                time_values = preprocessed_data['time']
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(time_values[-len(y_test):], y_test, label='Actual')
                ax.plot(time_values[-len(prediction):], prediction, label='Predicted')
                ax.set_xlabel('Time')
                ax.set_ylabel('Price')
                ax.legend()
                ax.plot(time_values[-future_candle:], y_test[-future_candle:], color='white', alpha=1)
                st.pyplot(fig)

                # Plot last 100 rows
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(time_values[-100:], y_test[-100:], label='Actual')
                ax.plot(time_values[-100:], prediction[-100:], label='Predicted')
                ax.set_xlabel('Time')
                ax.set_ylabel('Price')
                ax.grid(True)
                ax.legend()
                ax.plot(time_values[-future_candle:], y_test[-future_candle:], color='white', alpha=1)
                fig.tight_layout()
                st.pyplot(fig)

                # Visualize feature importance
                st.subheader("Permutation Feature Importance")
                perm_importance = permutation_feature_importance(lstm_model, X_test, y_test, feature_columns)
                sorted_importance = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
                importance_df = pd.DataFrame(sorted_importance, columns=["Feature", "Importance"])
                importance_df["Log Importance"] = np.log1p(importance_df["Importance"])

                # Create a bar chart
                st.bar_chart(importance_df.set_index("Feature")["Log Importance"])


            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main()
