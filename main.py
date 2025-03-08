print("Preparing everything...")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import os


# Fetching the data from Yahoo Finance
def fetch_data(stock_symbol):
    start_date = (datetime.now() - timedelta(days=7 * 365)).strftime("%Y-%m-%d")
    data = yf.download(stock_symbol, start=start_date, end=datetime.now().strftime("%Y-%m-%d"))
    data.to_csv(f'{stock_symbol}.csv')
    return data


# Data Preprocessing
def preprocess_data(data):
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


# Creating a dataset with a time step
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# Building the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout for regularization
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Deleting the CSV file
def delete_csv(stock_symbol):
    if os.path.exists(f"{stock_symbol}.csv"):
        os.remove(f"{stock_symbol}.csv")
        print(f"{stock_symbol}.csv deleted.")
    else:
        print(f"{stock_symbol}.csv not found.")


# Cross-validation function
def cross_validate_model(X, y, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    val_loss_per_fold = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = build_model((X_train.shape[1], 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=[early_stopping])

        val_loss = model.evaluate(X_val, y_val, verbose=0)
        val_loss_per_fold.append(val_loss)

    avg_val_loss = np.mean(val_loss_per_fold)
    print(f"Average validation loss across {k}-folds: {avg_val_loss}")
    return avg_val_loss


# Main function to execute the stock price prediction
def main():
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, TSLA): ").upper()
    data = fetch_data(stock_symbol)

    scaled_data, scaler = preprocess_data(data)
    X, y = create_dataset(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    cross_validate_model(X, y, k=5)

    test_period = 180
    train_size = len(scaled_data) - test_period

    # Train and test data split, reshaping data
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Building and training the model
    model = build_model((X_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 4)))))[:, 0]
    y_test = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4)))))[:, 0]

    prediction_input = scaled_data[-60:].reshape(1, 60, 1)
    future_pred = []
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 181)]

    for _ in range(180):
        future_price = model.predict(prediction_input)
        future_pred.append(future_price[0][0])
        prediction_input = np.append(prediction_input[:, 1:, :], future_price.reshape(1, 1, 1), axis=1)

    future_pred = scaler.inverse_transform(np.array(future_pred).reshape(-1, 1))

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[:len(y_train)], scaler.inverse_transform(scaled_data[:len(y_train)]), color='blue',
             label='Training Data')
    plt.plot(data.index[len(y_train):len(y_train) + len(y_test)], y_test, color='blue',
             label='Actual Stock Price (Test Data)')
    plt.plot(data.index[len(y_train):len(y_train) + len(predictions)], predictions, color='red',
             label='Predicted Stock Price (Test Data)')
    plt.plot(future_dates, future_pred, color='green', label=f'{stock_symbol} Predicted Stock Price for Next 6 Months')
    plt.title(f'{stock_symbol} Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    # Deleting the CSV file
    delete_csv(stock_symbol)


if __name__ == "__main__":
    main()
