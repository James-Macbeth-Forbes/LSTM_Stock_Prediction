# Stock Price Prediction with LSTM
This project uses a bidirectional LSTM model to predict stock prices based on historical closing prices. It fetches data from Yahoo Finance, preprocesses it, trains the model with cross-validation, and forecasts future prices.

Note: This model is for educational purposes only and should not be used for financial decisions.

# Features:
1. Fetches historical stock data from Yahoo Finance
2. Preprocesses and normalizes data for training
3. Uses a bidirectional LSTM with dropout regularization
4. Implements K-fold cross-validation for robustness
5. Predicts stock prices for the next 6 months
6. Visualizes past and predicted prices
# Dependencies
Ensure you have Python installed, then install the required libraries using:

<b> pip install yfinance pandas numpy matplotlib scikit-learn tensorflow </b>

# How to Use
1. Clone this repository or copy the script.
2. Run the script and enter a stock symbol (e.g., AAPL, TSLA).
3. The model will fetch data, train, evaluate, and predict future stock prices.
4. A graph will be displayed showing actual vs. predicted prices.
   
# ⚠Disclaimer⚠
This project is for educational and research purposes only. It does not provide financial advice or guarantee accuracy. Always conduct your own research before making financial decisions.
