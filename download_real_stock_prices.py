import yfinance as yf
import pandas as pd
import pdb

# Download historical stock data
ticker = "MSFT"  # Replace with the stock symbol of your choice
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")

# Extract the 'Close' price
close_prices = data[['Close']]

# Save to a CSV file (optional)
close_prices.to_csv("real_stock_prices.csv")

# Display the data
print(close_prices.head())
