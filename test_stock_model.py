import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from stock_RNN import *
from future import *


# Load stock price data (e.g., closing prices)
data = pd.read_csv("real_stock_prices.csv")  # Assume "date" and "close" columns
prices = data["Close"].values

# Normalize data
scaler = MinMaxScaler()
prices = scaler.fit_transform(prices.reshape(-1, 1))


# Load the entire model
model = torch.load("rnn_stock_model.pth")

# Assume `prices` is the full dataset (scaled), and the model is trained
recent_data = prices  # Use the scaled dataset

# Predict the next 7 days
days_to_predict = 30
seq_length = 10
future_prices = predict_future(model, recent_data, seq_length, days_to_predict, scaler)

# Print predictions
print("Predicted Prices for the Next {} Days:".format(days_to_predict))
print(future_prices)
