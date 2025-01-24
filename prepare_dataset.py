import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from stock_RNN import *
from utilities import *

# Load and clean the dataset
x_tensor, y_tensor, scaler = load_and_prepare_dataset("real_stock_prices.csv")

# Train the NN model
model = train_model(x_tensor, y_tensor)

# Save the model
torch.save(model, "rnn_stock_model.pth")

# Test the model
model.eval()
with torch.no_grad():
    predictions = model(x_tensor).numpy()

# Rescale predictions back to original scale
predictions = scaler.inverse_transform(predictions)
y_actual = scaler.inverse_transform(y_tensor.numpy().reshape(-1, 1))

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.plot(y_actual, label="Actual Prices", color="blue")
plt.plot(predictions, label="Predicted Prices", color="red")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

