import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from stock_RNN import *


def load_and_prepare_dataset(dataset_file_name):
    # Load stock price data (e.g., closing prices)
    data = pd.read_csv(dataset_file_name)  # Assume "date" and "close" columns
    prices = data["Close"].values

    # Normalize data
    scaler = MinMaxScaler()
    prices = scaler.fit_transform(prices.reshape(-1, 1))

    # Create sequences
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(x), np.array(y)

    seq_length = 20
    x, y = create_sequences(prices, seq_length)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return [x_tensor, y_tensor, scaler]


def train_model(x_tensor, y_tensor):
    # Hyperparameters
    input_size = 1
    hidden_size = 50
    num_layers = 2
    output_size = 1
    learning_rate = 0.001
    num_epochs = 100

    # Initialize the model, loss function, and optimizer
    model = StockRNN(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Reshape input for LSTM (batch_size, seq_length, input_size)
    x_tensor = x_tensor.view(x_tensor.size(0), x_tensor.size(1), input_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    return model