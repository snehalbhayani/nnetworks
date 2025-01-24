import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from stock_RNN import *
from future import *


def predict_future(model, recent_data, seq_length, days_to_predict, scaler):
    """
    Predict future stock prices.
    
    Args:
        model: Trained PyTorch RNN model.
        recent_data: Most recent `seq_length` days of stock prices (scaled).
        seq_length: Number of days used in each sequence.
        days_to_predict: Number of future days to predict.
        scaler: The scaler used to normalize and inverse-transform the data.
    
    Returns:
        List of future predictions (rescaled to original scale).
    """
    model.eval()
    predictions = []
    input_seq = recent_data[-seq_length:].tolist()  # Start with the latest data
    
    for _ in range(days_to_predict):
        # Convert input_seq to PyTorch tensor
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).view(1, seq_length, 1)
        
        # Predict the next value
        with torch.no_grad():
            next_price = model(input_tensor).item()
        
        # Append the prediction
        predictions.append(next_price)
        
        # Update the input sequence
        input_seq.append([next_price])  # Add the predicted price
        input_seq.pop(0)  # Remove the oldest price
    
    # Rescale predictions to original scale
    predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions_rescaled.flatten()
