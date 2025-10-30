import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from utils.preprocessing import prepare_data_for_lstm

def predict_next_days(ticker, df, days=30, lookback=60):
    """
    Predict future stock prices
    
    Args:
        ticker: Stock ticker
        df: DataFrame with historical data
        days: Number of days to predict
        lookback: Lookback period (must match training)
    
    Returns:
        predictions: Array of predicted prices
    """
    # Load model and scaler
    model = load_model(f"models/lstm_{ticker}.h5")
    with open(f"models/scaler_{ticker}.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare data (we only need the filtered/sorted ticker_data at the end)
    _, _, _, _, _, _, _, ticker_data = prepare_data_for_lstm(df, ticker, lookback)
    
    # Get last 60 days
    prices = ticker_data['Close'].values.reshape(-1, 1)
    scaled_prices = scaler.transform(prices)
    
    # Start with last 60 days
    last_sequence = scaled_prices[-lookback:].reshape(1, lookback, 1)
    
    predictions = []
    
    for _ in range(days):
        # Predict next day
        pred = model.predict(last_sequence, verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence (remove first, add prediction)
        last_sequence = np.append(last_sequence[:, 1:, :], [[[pred[0, 0]]]], axis=1)
    
    # Inverse transform to get actual prices
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions.flatten()