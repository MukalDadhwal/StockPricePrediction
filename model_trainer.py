import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
from utils.preprocessing import prepare_data_for_lstm

def build_lstm_model(lookback=60, dropout_rate=0.3):
    """
    Build a lightweight 2-layer LSTM model with configurable dropout
    
    Args:
        lookback: Number of time steps to look back
        dropout_rate: Dropout rate to prevent overfitting (0.3-0.5 recommended)
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout_rate),  # Increased dropout to reduce overfitting
        
        # Second LSTM layer
        LSTM(units=50, return_sequences=False),
        Dropout(dropout_rate),  # Increased dropout
        
        # Output layer
        Dense(units=1)
    ])
    
    # Compile model with learning rate adjustment
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

def train_model(ticker="AAPL", lookback=60, epochs=50, batch_size=32, dropout_rate=0.3):
    """
    Train LSTM model for a specific ticker
    
    Args:
        ticker: Stock ticker symbol
        lookback: Days to look back
        epochs: Training epochs
        batch_size: Batch size
        dropout_rate: Dropout rate for regularization
    """
    print(f"📊 Training LSTM model for {ticker}...")
    print(f"   - Dropout rate: {dropout_rate}")
    print(f"   - Epochs: {epochs}")
    
    # Download data (5 years)
    print("📥 Downloading stock data...")
    end_date = date.today()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = df.reset_index()
    df['Ticker'] = ticker
    
    # Prepare data
    print("🔧 Preparing data (chronological split)...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, _ = prepare_data_for_lstm(df, ticker, lookback)
    
    print(f"✅ Training data shape: {X_train.shape}")
    print(f"✅ Testing data shape: {X_test.shape}")
    
    # Build model with increased dropout
    print("🏗️ Building LSTM model...")
    model = build_lstm_model(lookback, dropout_rate=dropout_rate)
    
    # Early stopping with more patience and restore best weights
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n✅ Training Loss: {train_loss:.6f}")
    print(f"✅ Testing Loss: {test_loss:.6f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/lstm_{ticker}.h5"
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Save scaler
    import pickle
    scaler_path = f"models/scaler_{ticker}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to {scaler_path}")
    
    # Save model metrics
    metrics = {
        'train_loss': float(train_loss),
        'val_loss': float(model.evaluate(X_val, y_val, verbose=0)),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0])
    }
    
    import json
    metrics_path = f"models/metrics_{ticker}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to {metrics_path}")
    
    return model, history, scaler

if __name__ == "__main__":
    # Train for AAPL (you can change ticker)
    train_model(ticker="AAPL", lookback=60, epochs=50, batch_size=32)
    train_model(ticker="MSFT", lookback=60, epochs=50, batch_size=32)
    train_model(ticker="GOOGL", lookback=60, epochs=50, batch_size=32)
    train_model(ticker="AMZN", lookback=60, epochs=50, batch_size=32)
    train_model(ticker="TSLA", lookback=60, epochs=50, batch_size=32)