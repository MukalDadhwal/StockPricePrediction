"""
Quick script to generate metrics for already trained models
Run this if you trained models before the metrics feature was added
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from tensorflow.keras.models import load_model
from utils.preprocessing import prepare_data_for_lstm
import pickle
import json
import os

def generate_metrics_for_model(ticker, lookback=60):
    """Generate and save metrics for an existing trained model"""
    
    print(f"\n{'='*60}")
    print(f"📊 Generating metrics for {ticker}")
    print(f"{'='*60}")
    
    try:
        # Check if model exists
        model_path = f"models/lstm_{ticker}.h5"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False
        
        # Load model
        print("📥 Loading model...")
        model = load_model(model_path)
        
        # Download data (5 years)
        print("📥 Downloading stock data...")
        end_date = date.today()
        start_date = end_date - timedelta(days=5*365)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df = df.reset_index()
        df['Ticker'] = ticker
        
        # Prepare data (with proper train/val/test split)
        print("🔧 Preparing data (chronological split)...")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, _ = prepare_data_for_lstm(df, ticker, lookback)
        
        # Evaluate model
        print("📊 Evaluating model...")
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"✅ Training Loss: {train_loss:.6f}")
        print(f"✅ Validation Loss: {val_loss:.6f}")
        print(f"✅ Testing Loss: {test_loss:.6f}")
        
        # Save metrics
        metrics = {
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'test_loss': float(test_loss),
            'epochs_trained': 50,  # Default value for pre-existing models
            'final_train_loss': float(train_loss),
            'final_val_loss': float(val_loss),
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'test_samples': int(X_test.shape[0])
        }
        
        metrics_path = f"models/metrics_{ticker}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Metrics saved to {metrics_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    print("\n" + "="*60)
    print("🚀 GENERATING METRICS FOR TRAINED MODELS")
    print("="*60)
    
    results = {}
    for ticker in tickers:
        success = generate_metrics_for_model(ticker)
        results[ticker] = "✅ Success" if success else "❌ Failed"
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    for ticker, status in results.items():
        print(f"{ticker:6s}: {status}")
    print("="*60)
    print("✅ Done!")
