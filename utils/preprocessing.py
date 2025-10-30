import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_data_for_lstm(df, ticker, lookback=60, val_size=0.1, test_size=0.2):
    """
    Prepare stock data for LSTM training using a proper chronological split.

    Key properties:
    - No shuffling; preserves time order
    - Scaler is fit ONLY on the training window to avoid leakage
    - Returns explicit train/val/test sets

    Args:
        df: DataFrame with stock data. Must contain columns: ['Date', 'Close', 'Ticker']
        ticker: Stock ticker to filter and prepare
        lookback: Number of time steps to look back for each sample
        val_size: Fraction of samples used for validation (default 0.10)
        test_size: Fraction of samples used for test (default 0.20)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, ticker_data
    """
    assert 0 < test_size < 0.9, "test_size must be in (0, 0.9)"
    assert 0 < val_size < 0.5, "val_size must be in (0, 0.5)"
    assert (val_size + test_size) < 0.9, "val_size + test_size must be < 0.9 to leave room for training"

    # 1) Sort and filter
    ticker_data = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
    prices = ticker_data['Close'].values.reshape(-1, 1)

    if prices.shape[0] <= lookback + 10:
        raise ValueError("Not enough data after lookback to create train/val/test splits.")

    # 2) Determine split points in SAMPLE space (after sequence construction)
    n = prices.shape[0]
    n_samples = n - lookback  # number of (X, y) pairs
    # compute indices for splits
    train_frac = 1.0 - (val_size + test_size)
    train_end = max(int(train_frac * n_samples), 1)
    val_end = max(train_end + int(val_size * n_samples), train_end + 1)
    val_end = min(val_end, n_samples - 1)  # leave at least 1 for test when possible

    # 3) Fit scaler on TRAINING PRICES ONLY (up to the last training label index)
    # The last training target index in the original price array is at: lookback + train_end - 1
    train_price_cutoff = lookback + train_end  # exclusive index for prices used to scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(prices[:train_price_cutoff])

    # 4) Transform all prices with the train-fitted scaler
    scaled_prices = scaler.transform(prices)

    # 5) Build sequences across the entire scaled series
    X, y = [], []
    for i in range(lookback, len(scaled_prices)):
        X.append(scaled_prices[i - lookback:i, 0])
        y.append(scaled_prices[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 6) Chronological split (no shuffle)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Safety checks
    if len(X_val) == 0:  # fallback to small validation if rounding created empty val
        split_fix = max(1, int(0.1 * n_samples))
        X_train, y_train = X[:-split_fix], y[:-split_fix]
        X_val, y_val = X[-split_fix:-1], y[-split_fix:-1]
        X_test, y_test = X[-1:], y[-1:]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, ticker_data