# 📈 Stock Price Prediction Dashboard

A comprehensive stock market analysis and prediction dashboard built with Streamlit and LSTM neural networks.

## Features

### 📊 Data Visualization
- **Real-time Stock Data**: Fetch live stock data using yfinance
- **Interactive Charts**: Price charts, volume, normalized performance, and daily returns
- **Multiple Stocks**: Compare up to 5 stocks simultaneously (AAPL, MSFT, GOOGL, AMZN, TSLA)
- **Customizable Date Range**: Select any historical date range
- **Log Scale**: Toggle logarithmic scale for better percentage comparison

### 🤖 LSTM Price Predictions
- **Deep Learning Model**: 2-layer LSTM neural network for time series forecasting
- **Multi-Stock Support**: Separate trained models for each stock
- **Future Predictions**: Predict stock prices up to 90 days ahead
- **Confidence Intervals**: Optional ±5% confidence bands
- **Visual Comparison**: Side-by-side historical vs predicted prices
- **Export Data**: Download predictions as CSV

### 📈 Key Metrics
- Current stock prices
- Daily price changes and percentages
- Predicted prices (7-day, 30-day, custom)
- Price change projections

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data**: yfinance, pandas
- **Visualization**: Plotly Express
- **Machine Learning**: TensorFlow/Keras (LSTM)
- **Data Processing**: scikit-learn, NumPy

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
pip install streamlit yfinance plotly pandas tensorflow scikit-learn
```

## 🎯 Project Structure

```
datascience_project/
├── dashboard.py              # Main Streamlit dashboard
├── model_trainer.py          # LSTM model training script
├── model_predictor.py        # Prediction logic
├── utils/
│   ├── __init__.py
│   └── preprocessing.py      # Data preprocessing functions
├── models/                   # Trained LSTM models
│   ├── lstm_AAPL.h5
│   ├── lstm_MSFT.h5
│   ├── lstm_GOOGL.h5
│   ├── lstm_AMZN.h5
│   ├── lstm_TSLA.h5
│   ├── scaler_AAPL.pkl
│   └── ...
└── README.md                 # This file
```

## 🏃 Usage

### Run the Dashboard
```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Train New Models
To train LSTM models for all stocks:
```bash
python model_trainer.py
```

Training parameters:
- **Lookback**: 60 days
- **Epochs**: 50
- **Batch Size**: 32
- **Split**: Chronological 70/10/20 (Train/Val/Test)

### Data Splitting and Leakage Prevention
- The split is strictly chronological (no shuffling) to respect time order.
- A MinMax scaler is fit only on the training window, then applied to val/test to avoid data leakage.
- The model trains with a validation set and EarlyStopping monitors `val_loss`.


## 📝 Model Performance

All models trained for 50 epochs with chronological train/validation/test split (835/119/240 samples).

### Performance Summary:

| Stock | Training Loss | Test Loss | Status |
|-------|--------------|-----------|--------|
| TSLA  | 0.001165 | 0.002344 | ✅ Excellent |
| GOOGL | 0.001195 | 0.003398 | ✅ Excellent |
| AMZN  | 0.001769 | 0.003063 | ✓ Good |
| MSFT  | 0.001444 | 0.003111 | ✓ Good |
| AAPL  | 0.003187 | 0.008541 | ⚠️ Fair |

**Notes**: 
- Loss values are MSE (Mean Squared Error) on normalized data
- Lower values indicate better predictions
- Test/Train ratios range from 1.7x to 2.8x (acceptable for time series)
- Models show reasonable generalization with minimal overfitting

## 🤝 Contributing

This is an educational project. Feel free to fork and extend it!

## 📄 License

MIT License - Feel free to use for learning and education.

# StockPricePrediction
