import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import date, timedelta
import os
import json

# Import prediction module
try:
    from model_predictor import predict_next_days
    MODELS_AVAILABLE = True
except:
    MODELS_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="Stock Dashboard", layout="wide", page_icon="📈")

# Title
st.title("📈 Stock Data Dashboard")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    # Stock selection
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    selected_tickers = st.multiselect(
        "Select Stock Tickers",
        options=default_tickers,
        default=["AAPL", "MSFT", "GOOGL"]
    )
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today()
        )
    
    # Chart options
    st.subheader("Chart Options")
    log_scale = st.checkbox("Use Log Scale", value=False)
    show_volume = st.checkbox("Show Volume Chart", value=True)

# Function to load stock data
@st.cache_data(show_spinner=False)
def load_stock_data(tickers, start, end):
    """Load stock data for multiple tickers"""
    all_data = []
    
    for ticker in tickers:
        try:
            # Download stock data
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

            if not df.empty:
                # Reset index and flatten any multi-level columns
                df = df.reset_index()
                
                # Handle multi-level columns if they exist
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
                
                df["Ticker"] = ticker
                
                # Select only the columns we need
                cols_to_keep = []
                for col_name in ["Date", "Open", "High", "Low", "Close", "Volume"]:
                    matching_cols = [col for col in df.columns if col_name in col]
                    if matching_cols:
                        df[col_name] = df[matching_cols[0]]
                        cols_to_keep.append(col_name)
                
                cols_to_keep.append("Ticker")
                all_data.append(df[cols_to_keep])
        except Exception as e:
            st.warning(f"Could not load data for {ticker}: {str(e)}")
            
    print("All data", all_data)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# Main content
if not selected_tickers:
    st.info("👈 Please select at least one stock ticker from the sidebar to get started.")
    st.stop()

# Load data
with st.spinner("Loading stock data..."):
    stock_data = load_stock_data(selected_tickers, start_date, end_date)

if stock_data.empty:
    st.error("No data available for the selected tickers and date range.")
    st.stop()

# Display key metrics
st.subheader("📊 Key Metrics")
metrics_cols = st.columns(len(selected_tickers))

for idx, ticker in enumerate(selected_tickers):
    ticker_data = stock_data[stock_data["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    
    if not ticker_data.empty and len(ticker_data) > 0:
        # Get the last row
        last_row = ticker_data.iloc[-1]
        latest_price = last_row["Close"]
        
        # Get previous price if available
        if len(ticker_data) > 1:
            prev_row = ticker_data.iloc[-2]
            previous_price = prev_row["Close"]
        else:
            previous_price = latest_price
        
        price_change = latest_price - previous_price
        price_change_pct = (price_change / previous_price * 100) if previous_price != 0 else 0
        
        with metrics_cols[idx]:
            st.metric(
                label=ticker,
                value=f"${latest_price:.2f}",
                delta=f"{price_change_pct:+.2f}%"
            )

st.markdown("---")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Normalized Performance", "📉 Daily Returns", "🔮 LSTM Predictions"])

with tab1:
    st.subheader("Stock Price Over Time")
    fig_price = px.line(
        stock_data,
        x="Date",
        y="Close",
        color="Ticker",
        title="Closing Price",
        labels={"Close": "Price ($)", "Date": "Date"},
        template="plotly_white"
    )
    
    if log_scale:
        fig_price.update_yaxes(type="log")
    
    fig_price.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    if show_volume:
        st.subheader("Trading Volume")
        fig_volume = px.bar(
            stock_data,
            x="Date",
            y="Volume",
            color="Ticker",
            title="Volume",
            labels={"Volume": "Volume", "Date": "Date"},
            template="plotly_white",
            opacity=0.7
        )
        
        fig_volume.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)

with tab2:
    st.subheader("Normalized Performance (Indexed to 100)")
    
    # Normalize prices to start at 100
    normalized_data = stock_data.sort_values("Date").copy()
    normalized_data["Normalized"] = normalized_data.groupby("Ticker")["Close"].transform(
        lambda x: 100 * x / x.iloc[0]
    )
    
    fig_normalized = px.line(
        normalized_data,
        x="Date",
        y="Normalized",
        color="Ticker",
        title="Normalized Price Performance",
        labels={"Normalized": "Indexed Value (Base = 100)", "Date": "Date"},
        template="plotly_white"
    )
    
    fig_normalized.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_normalized, use_container_width=True)

with tab3:
    st.subheader("Daily Returns")
    
    # Calculate daily returns
    returns_data = stock_data.sort_values("Date").copy()
    returns_data["Daily_Return"] = returns_data.groupby("Ticker")["Close"].pct_change() * 100
    returns_data = returns_data.dropna(subset=["Daily_Return"])
    
    fig_returns = px.line(
        returns_data,
        x="Date",
        y="Daily_Return",
        color="Ticker",
        title="Daily Returns (%)",
        labels={"Daily_Return": "Return (%)", "Date": "Date"},
        template="plotly_white"
    )
    
    fig_returns.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_returns, use_container_width=True)

with tab4:
    st.subheader("🔮 LSTM Price Predictions")
    
    if not MODELS_AVAILABLE:
        st.error("❌ Prediction module not available. Please check model_predictor.py")
        st.stop()
    
    # Check which selected tickers have trained models
    available_models = []
    for ticker in selected_tickers:
        model_path = f"models/lstm_{ticker}.h5"
        if os.path.exists(model_path):
            available_models.append(ticker)
    
    if not available_models:
        st.warning("⚠️ No trained models found for the selected stocks.")
        st.info("🤖 Train models by running: `python model_trainer.py`")
        st.code("python model_trainer.py", language="bash")
        
        # Show which models are missing
        st.info(f"ℹ️ Selected stocks without trained models: {', '.join(selected_tickers)}")
    else:
        st.success(f"✅ Models available for: {', '.join(available_models)}")
        
        # Show info if some selected stocks don't have models
        missing_models = [t for t in selected_tickers if t not in available_models]
        if missing_models:
            st.info(f"ℹ️ Selected stocks without trained models: {', '.join(missing_models)}")
        
        # Ticker selection for prediction (only from selected tickers with models)
        pred_ticker = st.selectbox(
            "Select Stock for Prediction",
            options=available_models,
            index=0
        )
        
        # Load and display model metrics
        import json
        metrics_path = f"models/metrics_{pred_ticker}.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Display metrics in an expander
            with st.expander("📊 Model Performance Metrics", expanded=False):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Training Loss",
                        f"{metrics['train_loss']:.6f}",
                        help="Lower is better. Measures error on training data."
                    )
                with col2:
                    st.metric(
                        "Validation Loss",
                        f"{metrics.get('val_loss', metrics.get('test_loss', float('nan'))):.6f}",
                        help="Lower is better. Measures error on unseen validation data."
                    )
                with col3:
                    st.metric(
                        "Testing Loss",
                        f"{metrics.get('test_loss', float('nan')):.6f}",
                        help="Lower is better. Measures error on the final held-out test set."
                    )
                with col4:
                    st.metric(
                        "Epochs Trained",
                        metrics['epochs_trained'],
                        help="Number of training iterations completed."
                    )
                with col5:
                    # Calculate overfitting indicator
                    overfit_ratio = metrics['test_loss'] / metrics['train_loss']
                    
                    # More nuanced status assessment
                    if overfit_ratio < 2.0:
                        overfit_status = "Excellent"
                        status_color = "normal"
                    elif overfit_ratio < 2.5:
                        overfit_status = "Good"
                        status_color = "normal"
                    elif overfit_ratio < 3.0:
                        overfit_status = "Fair"
                        status_color = "normal"
                    else:
                        overfit_status = "Overfitting"
                        status_color = "inverse"
                    
                    st.metric(
                        "Model Status",
                        overfit_status,
                        help=f"Test/Train ratio: {overfit_ratio:.2f}x\n\n"
                             f"< 2.0x = Excellent\n"
                             f"< 2.5x = Good\n"
                             f"< 3.0x = Fair\n"
                             f"≥ 3.0x = Overfitting"
                    )
                
                # Additional details
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Training Samples:** {metrics['train_samples']}")
                    st.markdown(f"**Validation Samples:** {metrics.get('val_samples', 'N/A')}")
                    st.markdown(f"**Test Samples:** {metrics['test_samples']}")
                with col2:
                    st.markdown(f"**Final Training Loss:** {metrics['final_train_loss']:.6f}")
                    st.markdown(f"**Final Validation Loss:** {metrics['final_val_loss']:.6f}")
                    st.markdown(f"**Test/Train Ratio:** {overfit_ratio:.2f}x")
                
                
        else:
            st.warning("⚠️ Model metrics not found. Retrain the model to generate metrics.")
        
        # Prediction days slider
        col1, col2 = st.columns([2, 1])
        with col1:
            prediction_days = st.slider(
                "Days to Predict",
                min_value=7,
                max_value=90,
                value=30,
                help="Number of days to predict into the future"
            )
        with col2:
            show_confidence = st.checkbox("Show Confidence Interval", value=False)
        
        # Make predictions
        try:
            with st.spinner(f"Predicting {pred_ticker} prices for next {prediction_days} days..."):
                # Since we're only showing selected tickers, data should already be loaded
                predictions = predict_next_days(pred_ticker, stock_data, days=prediction_days)
            
            # Create future dates
            last_date = stock_data[stock_data['Ticker'] == pred_ticker]['Date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
            
            # Historical data (last 100 days for context)
            historical = stock_data[stock_data['Ticker'] == pred_ticker].sort_values('Date').tail(100)
            
            # Prediction data
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Price': predictions,
                'Type': 'Predicted'
            })
            
            hist_df = historical[['Date', 'Close']].copy()
            hist_df.columns = ['Date', 'Price']
            hist_df['Type'] = 'Historical'
            
            # Combine historical and predicted
            combined = pd.concat([hist_df, pred_df], ignore_index=True)
            
            # Create the plot
            fig_pred = px.line(
                combined,
                x='Date',
                y='Price',
                color='Type',
                title=f"{pred_ticker} - Historical vs LSTM Predicted Prices",
                labels={'Price': 'Price ($)', 'Date': 'Date'},
                template='plotly_white',
                color_discrete_map={'Historical': '#1f77b4', 'Predicted': '#ff7f0e'}
            )
            
            # Add vertical line at prediction start
            fig_pred.add_vline(
                x=last_date.timestamp() * 1000,
                line_dash="dash",
                line_color="gray",
                annotation_text="Prediction Start"
            )
            
            # Add confidence interval if requested
            if show_confidence:
                # Simple confidence interval (±5% for demonstration)
                pred_df['Upper'] = predictions * 1.05
                pred_df['Lower'] = predictions * 0.95
                
                fig_pred.add_scatter(
                    x=pred_df['Date'],
                    y=pred_df['Upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                )
                fig_pred.add_scatter(
                    x=pred_df['Date'],
                    y=pred_df['Lower'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    fill='tonexty',
                    showlegend=True,
                    name='Confidence Interval (±5%)'
                )
            
            fig_pred.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Prediction summary
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = historical['Close'].iloc[-1]
            predicted_price = predictions[-1]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}"
                )
            with col2:
                st.metric(
                    f"Predicted ({prediction_days}d)",
                    f"${predicted_price:.2f}",
                    delta=f"{price_change_pct:+.2f}%"
                )
            with col3:
                st.metric(
                    "7-Day Prediction",
                    f"${predictions[6] if len(predictions) >= 7 else predictions[-1]:.2f}"
                )
            with col4:
                st.metric(
                    "30-Day Prediction",
                    f"${predictions[29] if len(predictions) >= 30 else predictions[-1]:.2f}"
                )
            
            # Prediction table
            with st.expander("📊 View Prediction Table"):
                pred_table = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': [f"${p:.2f}" for p in predictions],
                    'Days Ahead': range(1, prediction_days + 1)
                })
                st.dataframe(pred_table, use_container_width=True)
            
            # Download predictions
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"{pred_ticker}_predictions_{prediction_days}days.csv",
                mime="text/csv"
            )
            
            # Warning about predictions
            st.warning(
                "⚠️ **Disclaimer**: These predictions are based on historical patterns and should not be used "
                "for actual trading decisions. Stock markets are influenced by many factors not captured by this model."
            )
            
        except Exception as e:
            st.error(f"❌ Error making predictions: {str(e)}")
            st.info("Try retraining the model or check if the data format is correct.")

# Display raw data
st.markdown("---")
with st.expander("📋 View Raw Data"):
    st.dataframe(stock_data.sort_values(["Ticker", "Date"], ascending=[True, False]), use_container_width=True)
