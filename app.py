"""
Phase 4, Day 13: The Quantitative Backtester
Project: Q-Alpha (Hybrid Quantum-Classical Gold Price Prediction)
Description: Professional UI with Volatility Cone, RSI Momentum, 
V2 AI Inference, and a live 30-Day Directional Accuracy Backtester.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
import pandas_ta as ta

# Import your custom Hybrid Architecture
from day5_hybrid_model import QAlphaHybrid

# ---------------------------------------------------------
# 1. UI Skeleton & Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Q-Alpha: Quantum Gold Oracle", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

CURRENCIES = {
    "USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", 
    "JPY": "¥", "AUD": "A$", "CAD": "C$", "CHF": "CHF", "CNY": "¥"
}
UNIT_MULTIPLIERS = {"Troy Ounces": 1.0, "Grams": 0.0321507, "Kilograms": 32.1507466}

# ---------------------------------------------------------
# 2. Caching the V2 AI Brain & Exchange Rates
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    scaler = joblib.load('gold_scaler_v2.save')
    model = QAlphaHybrid(input_size=2, hidden_size=50, n_qubits=4, n_q_layers=2, output_size=1)
    model.load_state_dict(torch.load('q_alpha_model_v2.pth', weights_only=True))
    model.eval()
    return model, scaler

@st.cache_data(ttl=3600)
def fetch_live_data(ticker="GC=F", days=90): # DAY 13: Increased to 90 days for backtesting!
    data = yf.download(ticker, period=f"{days}d", interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])
    return data

@st.cache_data(ttl=3600)
def get_exchange_rate(target_currency):
    if target_currency == "USD": return 1.0
    try:
        forex_data = yf.download(f"USD{target_currency}=X", period="1d", interval="1d")
        return float(np.squeeze(forex_data['Close'].values[-1]))
    except:
        return 1.0

# ---------------------------------------------------------
# 3. Sidebar Settings
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d7/Gold_bullion_bars.jpg", width="stretch")
    st.header("⚙️ Oracle Settings")
    st.markdown("Adjust parameters for precise valuations.")
    
    selected_currency = st.selectbox("🌍 Currency", options=list(CURRENCIES.keys()), index=0)
    gold_unit = st.selectbox("⚖️ Measurement Unit", options=list(UNIT_MULTIPLIERS.keys()), index=0)
    gold_quantity = st.number_input("🔢 Quantity", min_value=0.01, value=1.00, step=0.10)
    
exchange_rate = get_exchange_rate(selected_currency)
symbol = CURRENCIES[selected_currency]
unit_factor = UNIT_MULTIPLIERS[gold_unit]
grand_multiplier = exchange_rate * unit_factor * gold_quantity

# ---------------------------------------------------------
# 4. Main Page Header & Global Data Prep
# ---------------------------------------------------------
st.title("🔮 Q-Alpha: Quantum Gold Oracle")
st.markdown("Leveraging Hybrid Quantum-Classical Machine Learning to forecast commodities.")
st.divider()

with st.spinner("Connecting to live data and waking up V2 AI..."):
    live_gold_data = fetch_live_data(ticker="GC=F", days=90)
    model, scaler = load_model_and_scaler()

if not live_gold_data.empty:
    # Feature Engineering
    live_gold_data['RSI'] = ta.rsi(live_gold_data['Close'], length=14)
    live_gold_data = live_gold_data.dropna()
    
    # --- DAY 13: THE BACKTESTER ENGINE ---
    def run_backtest(df, mdl, scl, backtest_days=30):
        features = df[['Close', 'RSI']].values
        if len(features) < backtest_days + 30:
            return None # Not enough data
        
        correct_directions = 0
        for i in range(backtest_days):
            target_idx = len(features) - backtest_days + i
            window = features[target_idx - 30 : target_idx]
            
            prev_price = window[-1, 0]
            actual_price = features[target_idx, 0]
            
            # Predict
            scaled_window = scl.transform(window)
            X_tensor = torch.tensor(scaled_window, dtype=torch.float32).view(1, 30, 2)
            with torch.no_grad():
                scaled_pred = mdl(X_tensor)
                
            dummy_pad = np.zeros((1, 2))
            dummy_pad[0, 0] = scaled_pred.numpy()[0][0]
            predicted_price = float(scl.inverse_transform(dummy_pad)[0, 0])
            
            # Did it guess the correct direction?
            actual_dir = 1 if actual_price > prev_price else -1
            pred_dir = 1 if predicted_price > prev_price else -1
            
            if actual_dir == pred_dir:
                correct_directions += 1
                
        return (correct_directions / backtest_days) * 100

    win_rate = run_backtest(live_gold_data, model, scaler, backtest_days=30)
    
    # Inject the Win Rate into the Sidebar
    with st.sidebar:
        st.divider()
        if win_rate is not None:
            st.metric("🎯 30-Day Directional Accuracy", f"{win_rate:.1f}%")
        st.caption("🧠 **AI Engine:** V2 Multivariate Quantum Hybrid")
        st.caption("📡 **Data Feed:** COMEX Gold Futures")

# ---------------------------------------------------------
# 5. Execution, Prediction & Volatility Logic
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1], gap="large") 

if not live_gold_data.empty:
    with col1:
        st.subheader("📊 Live Market Data")
        st.info(f"ℹ️ **Valuation for:** {gold_quantity} {gold_unit} in {selected_currency}")
        
        display_df = live_gold_data.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in display_df.columns:
                display_df[col] = display_df[col] * grand_multiplier
        
        st.markdown(f"**Latest Trading History ({selected_currency}):**")
        st.dataframe(display_df.tail(5), width='stretch')

    with col2:
        st.subheader("🤖 Quantum Oracle")
        
        last_30_features = live_gold_data[['Close', 'RSI']].values[-30:]
        scaled_30_days = scaler.transform(last_30_features)
        X_live_tensor = torch.tensor(scaled_30_days, dtype=torch.float32).view(1, 30, 2)
        
        with torch.no_grad(): 
            scaled_prediction = model(X_live_tensor)
        
        dummy_pad = np.zeros((1, 2))
        dummy_pad[0, 0] = scaled_prediction.numpy()[0][0]
        
        predicted_usd = float(scaler.inverse_transform(dummy_pad)[0, 0])
        current_usd = float(last_30_features[-1, 0])
        
        val_now = current_usd * grand_multiplier
        val_pred = predicted_usd * grand_multiplier
        diff = val_pred - val_now
        
        recent_prices = display_df['Close'].values[-7:]
        std_dev = np.std(recent_prices)
        upper_bound = val_pred + (std_dev * 1.5)
        lower_bound = val_pred - (std_dev * 1.5)
        
        st.markdown("### Market Snapshot")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Current Price", f"{symbol}{val_now:,.2f}")
        m_col2.metric("Predicted (24h)", f"{symbol}{val_pred:,.2f}", delta=f"{diff:,.2f}")
        
        if val_pred > val_now:
            st.success("🟢 **SIGNAL: BUY (Bullish)**")
        else:
            st.error("🔴 **SIGNAL: SELL (Bearish)**")
            
        st.caption(f"🎯 **Expected Range:** {symbol}{lower_bound:,.2f} - {symbol}{upper_bound:,.2f}")

# ---------------------------------------------------------
# 6. Interactive Plotly Visualization with Shadow Cone
# ---------------------------------------------------------
    st.divider()
    st.subheader("📈 Quantum Forecast with Volatility Cone")
    
    hist_dates = display_df.index[-30:]
    hist_prices = display_df['Close'].values[-30:]
    tomorrow = hist_dates[-1] + pd.Timedelta(days=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices, mode='lines', name='Historical', line=dict(color='#00F0FF', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1]], y=[val_now], mode='markers', name='Today',
        marker=dict(color='white', size=8, line=dict(width=2, color='white'))
    ))
    
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], tomorrow], y=[val_now, upper_bound],
        mode='lines', showlegend=False, line=dict(color='rgba(255, 215, 0, 0)')
    ))
    
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], tomorrow], y=[val_now, lower_bound],
        mode='lines', name='Confidence Interval',
        fill='tonexty', fillcolor='rgba(255, 215, 0, 0.2)', line=dict(color='rgba(255, 215, 0, 0)') 
    ))
    
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], tomorrow], y=[val_now, val_pred],
        mode='lines+markers', name='AI Forecast',
        line=dict(color='#FFD700', width=3, dash='dash'),
        marker=dict(size=12, symbol='star')
    ))
    
    fig.update_layout(
        template='plotly_dark', hovermode="x unified", yaxis_title=f"Price ({selected_currency})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Failed to fetch data.")