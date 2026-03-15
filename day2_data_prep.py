import yfinance as yf
import pandas as pd
import pandas_ta as ta  
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

def prepare_advanced_data(ticker="GC=F"):
    print(f"[*] Fetching data for {ticker}...")
    data = yf.download(ticker, start="2016-01-01")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Calculate 14-Day RSI
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data = data.dropna()
    
    # Extract TWO features (Price and RSI)
    features = data[['Close', 'RSI']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    joblib.dump(scaler, 'gold_scaler_v2.save')
    print("[*] Scaler v2 (Price + RSI) successfully saved.")
    
    return scaled_data

def create_advanced_sequences(data, window=30):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, :]) 
        y.append(data[i, 0])          
    return np.array(X), np.array(y)

if __name__ == "__main__":
    scaled_data = prepare_advanced_data()
    X, y = create_advanced_sequences(scaled_data)
    
    np.save('X_v2.npy', X)
    np.save('y_v2.npy', y)
    
    print("-" * 40)
    print(f"[*] SUCCESS! Data Pipeline Upgraded.")
    print(f"[*] Input Tensor (X) Shape: {X.shape}")
    print("-" * 40)