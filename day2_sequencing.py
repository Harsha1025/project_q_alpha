"""
Phase 1, Day 2: Data Preprocessing & Sequencing
Project: Q-Alpha (Hybrid Quantum-Classical Gold Price Prediction)
Description: This script isolates the Close prices, scales the data for neural 
network consumption, creates 30-day sliding windows, and converts them into PyTorch tensors.
"""

import numpy as np
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from day1_ingestion import get_clean_gold_data

def create_sliding_windows(data, window_size=30):
    """
    Iterates through scaled data to create input sequences and target labels.
    
    Args:
        data (np.ndarray): The 2D scaled dataset.
        window_size (int): Number of days used to predict the next day.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays for X (inputs) and y (targets).
    """
    X, y = [], []
    # Loop through the data, stopping exactly when we run out of a full window + target
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)]) # 30 days of data
        y.append(data[i + window_size])     # The 31st day target
        
    return np.array(X), np.array(y)

def prepare_pytorch_data(window_size=30, scaler_filename="gold_scaler.save"):
    """
    Full pipeline to load, scale, sequence, and tensorize the gold data.
    """
    print("[*] Loading clean data from Day 1 pipeline...")
    df = get_clean_gold_data()
    
    # 1. Data Extraction: Isolate 'Close' and convert to 2D NumPy array
    # reshape(-1, 1) ensures it's a 2D column array, which MinMaxScaler requires
    close_prices = df['Close'].values.reshape(-1, 1)
    
    print("[*] Scaling data between 0 and 1...")
    # 2. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Save the scaler for Phase 3 (Inverse transforming live predictions)
    joblib.dump(scaler, scaler_filename)
    print(f"[*] Scaler successfully saved to '{scaler_filename}'.")
    
    print(f"[*] Creating sliding windows (Window Size = {window_size})...")
    # 3. Sliding Windows
    X_numpy, y_numpy = create_sliding_windows(scaled_data, window_size)
    
    print("[*] Converting to PyTorch Tensors...")
    # 4. PyTorch Formatting
    # X naturally becomes shape (batch_size, sequence_length, features)
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype=torch.float32)
    
    # 5. Verification
    print("\n--- Tensor Verification Summary ---")
    print(f"Input Features (X) Shape: {X_tensor.shape}")
    print(f"Target Labels (y) Shape:  {y_tensor.shape}")
    print(f"Expected X Shape:         (Total_Samples, {window_size}, 1)")
    print(f"Expected y Shape:         (Total_Samples, 1)")
    print("-----------------------------------")
    
    if len(X_tensor.shape) == 3 and X_tensor.shape[1] == window_size and X_tensor.shape[2] == 1:
        print("[*] SUCCESS: PyTorch tensors are correctly shaped for LSTM ingestion!")
    else:
        print("[!] WARNING: Tensor shapes are incorrect. Please verify the pipeline.")

    return X_tensor, y_tensor, scaler

if __name__ == "__main__":
    # Execute the pipeline to verify the data transforms correctly
    X, y, scaler = prepare_pytorch_data(window_size=30)