"""
Phase 1, Day 3: The Classical LSTM Baseline
Project: Q-Alpha (Hybrid Quantum-Classical Gold Price Prediction)
Description: Builds, trains, and evaluates a PyTorch LSTM model to establish
a baseline Root Mean Square Error (RMSE) using historical gold prices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from day2_sequencing import prepare_pytorch_data

# ---------------------------------------------------------
# 1. Model Architecture
# ---------------------------------------------------------
class ClassicalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        """
        Initializes the classical LSTM baseline model.
        """
        super(ClassicalLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # The LSTM layer: batch_first=True ensures it expects tensors of shape (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        
        # The final linear layer that outputs our single predicted price
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        """
        Defines the forward pass of the neural network.
        """
        # Pass the sequence through the LSTM
        lstm_out, _ = self.lstm(input_seq)
        
        # We only care about the output from the final day of the 30-day window
        # lstm_out shape is (batch_size, sequence_length, hidden_layer_size)
        # lstm_out[:, -1, :] grabs the very last time step for every batch
        last_time_step_out = lstm_out[:, -1, :]
        
        # Pass the final time step through the linear layer to get the prediction
        predictions = self.linear(last_time_step_out)
        return predictions

# ---------------------------------------------------------
# 2. Main Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[*] Loading and formatting data from Day 2 pipeline...")
    # Load tensors and the scaler (needed to reverse the 0-1 scaling later)
    X_tensor, y_tensor, scaler = prepare_pytorch_data(window_size=30)
    
    # --- Data Splitting ---
    print("\n[*] Splitting data chronologically (80% Train / 20% Test)...")
    split_index = int(len(X_tensor) * 0.8)
    
    # CRUCIAL: Time-series data must NOT be shuffled. 
    # We train on the past to predict the future.
    X_train, X_test = X_tensor[:split_index], X_tensor[split_index:]
    y_train, y_test = y_tensor[:split_index], y_tensor[split_index:]
    
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    # --- Training Setup ---
    model = ClassicalLSTM(input_size=1, hidden_layer_size=50, output_size=1)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    
    print("\n[*] Initializing Classical LSTM Training Loop...")
    model.train() # Set model to training mode
    
    for epoch in range(epochs):
        # 1. Clear out the gradients from the previous epoch
        optimizer.zero_grad()
        
        # 2. Forward pass: generate predictions for the training set
        y_pred = model(X_train)
        
        # 3. Calculate the error (loss)
        loss = loss_function(y_pred, y_train)
        
        # 4. Backward pass: calculate gradients
        loss.backward()
        
        # 5. Update the weights
        optimizer.step()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3}/{epochs} | Loss (MSE): {loss.item():.6f}")
            
    print("[*] Training Complete.")
    
    # --- Evaluation & Benchmarking ---
    print("\n[*] Evaluating model on unseen test data...")
    model.eval() # Set model to evaluation mode
    
    with torch.no_grad(): # Disable gradient tracking to save memory/compute
        test_predictions = model(X_test)
        
    # Detach tensors and convert back to NumPy arrays
    test_predictions_np = test_predictions.numpy()
    y_test_np = y_test.numpy()
    
    # --- Real-World Metrics ---
    # Convert the 0-1 scaled numbers back into real-world dollar amounts
    real_predictions = scaler.inverse_transform(test_predictions_np)
    real_targets = scaler.inverse_transform(y_test_np)
    
    # Calculate Root Mean Square Error (RMSE) in actual dollars
    rmse = np.sqrt(np.mean((real_predictions - real_targets) ** 2))
    
    print("-" * 40)
    print(f"🎯 BASELINE CLASSICAL RMSE: ${rmse:.2f}")
    print("-" * 40)
    print("This is the benchmark your Quantum model needs to beat in Phase 2!")