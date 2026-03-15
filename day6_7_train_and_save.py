import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from day5_hybrid_model import QAlphaHybrid
import math

# 1. Load the new Phase 4 data (Price + RSI)
print("[*] Loading Advanced Phase 4 Data (v2)...")
X = np.load('X_v2.npy')
y = np.load('y_v2.npy')

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split data chronologically (80% Train / 20% Test)
split_idx = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# 2. Initialize the upgraded model (input_size=2)
print("\n[*] Booting up the V2 Q-Alpha Architecture...")
model = QAlphaHybrid(input_size=2, hidden_size=50, n_qubits=4, n_q_layers=2, output_size=1)

# Loss function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Training Loop (Sticking to your golden 1,000 epochs)
epochs = 1000
print("\n[*] Initiating V2 Hybrid Training Loop...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:4d}/{epochs} | Loss (MSE): {loss.item():.6f}")

print("\n[*] V2 Training Complete.")

# 4. Evaluation
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    rmse = math.sqrt(test_loss.item())
    
print("-" * 50)
print(f"🌌 V2 HYBRID QUANTUM RMSE (Scaled): {rmse:.4f}")
print("-" * 50)

# 5. Save the upgraded brain!
torch.save(model.state_dict(), "q_alpha_model_v2.pth")
print("✅ SUCCESS: Model safely saved to disk as q_alpha_model_v2.pth.")