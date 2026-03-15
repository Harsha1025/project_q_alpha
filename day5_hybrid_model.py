import torch
import torch.nn as nn
from day4_quantum_layer import create_quantum_layer

class QAlphaHybrid(nn.Module):
    # UPGRADE: input_size is now 2 (Price + RSI)
    def __init__(self, input_size=2, hidden_size=50, n_qubits=4, n_q_layers=2, output_size=1):
        super(QAlphaHybrid, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.bottleneck = nn.Linear(hidden_size, n_qubits)
        self.quantum_layer = create_quantum_layer(n_qubits=n_qubits, n_layers=n_q_layers)
        
        # This layer predicts the DELTA (the tiny daily change)
        self.final_linear = nn.Linear(n_qubits, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        bottleneck_out = self.bottleneck(last_step)
        quantum_out = self.quantum_layer(bottleneck_out)
        
        # 1. The quantum circuit predicts the tiny change (Delta)
        quantum_delta = self.final_linear(quantum_out)
        
        # 2. SKIP CONNECTION UPGRADE: 
        # x is shaped (Batch, 30 Days, 2 Features). 
        # We only want Feature 0 (Price) from the very last day (-1) to anchor our prediction.
        todays_price = x[:, -1, 0].unsqueeze(1) 
        
        # 3. Final Prediction = Today's Price + The Quantum Delta
        final_out = todays_price + quantum_delta
        
        return final_out