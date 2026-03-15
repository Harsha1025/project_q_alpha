"""
Phase 2, Day 4: Intro to PennyLane & The Quantum Layer
Project: Q-Alpha (Hybrid Quantum-Classical Gold Price Prediction)
Description: Builds a Variational Quantum Circuit (VQC) and wraps it 
as a standard PyTorch neural network layer.
"""

import pennylane as qml
import torch
import torch.nn as nn

def create_quantum_layer(n_qubits=4, n_layers=2):
    """
    Creates a PyTorch-compatible Quantum Layer using PennyLane.
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of entangling layers (depth of the circuit).
        
    Returns:
        qml.qnn.TorchLayer: A PyTorch neural network module.
    """
    # 1. Quantum Device Setup: We use the default simulator
    dev = qml.device("default.qubit", wires=n_qubits)

    # 2. The Quantum Circuit (QNode)
    # The interface="torch" argument tells PennyLane to track gradients for PyTorch
    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        # A. Angle Embedding: Translates classical numbers into quantum angles (rotations)
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        
        # B. Strongly Entangling Layers: The "trainable" part of our quantum brain
        # This applies a series of rotations and CNOT gates to entangle the qubits
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        
        # C. Measurement: Collapse the quantum state and measure the Pauli-Z expected value
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
    # 3. PyTorch Wrapper Setup
    # StronglyEntanglingLayers expects weights in a very specific 3D shape:
    # (number_of_layers, number_of_qubits, 3 parameters per qubit for X,Y,Z rotations)
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    
    # Wrap the QNode into a PyTorch layer
    qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    return qlayer

# ---------------------------------------------------------
# Verification Block
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[*] Initializing PennyLane Quantum Circuit...")
    
    n_qubits = 4
    q_layer = create_quantum_layer(n_qubits=n_qubits, n_layers=2)
    
    print(f"[*] Quantum Layer successfully created with {n_qubits} qubits.")
    
    # --- Simulating the LSTM Output ---
    print("\n[*] Generating dummy classical data (simulating LSTM output)...")
    # Imagine our batch size is 16, and the LSTM outputs 4 features
    dummy_lstm_output = torch.rand((16, n_qubits))
    
    print(f"Classical Input Shape (Batch, Features): {dummy_lstm_output.shape}")
    
    # --- The Quantum Pass ---
    print("[*] Passing classical data through the Quantum Layer...")
    quantum_output = q_layer(dummy_lstm_output)
    
    print(f"Quantum Output Shape (Batch, Qubit_Observables): {quantum_output.shape}")
    print("\nFirst row of quantum output:")
    print(quantum_output[0].detach().numpy())
    
    # --- Verification ---
    print("-" * 50)
    if quantum_output.shape == (16, n_qubits):
        print("✅ SUCCESS: The Quantum Layer behaves exactly like a PyTorch module!")
        print("It is ready to be spliced into your classical architecture on Day 5.")
    else:
        print("❌ WARNING: Shape mismatch detected.")
    print("-" * 50)