
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scfn import PolynomialStructuralCFN

def generate_synthetic_data(n_samples=2000):
    """
    Generate synthetic data with known interactions.
    y = x0^2 + sin(x1 * x2) + noise
    interaction: x1 <-> x2
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, 5) # 5 Features
    
    # Interaction: Feature 1 and Feature 2 are coupled
    # Feature 0 is independent
    # Features 3, 4 are noise
    
    y = (X[:, 0]**2) + np.sin(3 * X[:, 1] * X[:, 2]) + 0.1 * np.random.randn(n_samples)
    
    # Scale
    y = (y - y.mean()) / y.std()
    
    return X, y

def validate_interaction_recovery():
    print(">>> Running Synthetic Interaction Validation...")
    X, y = generate_synthetic_data()
    
    # Train StructuralCFN
    model = PolynomialStructuralCFN(input_dim=5, poly_degree=2, context_activation='tanh')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    print("Training...")
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = torch.nn.functional.mse_loss(pred, y_t)
        
        # L1 penalty for sparsity
        l1_loss = 0
        for node in model.dependency_layer.function_nodes:
             l1_loss += torch.norm(node.poly_node.direction, 1)
        
        total_loss = loss + 0.001 * l1_loss
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
    # Get Dependency Matrix
    print("Extracting Dependency Matrix...")
    dep_matrix = model.get_dependency_matrix()
    
    # Verify: M[1,2] and M[2,1] should be high
    target_interaction = dep_matrix[1, 2] + dep_matrix[2, 1]
    noise_interaction = dep_matrix[3, 4]  # Should be low
    
    print("\n--- Validation Results ---")
    print(f"Recovered Interaction Strength (x1-x2): {target_interaction:.4f}")
    print(f"Noise Interaction Strength (x3-x4):     {noise_interaction:.4f}")
    
    success = target_interaction > 2 * noise_interaction
    print(f"SUCCESS: {success}")
    
    # Save Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(dep_matrix, annot=True, fmt=".2f", cmap='viridis')
    plt.title("Recovered Interaction Matrix\n(Ground Truth: x1-x2 coupled)")
    plt.savefig("benchmarks/validation_heatmap.png")
    print("Heatmap saved to benchmarks/validation_heatmap.png")

if __name__ == "__main__":
    validate_interaction_recovery()
