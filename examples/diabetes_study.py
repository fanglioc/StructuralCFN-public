import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add path so it finds structural_cfn local folder
from scfn import GenericStructuralCFN

def run_benchmark():
    print("--- REPRODUCTION BENCHMARK: StructuralCFN (Diabetes) ---")
    data = load_diabetes()
    X, y = data.data, data.target.reshape(-1, 1)
    feature_names = data.feature_names
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    # We use fold 1 for visualization
    viz_fold = 0
    viz_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Training Fold {fold+1}/5...")
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]
        
        sx, sy = StandardScaler(), StandardScaler()
        X_train = torch.tensor(sx.fit_transform(X_train_raw), dtype=torch.float32)
        y_train = torch.tensor(sy.fit_transform(y_train_raw), dtype=torch.float32)
        X_val = torch.tensor(sx.transform(X_val_raw), dtype=torch.float32)
        y_val = torch.tensor(sy.transform(y_val_raw), dtype=torch.float32)
        
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        v_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

        model = GenericStructuralCFN(input_dim=X.shape[1], poly_degree=2)
        opt = optim.Adam(model.parameters(), lr=0.01)
        crit = nn.MSELoss()
        
        # Training loop
        for epoch in range(150):
            model.train()
            for bx, by in loader:
                opt.zero_grad()
                crit(model(bx), by).backward()
                opt.step()
        
        model.eval()
        with torch.no_grad():
            fold_mse = 0
            for bx, by in v_loader:
                fold_mse += crit(model(bx), by).item()
            scores.append(fold_mse / len(v_loader))
        
        if fold == viz_fold:
            viz_model = model

    print("\n" + "="*40)
    print(f"Final Mean MSE (Scaled): {np.mean(scores):.4f} (+/- {np.std(scores):.3f})")
    print(f"Total Parameters: {sum(p.numel() for p in viz_model.parameters() if p.requires_grad)}")
    print("="*40)

    # Visualization
    print("\nSaving Dependency Map to 'benchmarks/diabetes_heatmap.png'...")
    dep_matrix = viz_model.get_dependency_matrix()
    plt.figure(figsize=(10, 8))
    sns.heatmap(dep_matrix, annot=True, xticklabels=feature_names, yticklabels=feature_names, cmap="Blues")
    plt.title("StructuralCFN Learned Interactions (Diabetes)")
    plt.savefig(os.path.join(os.path.dirname(__file__), 'diabetes_heatmap.png'))

if __name__ == "__main__":
    run_benchmark()
