import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes, fetch_california_housing, load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scfn import PolynomialStructuralCFN, HighRankPolynomialStructuralCFN
from benchmarks.dataset_loaders import load_wine_quality, load_heart_disease, load_ionosphere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def train_and_eval(model, train_loader, val_loader, is_classification=False, epochs=200, l1_lambda=0.0001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss() if is_classification else nn.MSELoss()
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            # Synchronize L1 penalty with main benchmarks
            if hasattr(model, 'dependency_layer') and l1_lambda > 0:
                l1_penalty = 0
                for node in model.dependency_layer.function_nodes:
                    l1_penalty += torch.norm(node.poly_node.direction, 1)
                    l1_penalty += torch.norm(node.sin_node.direction, 1)
                loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break
    return best_val_loss

def run_ablation(dataset_name="Diabetes"):
    loader_map = {
        "Diabetes": (load_diabetes, False),
        "CA Housing": (fetch_california_housing, False),
        "Breast Cancer": (load_breast_cancer, True),
        "Wine Quality": (load_wine_quality, True),
        "Heart Disease": (load_heart_disease, True),
        "Ionosphere": (load_ionosphere, True)
    }
    
    if dataset_name not in loader_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    load_fn, is_classification = loader_map[dataset_name]
    data = load_fn()
    X, y = data.data, data.target
    if not is_classification:
        y = y.reshape(-1, 1)
    
    # Subsampling logic locked with Seed 42 for consistency
    if dataset_name == "CA Housing" or len(X) > 5000:
        set_seed(42) # Lock seed before sampling
        limit = 5000 if dataset_name == "CA Housing" else 3000
        indices = np.random.choice(len(X), min(len(X), limit), replace=False)
        X, y = X[indices], y[indices]
        
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) if is_classification else KFold(n_splits=10, shuffle=True, random_state=42)
    
    configs = [
        ("Champion (Sigmoid + Hybrid)", "sigmoid", 2, True),
        ("Tanh-Gated (Polarity Mode)", "tanh", 2, True),
        ("Basis: Polynomial-Only", "sigmoid", 2, False), # Set sin importance to 0 in future if needed, for now just change init or code
        ("Basis: Sinusoidal-Only", "sigmoid", 0, True),  # Degree 0 Poly is basically a constant
        ("Open-Interaction (Linear Mode)", None, 2, True),
    ]
    
    results = {c[0]: [] for c in configs}
    
    print(f"\n>>> Running Global Ablation: {dataset_name} (10-fold CV / Seed 42)")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y if is_classification else np.zeros(len(y)))):
        print(f"Fold {fold+1}/10...", end='\r')
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]
        
        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train_raw)
        X_val_scaled = scaler_x.transform(X_val_raw)
        
        if not is_classification:
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train_raw)
            y_val_scaled = scaler_y.transform(y_val_raw)
        else:
            y_train_scaled, y_val_scaled = y_train_raw, y_val_raw
        
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1)
        
        b_size = 512 if dataset_name == "CA Housing" else 64
        t_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=b_size, shuffle=True)
        v_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=b_size, shuffle=False)
        
        for name, act, poly_deg, use_sin in configs:
            set_seed(42)
            # Use HighRank for Housing, Standard for others
            if dataset_name == "CA Housing":
                model = HighRankPolynomialStructuralCFN(input_dim=X.shape[1], context_activation=act, 
                                                      poly_degree=poly_deg, classification=is_classification)
            else:
                model = PolynomialStructuralCFN(input_dim=X.shape[1], context_activation=act, 
                                               poly_degree=poly_deg, classification=is_classification)
            
            # Basis Ablation Logic: zero out weights if needed or use specialized config
            # For this quick ablation script, we modify the nodes directly if name indicates single basis
            if "Polynomial-Only" in name:
                for node in model.dependency_layer.function_nodes:
                    node.combine.weight.data[0, 1] = 0 # Zero out sin
                    node.combine.weight.requires_grad = False # Freeze to keep it zero
            elif "Sinusoidal-Only" in name:
                for node in model.dependency_layer.function_nodes:
                    node.combine.weight.data[0, 0] = 0 # Zero out poly
                    node.combine.weight.requires_grad = False
                
            loss_fn = nn.BCELoss() if is_classification else nn.MSELoss()
            score = train_and_eval(model, t_loader, v_loader, is_classification)
            results[name].append(score)
            
    metric_name = "LogLoss" if is_classification else "MSE"
    print("\n\n" + "="*50)
    print(f"{'Config':<40} | {metric_name} (Mean±Std)")
    print("-" * 50)
    for name, scores in results.items():
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"{name:<40} | {mean:.4f}±{std:.4f}")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Diabetes")
    args = parser.parse_args()
    
    # Stratified KFold needs this for classification
    from sklearn.model_selection import StratifiedKFold
    
    run_ablation(args.dataset)
