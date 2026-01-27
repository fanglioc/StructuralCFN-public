import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes, fetch_california_housing, load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from scipy.stats import ttest_rel
import numpy as np
import xgboost as xgb
import optuna
import optuna
import warnings

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: lightgbm not found. Skipping LightGBM benchmarks.")

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try importing TabNet
try:
    from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False
    print("Warning: pytorch-tabnet not found. Skipping TabNet benchmarks.")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from scfn import GenericStructuralCFN, PolynomialStructuralCFN, HighRankPolynomialStructuralCFN
from benchmarks.dataset_loaders import load_wine_quality, load_heart_disease, load_ionosphere, load_bank_marketing

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)

# --- Models ---

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[32, 16], output_dim=1, classification=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.1)) # Removed for standard baseline match
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        if classification:
            layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Training Helper ---

def train_and_eval_pytorch(model, train_loader, val_loader, is_classification=False, epochs=200, lr=0.01, l1_lambda=0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss() if is_classification else nn.MSELoss()
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            # L1 for CFN
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

# --- Tuning Helpers ---

def tune_xgboost(X_train, y_train, X_val, y_val, is_classification):
    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'binary:logistic' if is_classification else 'reg:squarederror',
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        }
        
        if is_classification:
            model = xgb.XGBClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, preds)
        else:
            model = xgb.XGBRegressor(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) # Modest n_trials for speed
    return study.best_params

def tune_tabnet(X_train, y_train, X_val, y_val, is_classification):
    def objective(trial):
        params = {
            'n_d': trial.suggest_int('n_d', 8, 32),
            'n_a': trial.suggest_int('n_a', 8, 32),
            'n_steps': trial.suggest_int('n_steps', 3, 7),
            'gamma': trial.suggest_float('gamma', 1.0, 1.5),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-4, 1e-1, log=True),
            'optimizer_params': dict(lr=trial.suggest_float('lr', 1e-3, 1e-1, log=True)),
            'verbose': 0
        }
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        virtual_batch_size = trial.suggest_categorical('virtual_batch_size', [4, 8])
        
        if is_classification:
            model = TabNetClassifier(**params)
            # Log loss expects probabilities
            model.fit(X_train, y_train.flatten(), eval_set=[(X_val, y_val.flatten())], patience=10, max_epochs=30, batch_size=batch_size, virtual_batch_size=virtual_batch_size)
            preds = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, preds)
        else:
            model = TabNetRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=10, max_epochs=30, batch_size=batch_size, virtual_batch_size=virtual_batch_size)
            preds = model.predict(X_val)
            return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15) # Fast tuning
    study.optimize(objective, n_trials=15) # Fast tuning
    return study.best_params

def tune_lightgbm(X_train, y_train, X_val, y_val, is_classification):
    def objective(trial):
        param = {
            'verbosity': -1,
            'metric': 'binary_logloss' if is_classification else 'mse',
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        }
        
        if is_classification:
            model = lgb.LGBMClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=10)])
            preds = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, preds)
        else:
            model = lgb.LGBMRegressor(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=10)])
            preds = model.predict(X_val)
            return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    return study.best_params

# --- Experiment Runner ---

def run_dataset_benchmark(name, load_fn, is_classification=False, models_to_run=['CFN', 'MLP', 'XGB', 'LightGBM', 'TabNet']):
    print(f"\n>>> Benchmarking Dataset: {name}")
    data = load_fn()
    X, y = data.data, data.target
    if not is_classification:
         y = y.reshape(-1, 1)
    
    # Subsample California for speed if needed, but reviewer asked for real data
    if name == "California Housing":
        indices = np.random.choice(len(X), 5000, replace=False) # 5k samples
        X, y = X[indices], y[indices]
        
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) if is_classification else KFold(n_splits=10, shuffle=True, random_state=42)
    
    results = {m: [] for m in models_to_run}
    params_count = {m: 0 for m in models_to_run}
    
    # Split handling
    splits = list(kf.split(X, y if is_classification else np.zeros(len(y))))
    
    for fold, (train_idx, val_idx) in enumerate(splits):
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
            
        # Tensor conversion
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1)
        
        b_size = 512 if name == "California Housing" else 64
        t_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=b_size, shuffle=True)
        v_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=b_size, shuffle=False)
        
        if 'CFN' in models_to_run:
            gate = 'adaptive'
            poly_deg = 2
            
            if name == "California Housing":
                cfn = HighRankPolynomialStructuralCFN(input_dim=X.shape[1], poly_degree=poly_deg, rank=16, 
                                                      context_activation=gate, classification=is_classification)
            else:
                cfn = PolynomialStructuralCFN(input_dim=X.shape[1], poly_degree=poly_deg, 
                                              context_activation=gate, classification=is_classification)
            
            # (No poly freeze needed - Tanh Hybrid is now the Champion)

            score = train_and_eval_pytorch(cfn, t_loader, v_loader, is_classification, l1_lambda=0.0001)
            results['CFN'].append(score)
            params_count['CFN'] = count_parameters(cfn)
            
        # 2. MLP
        if 'MLP' in models_to_run:
            mlp = MLP(input_dim=X.shape[1], classification=is_classification)
            score = train_and_eval_pytorch(mlp, t_loader, v_loader, is_classification, epochs=200)
            results['MLP'].append(score)
            params_count['MLP'] = count_parameters(mlp)
            
        # 3. XGBoost (Tuned)
        if 'XGB' in models_to_run:
            if fold == 0:
                best_params = tune_xgboost(X_train_scaled, y_train_raw if is_classification else y_train_scaled, 
                                         X_val_scaled, y_val_raw if is_classification else y_val_scaled, 
                                         is_classification)
                print(f" [XGB Best Params: {best_params}] ", end='')
            
            if is_classification:
                m_xgb = xgb.XGBClassifier(**best_params, verbosity=0)
                m_xgb.fit(X_train_scaled, y_train_raw)
                preds = m_xgb.predict_proba(X_val_scaled)[:, 1]
                score = log_loss(y_val_raw, preds)
            else:
                m_xgb = xgb.XGBRegressor(**best_params, verbosity=0)
                m_xgb.fit(X_train_scaled, y_train_scaled)
                preds = m_xgb.predict(X_val_scaled)
                score = mean_squared_error(y_val_scaled, preds)
            results['XGB'].append(score)

        # 4. LightGBM (Tuned)
        if 'LightGBM' in models_to_run and HAS_LGBM:
            if fold == 0:
                print(" [Tuning LightGBM...] ", end='')
                lgbm_params = tune_lightgbm(X_train_scaled, y_train_raw if is_classification else y_train_scaled,
                                           X_val_scaled, y_val_raw if is_classification else y_val_scaled,
                                           is_classification)
            
            if is_classification:
                m_lgb = lgb.LGBMClassifier(**lgbm_params, verbosity=-1)
                m_lgb.fit(X_train_scaled, y_train_raw, eval_set=[(X_val_scaled, y_val_raw)], callbacks=[lgb.early_stopping(10, verbose=False)])
                preds = m_lgb.predict_proba(X_val_scaled)[:, 1]
                score = log_loss(y_val_raw, preds)
            else:
                m_lgb = lgb.LGBMRegressor(**lgbm_params, verbosity=-1)
                m_lgb.fit(X_train_scaled, y_train_scaled, eval_set=[(X_val_scaled, y_val_scaled)], callbacks=[lgb.early_stopping(10, verbose=False)])
                preds = m_lgb.predict(X_val_scaled)
                score = mean_squared_error(y_val_scaled, preds)
            results['LightGBM'].append(score)
            params_count['LightGBM'] = 0

        # 5. TabNet (Tuned)
        if 'TabNet' in models_to_run and HAS_TABNET:
            if fold == 0:
                print(" [Tuning TabNet...] ", end='')
                tn_params = tune_tabnet(X_train_scaled, y_train_raw if is_classification else y_train_scaled,
                                      X_val_scaled, y_val_raw if is_classification else y_val_scaled,
                                      is_classification)
            # Restructure parameters for TabNet
            lr_val = tn_params.pop('lr', 0.01)
            b_size = tn_params.pop('batch_size', 64)
            v_b_size = tn_params.pop('virtual_batch_size', 8)
            tn_params['optimizer_params'] = dict(lr=lr_val)
            
            if is_classification:
                clf = TabNetClassifier(**tn_params, verbose=0)
                clf.fit(X_train_scaled, y_train_raw.flatten(), eval_set=[(X_val_scaled, y_val_raw.flatten())], 
                        patience=20, max_epochs=200, batch_size=b_size, virtual_batch_size=v_b_size)
                preds = clf.predict_proba(X_val_scaled)[:, 1]
                score = log_loss(y_val_raw, preds)
            else:
                clf = TabNetRegressor(**tn_params, verbose=0)
                clf.fit(X_train_scaled, y_train_scaled, eval_set=[(X_val_scaled, y_val_scaled)], 
                        patience=20, max_epochs=200, batch_size=b_size, virtual_batch_size=v_b_size)
                preds = clf.predict(X_val_scaled)
                score = mean_squared_error(y_val_scaled, preds)
            results['TabNet'].append(score)
            params_count['TabNet'] = "N/A"

    print("\nBenchmark Complete.")
    
    # Statistics
    stats = {}
    cfn_scores = np.array(results['CFN']) if 'CFN' in results else None
    for m in models_to_run:
        scores = np.array(results[m])
        mean = np.mean(scores)
        std = np.std(scores)
        p_value = 1.0
        if m != 'CFN' and cfn_scores is not None and len(scores) == len(cfn_scores):
            _, p_value = ttest_rel(cfn_scores, scores)
        stats[m] = {
            'mean': mean,
            'std': std,
            'p_value': p_value,
            'params': params_count.get(m, 0)
        }
    return stats

def main():
    datasets = [
        ("Diabetes", load_diabetes, False),
        ("California Housing", fetch_california_housing, False),
        ("Breast Cancer", load_breast_cancer, True),
        ("Wine Quality", load_wine_quality, True),
        ("Heart Disease", load_heart_disease, True),
        ("Ionosphere", load_ionosphere, True),
    ]
    
    summary = {}
    for name, load_fn, is_class in datasets:
        summary[name] = run_dataset_benchmark(name, load_fn, is_class)
        
    print("\n" + "="*120)
    print(f"{'Dataset':<20} | {'Metric':<10} | {'Model':<10} | {'Score (Mean±SD)':<20} | {'p-val (vs CFN)':<15} | {'Params':<8}")
    print("-" * 120)
    
    for name, res in summary.items():
        is_classification_task = any(x in name for x in ["Cancer", "Wine", "Heart", "Ionosphere", "Bank"])
        m_type = "LogLoss" if is_classification_task else "MSE"
        is_classification_task = any(x in name for x in ["Cancer", "Wine", "Heart", "Ionosphere", "Bank"])
        m_type = "LogLoss" if is_classification_task else "MSE"
        first = True
        for model_name in ['CFN', 'MLP', 'XGB', 'LightGBM', 'TabNet']:
            if model_name not in res: continue
            dat = res[model_name]
            dataset_str = name if first else ""
            metric_str = m_type if first else ""
            first = False
            print(f"{dataset_str:<20} | {metric_str:<10} | {model_name:<10} | {dat['mean']:.4f}±{dat['std']:.3f}   | {dat['p_value']:.4f}          | {dat['params']}")
        print("-" * 120)

if __name__ == "__main__":
    main()
