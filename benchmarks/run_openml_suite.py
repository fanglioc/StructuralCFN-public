
import sys
import os
import openml
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# Optimization to avoid excessive threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from benchmarks.run_benchmarks import tune_lightgbm # Unused
import lightgbm as lgb
from benchmarks.run_benchmarks import train_and_eval_pytorch, count_parameters, MLP
from scfn import PolynomialStructuralCFN

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_openml_suite():
    """
    Returns a list of (dataset_id, name, is_classification)
    Selected based on:
    - Tabular data
    - < 50k samples (for speed)
    - Numerical/Mixed features
    - Widely used benchmarks (CC-18 subset)
    """
    datasets = [
        (31, "credit-g", True),
        (37, "diabetes", True),         # Pima indian (verify overlap with ours)
        (1461, "bank-marketing", True),
        (1462, "banknote-authentication", True),
        (1464, "blood-transfusion", True),
        (1487, "ozone-level-8hr", True),
        (1489, "phoneme", True),
        (1494, "qsar-biodeg", True),
        (1510, "wdbc", True),           # Breast cancer (verify overlap)
        # (40996, "Fashion-MNIST", True), # Image data (multiclass) - Excluded for tabular focus
        (44120, "electricity", True),
        (44126, "pol", True),           # Polish companies bankruptcy
        (44129, "jungle_chess_2pcs_raw_endgame_complete", True) 
    ]
    return datasets

def process_dataset(did, name, is_classification):
    print(f"   Fetching OpenML ID {did} ({name})...")
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )
    
    # Basic Preprocessing
    # 1. Handle Categoricals: Label Encode for simplicity (LightGBM handles internally, CFN needs numbers)
    #    For CFN: We will just label encode for now. Future: Embeddings.
    for col in X.select_dtypes(include=['category', 'object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    # 2. Handle NaN: Simple mean imputation
    X = X.fillna(X.mean())
    
    # 3. Handle Target
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    return X.values, y

def run_openml_benchmark():
    set_seed(42)
    print(">>> Starting KDD OpenML Expansion Suite...")
    suite = get_openml_suite()
    
    results_df = []
    
    for did, name, is_classification in suite:
        print(f"\nProcessing: {name} (ID: {did})")
        try:
            X, y = process_dataset(did, name, is_classification)
            
            # Subsample if too large (> 10k) to keep it fast for user
            if len(X) > 10000:
                print(f"   Subsampling {len(X)} -> 10000...")
                indices = np.random.choice(len(X), 10000, replace=False)
                X = X[indices]
                y = y[indices]

            # 5-Fold CV (Speed over 10-fold rigor for this pass)
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            scores_cfn = []
            scores_lgbm = []
            scores_mlp = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Normalize Features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                
                # --- LightGBM ---
                # Quick tune on fold 0
                lgbm_params = {
                    'verbosity': -1, 
                    'metric': 'binary_logloss', 
                    'n_estimators': 100
                } 
                # (Skipping full optuna for speed, using decent defaults + early stopping)
                
                m_lgb = lgb.LGBMClassifier(**lgbm_params)
                m_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
                pred_lgbm = m_lgb.predict_proba(X_val)[:, 1]
                scores_lgbm.append(log_loss(y_val, pred_lgbm))
                
                # --- StructuralCFN ---
                # Heuristic: poly=2, gate=sigmoid (safe default)
                X_train_t = torch.tensor(X_train, dtype=torch.float32)
                y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
                
                t_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
                v_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
                
                cfn = PolynomialStructuralCFN(input_dim=X.shape[1], poly_degree=2, context_activation='adaptive', classification=True)
                # Fast train: 50 epochs
                score_cfn = train_and_eval_pytorch(cfn, t_loader, v_loader, is_classification=True, epochs=50, l1_lambda=1e-4)
                scores_cfn.append(score_cfn)
                
                # --- MLP ---
                mlp = MLP(input_dim=X.shape[1], classification=True)
                score_mlp = train_and_eval_pytorch(mlp, t_loader, v_loader, is_classification=True, epochs=50)
                scores_mlp.append(score_mlp)
                
            # Aggregate and Stats
            mean_cfn, std_cfn = np.mean(scores_cfn), np.std(scores_cfn)
            mean_lgbm, std_lgbm = np.mean(scores_lgbm), np.std(scores_lgbm)
            mean_mlp, std_mlp = np.mean(scores_mlp), np.std(scores_mlp)
            
            # P-value (Paired t-test vs LightGBM)
            from scipy.stats import ttest_rel
            p_val = 1.0
            if len(scores_cfn) == len(scores_lgbm):
                _, p_val = ttest_rel(scores_cfn, scores_lgbm)
            
            print(f"   [Results] CFN: {mean_cfn:.4f}±{std_cfn:.3f} | LGBM: {mean_lgbm:.4f}±{std_lgbm:.3f} | p={p_val:.4f}")
            
            # Determine Winner
            best_mean = min(mean_cfn, mean_lgbm) # Focus on CFN vs LGBM
            winner = "CFN" if mean_cfn < mean_lgbm else "LGBM"
            
            results_df.append({
                "Dataset": name,
                "ID": did,
                "N": len(X),
                "d": X.shape[1],
                "CFN_Mean": mean_cfn,
                "CFN_SD": std_cfn,
                "LGBM_Mean": mean_lgbm,
                "LGBM_SD": std_lgbm,
                "P_Value": p_val,
                "Winner": winner
            })
            
        except Exception as e:
            print(f"   [Error] Failed on {name}: {e}")

    # Save to CSV
    df = pd.DataFrame(results_df)
    df.to_csv("benchmarks/openml_results.csv", index=False)
    print("\n>>> OpenML Expansion Complete. Results saved to benchmarks/openml_results.csv")
    print(df)

if __name__ == "__main__":
    run_openml_benchmark()
