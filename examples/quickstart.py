import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scfn import GenericStructuralCFN

def demo():
    # 1. Load and Prepare Data
    print("--- 📂 Loading Data ---")
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Standardize for functional discovery
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Initialize and Fit Model (New high-level API)
    print("\n--- 🧠 Training StructuralCFN ---")
    model = GenericStructuralCFN(input_dim=X.shape[1], classification=False)
    
    # Simple scikit-learn style fit
    model.fit(X_train, y_train, epochs=100, lr=0.01, verbose=True)
    
    # 3. Predict and Evaluate
    print("\n--- 📈 Evaluation ---")
    preds = model.predict(X_test)
    mse = np.mean((preds - y_test)**2)
    print(f"Test MSE: {mse:.4f}")
    
    # 4. Intrinsic Interpretability: Discovering Laws
    print("\n--- 🔍 Symbolic Discovery: Learned Inter-Feature Laws ---")
    knowledge = model.distill_knowledge(feature_names=feature_names)
    
    # Examine law for 's5' (Serum Triglycerides)
    s5_law = knowledge["dependencies"]["s5"]
    print(f"Target: s5")
    print(f"Primary Drivers: {s5_law['inputs']}")
    print(f"Poly Law: {s5_law['poly_law']}")
    print(f"Sin Law:  {s5_law['sin_law']}")
    
    # 5. Global Relational Schema: Dependency Matrix
    print("\n--- 🗺️ Global Relational Schema ---")
    matrix = model.get_dependency_matrix()
    print(f"Dependency Matrix Shape: {matrix.shape}")
    print("Matrix[i, j] represents how much feature j influences context i.")

if __name__ == "__main__":
    demo()
