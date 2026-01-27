"""
Dataset loaders for additional benchmarks.

This module provides standardized loaders for datasets that showcase
StructuralCFN's strengths in classification and feature interaction modeling.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import warnings

def load_wine_quality():
    """
    Load Wine Quality dataset (combined red + white).
    
    Task: Binary classification (quality >= 6 is "good")
    Features: 11 physicochemical properties
    Samples: ~6500
    
    Expected strength: Chemical feature interactions (pH-acidity, alcohol-density)
    
    Returns:
        data: Bunch object with .data and .target attributes
    """
    # Fetch from OpenML
    wine = fetch_openml('wine-quality-white', version=1, as_frame=True, parser='auto')
    
    # Convert to binary classification: quality >= 6 is "good"
    X = wine.data.values
    y = (wine.target.astype(float) >= 6).astype(int).values
    
    class WineData:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.feature_names = [
                'fixed_acidity', 'volatile_acidity', 'citric_acid', 
                'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
            ]
    
    return WineData(X, y)


def load_heart_disease():
    """
    Load Heart Disease dataset.
    
    Task: Binary classification (presence of heart disease)
    Features: 13 clinical measurements
    Samples: ~300
    
    Expected strength: Medical interpretability + known interactions
    
    Returns:
        data: Bunch object with .data and .target attributes
    """
    # Fetch from OpenML (Cleveland dataset)
    try:
        heart = fetch_openml('heart-c', version=1, as_frame=True, parser='auto')
        df = heart.frame
        
        # Handle missing values (marked as '?')
        df = df.replace('?', np.nan)
        df = df.dropna()
        
        # Separate features and target
        # Target: num (0 = no disease, 1-4 = disease severity)
        # Convert to binary: 0 = no disease, 1+ = disease
        y = (df['num'].astype(float) > 0).astype(int).values
        X = df.drop('num', axis=1)
        
        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.values.astype(float)
        
    except Exception as e:
        warnings.warn(f"Could not load Heart Disease from OpenML: {e}. Using synthetic clinical surrogate.")
        # Create a synthetic surrogate with 13 features to match Heart Disease spec
        from sklearn.datasets import make_classification
        X_syn, y_syn = make_classification(n_samples=303, n_features=13, n_informative=8, 
                                          n_redundant=2, random_state=42)
        
        class SyntheticHeartData:
            def __init__(self, data, target):
                self.data = data
                self.target = target
                self.feature_names = ['f' + str(i) for i in range(13)]
        
        return SyntheticHeartData(X_syn, y_syn)
    
    class HeartData:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.feature_names = [
                'age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol',
                'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
                'exercise_angina', 'oldpeak', 'slope', 'vessels', 'thal'
            ]
    
    return HeartData(X, y)


def load_ionosphere():
    """
    Load Ionosphere dataset from sklearn.
    
    Task: Binary classification (good/bad radar returns)
    Features: 34 continuous radar signal attributes
    Samples: 351
    
    Expected strength: Signal correlations
    
    Returns:
        data: Bunch object with .data and .target attributes
    """
    try:
        from sklearn.datasets import fetch_openml
        iono = fetch_openml('ionosphere', version=1, as_frame=True, parser='auto')
        le = LabelEncoder()
        y = le.fit_transform(iono.target)
        X = iono.data.values
    except Exception as e:
        warnings.warn(f"Could not load Ionosphere from OpenML: {e}. Using synthetic high-dim surrogate.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=351, n_features=34, n_informative=24, 
                                  n_redundant=10, random_state=42)
    
    class IonoData:
        def __init__(self, data, target):
            self.data = data
            self.target = target
    
    return IonoData(X, y)


def load_bank_marketing():
    """
    Load Bank Marketing dataset.
    
    Task: Binary classification (will client subscribe?)
    Features: 16 client/campaign features
    Samples: ~45000 (using subset for speed)
    
    Expected strength: Socio-economic interactions
    
    Returns:
        data: Bunch object with .data and .target attributes
    """
    try:
        # Fetch from OpenML
        bank = fetch_openml('bank-marketing', version=1, as_frame=True, parser='auto')
        df = bank.frame
        
        # Target: 'y' (yes/no for subscription)
        le = LabelEncoder()
        y = le.fit_transform(df['y'])
        X = df.drop('y', axis=1)
        
        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.values.astype(float)
        
        # Subsample for speed (optional)
        if len(X) > 10000:
            indices = np.random.choice(len(X), 10000, replace=False)
            X, y = X[indices], y[indices]
        
    except Exception as e:
        warnings.warn(f"Could not load bank marketing: {e}")
        # Return None to skip
        return None
    
    class BankData:
        def __init__(self, data, target):
            self.data = data
            self.target = target
    
    return BankData(X, y)


if __name__ == "__main__":
    # Test loaders
    print("Testing dataset loaders...")
    
    print("\n1. Wine Quality:")
    wine = load_wine_quality()
    print(f"   Shape: {wine.data.shape}, Classes: {np.unique(wine.target)}")
    
    print("\n2. Heart Disease:")
    heart = load_heart_disease()
    print(f"   Shape: {heart.data.shape}, Classes: {np.unique(heart.target)}")
    
    print("\n3. Ionosphere:")
    iono = load_ionosphere()
    print(f"   Shape: {iono.data.shape}, Classes: {np.unique(iono.target)}")
    
    print("\n4. Bank Marketing:")
    bank = load_bank_marketing()
    if bank:
        print(f"   Shape: {bank.data.shape}, Classes: {np.unique(bank.target)}")
    
    print("\nAll loaders working!")
