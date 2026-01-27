# StructuralCFN: Interpretable Functional Tabular Discovery

[![PyPI version](https://img.shields.io/badge/version-1.1.0-blue)](https://github.com/fanglioc/StructuralCFN-public)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**StructuralCFN** is a PyTorch-based architecture that bridges the gap between Deep Learning and Symbolic Regression for tabular data. It imposes a **Relation-Aware Inductive Bias** to discover the governing "laws" of a data manifold through differentiable mathematical compositions.

## 🚀 Key Features

- **Intrinsic Symbolic Interpretability**: Recover human-readable mathematical laws (polynomial, sinusoidal, sigmoidal) directly from the data.
- **Differentiable Adaptive Gating**: Automatically discovers the optimal activation "physics" (attention-style filtering vs. inhibitory polarity) for each feature relationship.
- **Expert Knowledge Integration**: Inject domain-specific relational priors directly into the architecture to guide discovery.
- **Extreme Parameter Efficiency**: High performance with orders of magnitude fewer parameters than standard MLPs or Transformers.
- **Scikit-Learn Compatible API**: Simple `fit()` and `predict()` methods for seamless integration into existing pipelines.

---

## 🛠️ Installation

```bash
git clone https://github.com/fanglioc/StructuralCFN-public.git
cd StructuralCFN-public
pip install -e .
```

---

## 📖 Quick Start: Symbolic Discovery

Discover the laws governing your data in just a few lines of code:

```python
from scfn import GenericStructuralCFN
from sklearn.datasets import load_diabetes

# 1. Load data
data = load_diabetes()
X, y = data.data, data.target

# 2. Train with the high-level API
model = GenericStructuralCFN(input_dim=X.shape[1])
model.fit(X, y, epochs=100)

# 3. Predict
preds = model.predict(X)

# 4. Discover Inter-Feature Laws (Symbolic Regression)
knowledge = model.distill_knowledge(feature_names=data.feature_names)
print(f"Law for S5: {knowledge['dependencies']['s5']['poly_law']}")

# 5. Extract Global Relational Schema
matrix = model.get_dependency_matrix() 
# matrix[i, j] shows the directed influence of feature j on feature i
```

For a full demo, see [examples/quickstart.py](StructuralCFN-public/examples/quickstart.py).

---

### 📚 Documentation

- **[API Reference](StructuralCFN-public/API.md)**: Technical details on core classes, methods, and parameters.
- **[Contributing Guide](StructuralCFN-public/CONTRIBUTING.md)**: Guidelines for future enhancements.

---

## 📁 Repository Structure

- `scfn/`: Core library containing `MaskedHybridNode` and `HybridDependencyLayer`.
- `benchmarks/`: Comprehensive suite for reproducing the paper results.
- `examples/`: Getting-started guides and case studies (e.g., Diabetes lipid metabolic discovery).

---

## 🧠 Why StructuralCFN?

Traditional neural networks treat features as independent dimensions in a flat vector. **StructuralCFN** assumes that features in a tabular manifold are contextually interdependent. By explicitly modeling each feature as a mathematical composition of its neighbors, the model provides a "glass-box" view into the learned relationships, ensuring that its predictions remain grounded in verifiable scientific principles.

## 📜 Citation

If you use StructuralCFN in your research, please cite my paper: (coming soon...)

