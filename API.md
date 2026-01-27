# StructuralCFN API Reference

This document provides a detailed technical reference for the core classes and methods in the StructuralCFN library.

---

## 🏛️ Core Model Classes

### `GenericStructuralCFN`
The base class for all StructuralCFN models. It implements a two-stage architecture: a **Dependency Layer** for relational context and an **Aggregation Layer** for final prediction.

- **`__init__(input_dim, poly_degree=2, dropout_p=0.0, context_activation="adaptive", classification=False, prior_matrix=None)`**
    - `input_dim` (int): Number of input features.
    - `poly_degree` (int): The degree of polynomial basis functions in dependency nodes.
    - `dropout_p` (float): Dropout probability applied to the structural context.
    - `context_activation` (str): Activation for context nodes. Options: `"adaptive"`, `"sigmoid"`, `"tanh"`, `"linear"`.
    - `classification` (bool): If True, uses Cross-Entropy loss and Sigmoid/Argmax output.
    - `prior_matrix` (Optional[Tensor]): Initial relational prior of shape `(input_dim, input_dim)`.

- **`fit(X, y, epochs=200, lr=0.01, batch_size=64, verbose=True)`**
    - Standard scikit-learn style training loop.
    - Supports `numpy` arrays or `torch` tensors.
    - Automatically handles device placement (CUDA/CPU).
    - Includes an $L_1$ penalty for interaction sparsity.

- **`predict(X)`**
    - Performs inference and returns a `numpy` array of predictions.
    - For classification, returns class labels (or binary 0/1).

- **`distill_knowledge(feature_names=None)`**
    - Returns a dictionary containing the learned symbolic laws for each feature context.
    - Provides closed-form expressions for the Polynomial and Sinusoidal basis components.

- **`get_dependency_matrix()`**
    - Returns the $N \times N$ row-normalized Interaction Schema $M$.
    - $M_{ij}$ indicates the directed influence of feature $j$ on the structural context of feature $i$.

---

### `PolynomialStructuralCFN`
A specific variant optimized for stable scientific discovery on smaller datasets. It uses a committee of 4 aggregation heads (1 Linear, 2 Polynomial, 1 Sinusoidal).

---

### `HighRankPolynomialStructuralCFN`
A high-capacity variant designed for large-scale manifolds (e.g., California Housing). It utilizes an 18-head aggregator committee and Layer Normalization for stability.

---

## 🏗️ Functional Primitives (`scfn.cfn_pytorch`)

### `FunctionNode`
Base class for all mathematical basis nodes.
- **`.formula()`**: Returns a symbolic string representation of the node's learned mapping (e.g., `0.5 * sin(1.2 * u + 0.1)`).

### `PolynomialFunctionNode`
Computes $f(u) = \sum c_i u^i$ along a learned projection direction.

### `SinusoidalFunctionNode`
Computes $f(u) = A \sin(\omega u + \phi)$ along a learned projection direction.

### `SigmoidFunctionNode`
Computes a gated activation or polarity shift.

---

## 🧪 Utilities

- **`scfn.model.get_parameter_count()`**: Returns the total number of trainable parameters in the model.
