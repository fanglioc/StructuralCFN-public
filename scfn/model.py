import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm

from .cfn_pytorch.function_nodes import FunctionNode, FunctionNodeFactory
from .cfn_pytorch.composition_layers import ParallelCompositionLayer, CompositionFunctionNetwork

class MaskedHybridNode(FunctionNode):
    """
    The Definitive Champion Structural Hybrid Node.
    Uses Poly + Sin basis with stable Intra-Interaction Scaling.
    """
    def __init__(self, input_dim: int, active_indices: List[int], total_input_dim: int, 
                 poly_degree: int = 2, context_activation: str = "adaptive",
                 prior_direction: Optional[torch.Tensor] = None):
        super().__init__(total_input_dim)
        self.active_indices = active_indices
        self.subset_dim = len(active_indices)
        self.context_activation = context_activation
        
        node_factory = FunctionNodeFactory()
        
        # Initialize directions with prior if provided
        p_dir, s_dir = None, None
        if prior_direction is not None:
            # Assumes prior_direction is of length total_input_dim, we need subset_dim
            p_dir = prior_direction[active_indices].clone()
            s_dir = prior_direction[active_indices].clone()

        self.poly_node = node_factory.create('Polynomial', input_dim=self.subset_dim, 
                                            degree=poly_degree, direction=p_dir)
        self.sin_node = node_factory.create('Sinusoidal', input_dim=self.subset_dim, 
                                           direction=s_dir)
        
        self.layernorm = nn.LayerNorm(2)
        self.combine = nn.Linear(2, 1)
        
        if self.context_activation == "adaptive":
            self.gate_weights = nn.Parameter(torch.zeros(2)) # [w_sig, w_tanh]
        self.output_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refactored forward pass: Normalizes the vector [poly, sin] before combination.
        This aligns with the reviewer's dimensionality fix.
        """
        x_subset = x[:, self.active_indices]
        v_poly = self.poly_node(x_subset)
        v_sin = self.sin_node(x_subset)
        
        # Form a basis vector h_i in R^2
        h_i = torch.cat([v_poly, v_sin], dim=1) 
        
        # Apply normalization to the vector (valid for dim=2)
        h_i = self.layernorm(h_i)
        
        # Linear combination to aggregate into context v_i
        v_i = self.combine(h_i)
        
        if self.context_activation == "sigmoid":
            out = torch.sigmoid(v_i)
        elif self.context_activation == "tanh":
            out = torch.tanh(v_i)
        elif self.context_activation == "adaptive":
            w = F.softmax(self.gate_weights, dim=0)
            out = w[0] * torch.sigmoid(v_i) + w[1] * torch.tanh(v_i)
        else:
            out = v_i
        return out

class HybridDependencyLayer(ParallelCompositionLayer):
    def __init__(self, input_dim: int, poly_degree: int = 2, 
                 context_activation: str = "adaptive",
                 prior_matrix: Optional[torch.Tensor] = None):
        """
        prior_matrix: (input_dim, input_dim) tensor of initial feature weights.
        """
        nodes = []
        for i in range(input_dim):
            p_dir = prior_matrix[i] if prior_matrix is not None else None
            nodes.append(MaskedHybridNode(input_dim - 1, [j for j in range(input_dim) if j != i], 
                                         input_dim, poly_degree, context_activation, p_dir))
        super().__init__(function_nodes=nodes, combination='concat')
        self.input_dim = input_dim

class GenericStructuralCFN(CompositionFunctionNetwork):
    def __init__(self, input_dim: int, poly_degree: int = 2, dropout_p: float = 0.0, 
                 context_activation: str = "adaptive", classification: bool = False,
                 prior_matrix: Optional[torch.Tensor] = None):
        super().__init__(layers=[], name="Standard Structural CFN")
        self.input_dim = input_dim
        self.is_classification = classification
        self.dependency_layer = HybridDependencyLayer(input_dim, poly_degree, context_activation, prior_matrix)
        
        self.aggregator = nn.Linear(input_dim * 2, 1)
        self.dropout = nn.Dropout(dropout_p)
        self.layers.append(self.dependency_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dependency_layer(x)
        z = self.dropout(z)
        combined = torch.cat([x, z], dim=1)
        out = self.aggregator(combined)
        if self.is_classification:
            out = torch.sigmoid(out)
        return out

    def distill_knowledge(self, feature_names: Optional[List[str]] = None) -> Dict:
        """
        Extracts learned symbolic laws from the dependency and aggregation layers.
        """
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.input_dim)]
        
        knowledge = {"dependencies": {}, "aggregation": []}
        
        # 1. Dependency Laws
        for i, node in enumerate(self.dependency_layer.function_nodes):
            active_names = [feature_names[j] for j in node.active_indices]
            w = F.softmax(node.combine.weight.data.squeeze(), dim=0) if hasattr(node.combine, 'weight') else torch.tensor([0.5, 0.5])
            
            # Simplified projection description
            p_formula = node.poly_node.formula()
            s_formula = node.sin_node.formula()
            
            knowledge["dependencies"][feature_names[i]] = {
                "inputs": active_names,
                "poly_law": p_formula,
                "sin_law": s_formula,
                "composition_weights": w.tolist()
            }
            
        # 2. Aggregator (Simplified)
        if hasattr(self, 'agg_ensemble'):
            for subnode in self.agg_ensemble.function_nodes:
                knowledge["aggregation"].append(subnode.describe())
        
        return knowledge

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], 
            epochs: int = 200, lr: float = 0.01, batch_size: int = 64, 
            verbose: bool = True):
        """
        Trains the model on the provided data.
        
        Args:
            X: Input features (numpy array or torch tensor).
            y: Target values.
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Training batch size.
            verbose: Whether to show progress bar.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y) if not self.is_classification else torch.LongTensor(y)
            
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss() if not self.is_classification else nn.CrossEntropyLoss()
        
        self.train()
        pbar = tqdm(range(epochs), disable=not verbose)
        for epoch in pbar:
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = self(batch_x)
                if not self.is_classification:
                    output = output.squeeze()
                
                loss = criterion(output, batch_y)
                
                # Add L1 penalty for sparsity per paper Section 3.6
                l1_penalty = 0
                for node in self.dependency_layer.function_nodes:
                    l1_penalty += torch.norm(node.poly_node.direction, 1)
                    l1_penalty += torch.norm(node.sin_node.direction, 1)
                loss += 1e-4 * l1_penalty
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            pbar.set_description(f"Loss: {total_loss/len(loader):.4f}")
            
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Performs inference on the provided data.
        
        Args:
            X: Input features.
            
        Returns:
            numpy.ndarray: Model predictions.
        """
        self.eval()
        device = next(self.parameters()).device
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        X = X.to(device)
        
        with torch.no_grad():
            output = self(X)
            if self.is_classification:
                if output.dim() > 1 and output.size(1) > 1:
                    return torch.argmax(output, dim=1).cpu().numpy()
                else:
                    return (output > 0.5).long().cpu().numpy().squeeze()
            return output.cpu().numpy().squeeze()

    def get_parameter_count(self) -> int:
        """Precisely count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_dependency_matrix(self, epsilon: float = 1e-8) -> np.ndarray:
        """
        Returns the row-normalized dependency matrix M.
        Added epsilon for numerical stability per reviewer feedback.
        """
        matrix = np.zeros((self.input_dim, self.input_dim))
        for i, node in enumerate(self.dependency_layer.function_nodes):
            p_dir = torch.abs(node.poly_node.direction).detach().cpu()
            s_dir = torch.abs(node.sin_node.direction).detach().cpu()
            w = torch.abs(node.combine.weight.data).detach().cpu().squeeze()
            
            # Weighted importance of indices
            importance = (w[0]*p_dir + w[1]*s_dir).numpy()
            
            for local_idx, global_idx in enumerate(node.active_indices):
                matrix[i, global_idx] = importance[local_idx]
            
            row_sum = np.sum(importance)
            if row_sum > epsilon:
                matrix[i, :] /= row_sum
        return matrix

class PolynomialStructuralCFN(GenericStructuralCFN):
    """The Winning Committee: Leaner and more Stable (1 Linear + 2 Poly + 1 Sin)."""
    def __init__(self, input_dim: int, poly_degree: int = 2, dropout_p: float = 0.0, 
                 context_activation: str = "adaptive", classification: bool = False,
                 prior_matrix: Optional[torch.Tensor] = None):
        super().__init__(input_dim, poly_degree, dropout_p, context_activation, classification, prior_matrix)
        factory = FunctionNodeFactory()
        from .cfn_pytorch.function_nodes import LinearFunctionNode
        
        # Committee of 4 (Balanced for small data)
        nodes = []
        nodes.append(LinearFunctionNode(input_dim=input_dim * 2, output_dim=1))
        nodes.append(factory.create('Polynomial', input_dim=input_dim * 2, degree=poly_degree))
        nodes.append(factory.create('Polynomial', input_dim=input_dim * 2, degree=poly_degree))
        nodes.append(factory.create('Sinusoidal', input_dim=input_dim * 2))
        
        self.agg_ensemble = ParallelCompositionLayer(nodes, combination='sum')
        # NO AggNorm for small data (prevents signal suppression)
        self.aggregator = self.agg_ensemble

class HighRankPolynomialStructuralCFN(GenericStructuralCFN):
    def __init__(self, input_dim: int, poly_degree: int = 2, rank: int = 16, 
                 dropout_p: float = 0.0, context_activation: str = "adaptive", classification: bool = False,
                 prior_matrix: Optional[torch.Tensor] = None):
        super().__init__(input_dim, poly_degree, dropout_p, context_activation, classification, prior_matrix)
        factory = FunctionNodeFactory()
        nodes = []
        from .cfn_pytorch.function_nodes import LinearFunctionNode
        for _ in range(2): nodes.append(LinearFunctionNode(input_dim=input_dim * 2, output_dim=1))
        for _ in range(8): nodes.append(factory.create('Polynomial', input_dim=input_dim * 2, degree=poly_degree))
        for _ in range(3): nodes.append(factory.create('Sinusoidal', input_dim=input_dim * 2))
        for _ in range(3): nodes.append(factory.create('Sigmoid', input_dim=input_dim * 2, direction=torch.randn(input_dim * 2)))
        
        self.agg_norm = nn.LayerNorm(input_dim * 2)
        self.agg_ensemble = ParallelCompositionLayer(nodes, combination='sum')
        self.aggregator = nn.Sequential(self.agg_norm, self.agg_ensemble)

class GatedStructuralCFN(GenericStructuralCFN):
    def __init__(self, input_dim: int, poly_degree: int = 2, dropout_p: float = 0.0, 
                 context_activation: str = "adaptive", classification: bool = False): 
        super().__init__(input_dim, poly_degree, dropout_p, context_activation, classification)
        self.aggregator = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dependency_layer(x)
        gate = torch.sigmoid(z)
        gated_features = x * gate
        gated_features = self.dropout(gated_features)
        out = self.aggregator(gated_features)
        if self.is_classification:
            out = torch.sigmoid(out)
        return out
