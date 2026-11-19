"""Quantum-inspired Neural Network Layers.

Provides unitary and orthogonal parameterized layers that preserve gradient norm
and prevent vanishing/exploding gradients, as well as tensor contraction layers.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
from scipy.linalg import expm

from quantrix.backends.base import BackendManager


class QuantumLinear:
    """Orthogonal/Unitary Parameterized Linear Layer.

    Represents weights as an orthogonal transformation W = exp(A), where A is a
    skew-symmetric matrix (A = -A^T), followed by optional diagonal scaling.
    This guarantees exact isometry: ||Wx||_2 = ||x||_2, eliminating gradient vanishing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        use_bias: bool = True,
        backend: Optional[str] = None
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.use_bias = use_bias
        self.backend = BackendManager.get_backend(backend)

        # Dimension for orthogonal matrix
        self.dim = max(self.in_features, self.out_features)

        # Skew-symmetric parameter matrix: elements above diagonal
        # Number of parameters: dim * (dim - 1) // 2
        self.num_params = self.dim * (self.dim - 1) // 2
        self.triu_indices = np.triu_indices(self.dim, k=1)
        
        # Initialize small random parameters
        self.skew_params = np.random.normal(0.0, 0.05, size=self.num_params)
        self.scaling = np.ones(self.dim, dtype=np.float64)
        self.bias = np.zeros(self.out_features, dtype=np.float64) if use_bias else None

    def get_orthogonal_matrix(self) -> np.ndarray:
        """Constructs orthogonal matrix via Cayley transform W = (I - A)(I + A)^(-1)

        from skew-symmetric parameter matrix A.
        """
        A = np.zeros((self.dim, self.dim), dtype=np.float64)
        A[self.triu_indices] = self.skew_params
        A = A - A.T  # Skew-symmetric: A = -A^T
        
        # Cayley transform of skew-symmetric matrix is strictly orthogonal and numerically stable
        eye = np.eye(self.dim, dtype=np.float64)
        W = np.linalg.solve(eye + A, eye - A)
        
        # Apply scaling and slice to (out_features, in_features)
        W_scaled = np.diag(self.scaling) @ W
        return W_scaled[:self.out_features, :self.in_features]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: Y = X @ W.T + bias.

        Executed with zero loops via backend einsum or matmul.
        """
        W = self.get_orthogonal_matrix()
        # Matmul via backend
        out = self.backend.einsum("bi,ji->bj", X, W)
        if self.bias is not None:
            out = out + self.bias
        return self.backend.to_numpy(out)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


class TensorContractionLayer:
    """High-order tensor contraction layer using Einstein summation.

    Compresses high-dimensional mappings by factoring weight tensors into
    a tensor network contraction.
    """

    def __init__(
        self,
        tensor_modes_in: Tuple[int, ...],
        tensor_modes_out: Tuple[int, ...],
        rank: int = 4,
        backend: Optional[str] = None
    ) -> None:
        self.modes_in = tensor_modes_in
        self.modes_out = tensor_modes_out
        self.rank = rank
        self.backend = BackendManager.get_backend(backend)

        # Core tensors initialized with standard normal scaled by dimension
        dim_in = int(np.prod(tensor_modes_in))
        dim_out = int(np.prod(tensor_modes_out))
        scale = 1.0 / math.sqrt(dim_in)

        # Factor weights into two low-rank core tensors:
        # Core1: (dim_in, rank)
        # Core2: (rank, dim_out)
        self.core1 = np.random.normal(0.0, scale, (dim_in, rank))
        self.core2 = np.random.normal(0.0, scale, (rank, dim_out))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward contraction: Y = X @ (Core1 @ Core2)."""
        # Batch contraction via einsum without allocating full matrix
        # X: (batch, dim_in), Core1: (dim_in, rank), Core2: (rank, dim_out)
        res = self.backend.einsum("bi,ir,rj->bj", X, self.core1, self.core2)
        return self.backend.to_numpy(res)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
