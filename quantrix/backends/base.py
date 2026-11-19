"""Abstract Base Class and Backend Management for Quantrix.

Defines unified execution interfaces for matrix and tensor operations across
NumPy, Numba, PyTorch, and compiled C extensions.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union
import numpy as np


class AbstractBackend(ABC):
    """Abstract interface defining required computational primitives for matrix

    and tensor operations in Quantrix.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Indicates whether backend dependencies are installed and accessible."""
        pass

    @abstractmethod
    def einsum(self, subscripts: str, *operands: Any, optimize: Union[bool, str] = "optimal") -> Any:
        """Executes Einstein summation over tensors."""
        pass

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication / batched matrix multiplication."""
        pass

    @abstractmethod
    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> Any:
        """Tensor contraction along specified axes."""
        pass

    @abstractmethod
    def norm(self, a: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Any:
        """L2 Euclidean / Frobenius norm."""
        pass

    @abstractmethod
    def apply_gate(
        self,
        state_tensor: Any,
        gate_matrix: Any,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> Any:
        """Applies a single or multi-qubit gate to state tensor via vectorized contraction."""
        pass

    @abstractmethod
    def inner_product(self, state_a: Any, state_b: Any, is_batched: bool = False) -> Any:
        """Computes inner product <state_a | state_b>."""
        pass

    @abstractmethod
    def expectation_value(
        self,
        state_tensor: Any,
        operator: Any,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> Any:
        """Computes <psi| O |psi> using einsum without explicit loops."""
        pass

    @abstractmethod
    def to_backend_array(self, array: Any, dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """Converts arbitrary array or tensor into native backend array."""
        pass

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Converts native backend array back into standard NumPy array."""
        pass


class BackendManager:
    """Registry and factory for managing computational backends."""

    _backends: Dict[str, AbstractBackend] = {}
    _default_backend_name: str = "numpy"

    @classmethod
    def register_backend(cls, backend: AbstractBackend) -> None:
        """Registers a backend instance in the manager."""
        cls._backends[backend.name.lower()] = backend

    @classmethod
    def get_backend(cls, name: Optional[str] = None) -> AbstractBackend:
        """Retrieves a backend by name or returns the default backend.

        If requested backend is unavailable, automatically falls back to NumPy.
        """
        target = (name or cls._default_backend_name).lower()
        if target not in cls._backends:
            raise KeyError(
                f"Backend '{target}' not registered. Registered: {list(cls._backends.keys())}"
            )
        backend = cls._backends[target]
        if not backend.is_available:
            # Fallback to NumPy
            if target != "numpy" and "numpy" in cls._backends:
                return cls._backends["numpy"]
            raise RuntimeError(f"Requested backend '{target}' is not available on this system.")
        return backend

    @classmethod
    def set_default_backend(cls, name: str) -> None:
        """Sets the global default computational backend."""
        target = name.lower()
        if target not in cls._backends:
            raise KeyError(f"Backend '{target}' is not registered.")
        if not cls._backends[target].is_available:
            raise RuntimeError(f"Cannot set default backend to '{target}'; backend is unavailable.")
        cls._default_backend_name = target

    @classmethod
    def list_backends(cls) -> Dict[str, bool]:
        """Returns dict of registered backends and their availability."""
        return {name: b.is_available for name, b in cls._backends.items()}
