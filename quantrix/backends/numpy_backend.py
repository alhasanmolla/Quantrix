"""High-performance NumPy Backend for Quantrix.

Harnesses vectorized SIMD array operations, broadcasting, and optimized
Einstein summation (einsum) to execute multi-qubit and batch transformations
with zero Python loops.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np

from quantrix.backends.base import AbstractBackend
from quantrix.ops.einsum_ops import (
    generate_gate_einsum,
    generate_expectation_einsum,
    generate_batched_inner_product_einsum,
)


class NumPyBackend(AbstractBackend):
    """Core backend utilizing NumPy's C-accelerated BLAS/LAPACK and einsum engines."""

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def is_available(self) -> bool:
        return True

    def einsum(
        self,
        subscripts: str,
        *operands: Any,
        optimize: Union[bool, str] = "optimal"
    ) -> np.ndarray:
        return np.einsum(subscripts, *operands, optimize=optimize)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return np.matmul(a, b)

    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> np.ndarray:
        return np.tensordot(a, b, axes=axes)

    def norm(
        self,
        a: Any,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> Union[float, np.ndarray]:
        return np.linalg.norm(a, axis=axis, keepdims=keepdims)

    def apply_gate(
        self,
        state_tensor: np.ndarray,
        gate_matrix: np.ndarray,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> np.ndarray:
        """Applies gate to state tensor via optimal einsum contraction.

        No iteration over states or qubits occurs in Python.
        """
        k = len(target_qubits)
        gate_single_dim = 1 << (2 * k)
        gate_is_batched = gate_matrix.size > gate_single_dim and (gate_matrix.size % gate_single_dim == 0)

        if gate_is_batched:
            b_size = gate_matrix.size // gate_single_dim
            gate_tensor_shape = (b_size,) + (2,) * (2 * k)
        else:
            gate_tensor_shape = (2,) * (2 * k)

        gate_tensor = gate_matrix.reshape(gate_tensor_shape)

        einsum_str, _ = generate_gate_einsum(
            target_qubits, n_qubits, is_batched=is_batched, gate_is_batched=gate_is_batched
        )
        return np.einsum(einsum_str, gate_tensor, state_tensor, optimize="optimal")

    def inner_product(
        self,
        state_a: np.ndarray,
        state_b: np.ndarray,
        is_batched: bool = False
    ) -> Union[complex, np.ndarray]:
        """Computes <state_a | state_b> = sum(conj(state_a) * state_b)."""
        if not is_batched:
            return np.vdot(state_a.reshape(-1), state_b.reshape(-1))
        
        # Batched inner product
        flat_a = state_a.reshape(state_a.shape[0], -1)
        flat_b = state_b.reshape(state_b.shape[0], -1)
        return np.sum(np.conj(flat_a) * flat_b, axis=-1)

    def expectation_value(
        self,
        state_tensor: np.ndarray,
        operator: np.ndarray,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> Union[float, complex, np.ndarray]:
        """Calculates <psi| O |psi> in one unified einsum contraction."""
        k = len(target_qubits)
        op_tensor = operator.reshape((2,) * (2 * k))
        einsum_str, _ = generate_expectation_einsum(target_qubits, n_qubits, is_batched=is_batched)
        
        val = np.einsum(einsum_str, op_tensor, np.conj(state_tensor), state_tensor, optimize="optimal")
        return val

    def to_backend_array(
        self,
        array: Any,
        dtype: Optional[Any] = None,
        device: Optional[str] = None
    ) -> np.ndarray:
        if isinstance(array, np.ndarray):
            if dtype is not None and array.dtype != dtype:
                return array.astype(dtype)
            return array
        return np.asarray(array, dtype=dtype)

    def to_numpy(self, array: Any) -> np.ndarray:
        if isinstance(array, np.ndarray):
            return array
        return np.asarray(array)
