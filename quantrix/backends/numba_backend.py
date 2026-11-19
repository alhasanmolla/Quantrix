"""Numba JIT-Compiled Accelerated Backend for Quantrix.

Employs native LLVM code generation, OpenMP-style multithreading (prange),
and SIMD vectorized kernels for extreme performance on CPU.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np

from quantrix.backends.base import AbstractBackend
from quantrix.backends.numpy_backend import NumPyBackend

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, nogil=True)
    def _numba_batched_inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Parallel calculation of inner products across batch dimension."""
        batch_size = a.shape[0]
        dim = a.shape[1]
        results = np.zeros(batch_size, dtype=np.complex128)
        for i in prange(batch_size):
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += np.conj(a[i, j]) * b[i, j]
            results[i] = acc
        return results

    @njit(parallel=True, fastmath=True, nogil=True)
    def _numba_probabilities(amplitudes: np.ndarray) -> np.ndarray:
        """Parallel computation of squared magnitude probabilities |c_i|^2."""
        n = amplitudes.size
        flat = amplitudes.ravel()
        probs = np.empty(n, dtype=np.float64)
        for i in prange(n):
            c = flat[i]
            probs[i] = c.real * c.real + c.imag * c.imag
        return probs.reshape(amplitudes.shape)

    @njit(parallel=True, fastmath=True, nogil=True)
    def _numba_single_qubit_gate(
        state: np.ndarray,
        gate: np.ndarray,
        target_qubit: int,
        n_qubits: int
    ) -> np.ndarray:
        """High-speed native stride-based single-qubit gate application.

        Applies 2x2 unitary directly to flat or reshaped buffer in parallel chunks.
        """
        dim = 1 << n_qubits
        out = np.empty_like(state)
        # Map qubit axis to flat stride: qubit 0 is axis 0 (step = 2^(n_qubits - 1 - target_qubit))
        bit_pos = n_qubits - 1 - target_qubit
        step = 1 << bit_pos
        stride = step << 1
        num_blocks = dim // stride

        u00 = gate[0, 0]
        u01 = gate[0, 1]
        u10 = gate[1, 0]
        u11 = gate[1, 1]

        for b in prange(num_blocks):
            base = b * stride
            for k in range(step):
                idx0 = base + k
                idx1 = idx0 + step
                v0 = state[idx0]
                v1 = state[idx1]
                out[idx0] = u00 * v0 + u01 * v1
                out[idx1] = u10 * v0 + u11 * v1
        return out


class NumbaBackend(AbstractBackend):
    """Numba LLVM JIT-accelerated backend with parallelized native loops."""

    def __init__(self) -> None:
        self._numpy_backend = NumPyBackend()

    @property
    def name(self) -> str:
        return "numba"

    @property
    def is_available(self) -> bool:
        return NUMBA_AVAILABLE

    def einsum(
        self,
        subscripts: str,
        *operands: Any,
        optimize: Union[bool, str] = "optimal"
    ) -> np.ndarray:
        return self._numpy_backend.einsum(subscripts, *operands, optimize=optimize)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return self._numpy_backend.matmul(a, b)

    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> np.ndarray:
        return self._numpy_backend.tensordot(a, b, axes=axes)

    def norm(
        self,
        a: Any,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> Union[float, np.ndarray]:
        return self._numpy_backend.norm(a, axis=axis, keepdims=keepdims)

    def apply_gate(
        self,
        state_tensor: np.ndarray,
        gate_matrix: np.ndarray,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> np.ndarray:
        if not NUMBA_AVAILABLE or is_batched or len(target_qubits) != 1:
            return self._numpy_backend.apply_gate(
                state_tensor, gate_matrix, target_qubits, n_qubits, is_batched
            )

        # Single-qubit unbatched acceleration with native compiled kernel
        flat_state = np.ascontiguousarray(state_tensor.reshape(-1), dtype=np.complex128)
        gate = np.ascontiguousarray(gate_matrix, dtype=np.complex128)
        out_flat = _numba_single_qubit_gate(flat_state, gate, target_qubits[0], n_qubits)
        return out_flat.reshape(state_tensor.shape)

    def inner_product(
        self,
        state_a: np.ndarray,
        state_b: np.ndarray,
        is_batched: bool = False
    ) -> Union[complex, np.ndarray]:
        if not NUMBA_AVAILABLE or not is_batched:
            return self._numpy_backend.inner_product(state_a, state_b, is_batched)
        
        flat_a = np.ascontiguousarray(state_a.reshape(state_a.shape[0], -1), dtype=np.complex128)
        flat_b = np.ascontiguousarray(state_b.reshape(state_b.shape[0], -1), dtype=np.complex128)
        return _numba_batched_inner_product(flat_a, flat_b)

    def expectation_value(
        self,
        state_tensor: np.ndarray,
        operator: np.ndarray,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> Union[float, complex, np.ndarray]:
        return self._numpy_backend.expectation_value(
            state_tensor, operator, target_qubits, n_qubits, is_batched
        )

    def probabilities(self, amplitudes: np.ndarray) -> np.ndarray:
        """Calculates probabilities via parallel Numba kernel."""
        if not NUMBA_AVAILABLE:
            return np.abs(amplitudes) ** 2
        c_amps = np.ascontiguousarray(amplitudes, dtype=np.complex128)
        return _numba_probabilities(c_amps)

    def to_backend_array(
        self,
        array: Any,
        dtype: Optional[Any] = None,
        device: Optional[str] = None
    ) -> np.ndarray:
        return self._numpy_backend.to_backend_array(array, dtype=dtype, device=device)

    def to_numpy(self, array: Any) -> np.ndarray:
        return self._numpy_backend.to_numpy(array)
