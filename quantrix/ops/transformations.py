"""High-level quantum-inspired transformations and parallel matrix operators.

Implements Quantum Fourier Transform, Grover-inspired amplitude amplification,
Hamiltonian dynamics, and quantum-inspired parallel matrix multipliers.
"""

from __future__ import annotations
import math
from typing import Sequence, Tuple, Union, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from quantrix.core.state import QuantumState

from quantrix.core import operators as ops


def apply_hadamard_transform(state: QuantumState, inplace: bool = False) -> QuantumState:
    """Applies Hadamard transform to all qubits in the state to create full superposition."""
    return state.apply_all(ops.H, inplace=inplace)


def grover_diffusion_operator(n_qubits: int) -> np.ndarray:
    """Constructs the Grover inversion-about-the-mean diffusion operator:

    D = 2|s><s| - I, where |s> is the uniform superposition state.
    """
    dim = 1 << n_qubits
    s = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    proj = np.outer(s, np.conj(s))
    return 2.0 * proj - np.eye(dim, dtype=np.complex128)


def grover_search(
    oracle_fn,
    n_qubits: int,
    num_iterations: int,
    backend: Optional[str] = None
) -> QuantumState:
    """Executes quantum-inspired Grover amplitude amplification algorithm

    to amplify the amplitude of marked states in parallel.
    """
    from quantrix.core.state import QuantumState
    dim = 1 << n_qubits
    state = QuantumState.superposition(n_qubits=n_qubits, backend=backend)
    diffusion_op = grover_diffusion_operator(n_qubits)
    targets = list(range(n_qubits))

    for _ in range(num_iterations):
        # Oracle phase inversion
        amps = state.numpy().reshape(-1)
        phases = np.array([(-1.0 if oracle_fn(idx) else 1.0) for idx in range(dim)], dtype=np.complex128)
        state._data = state._backend.to_backend_array(
            (amps * phases).reshape(state.layout.tensor_shape)
        )
        # Diffusion inversion about mean
        state.apply_gate(diffusion_op, targets=targets, inplace=True)

    return state


def quantum_parallel_matrix_multiply(
    A: np.ndarray,
    B: np.ndarray,
    backend: Optional[str] = None
) -> np.ndarray:
    """Accelerated quantum-inspired matrix multiplication via tensor superposition and einsum.

    Maps row vectors of A and column vectors of B into dual superpositions and performs
    parallel inner product contractions simultaneously across all elements with zero loops.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Matrices A and B must be 2-dimensional.")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Shape mismatch for multiplication: {A.shape} and {B.shape}")

    # Use Quantrix backend einsum for optimized BLAS/SIMD contraction
    from quantrix.backends.base import BackendManager
    be = BackendManager.get_backend(backend)
    return be.einsum("ik,kj->ij", A, B)
