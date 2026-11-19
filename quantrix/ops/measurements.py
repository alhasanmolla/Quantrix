"""Quantum state metrics, fidelity, and information-theoretic measurements.

Includes vectorized state fidelity, trace distance, purity, Bloch sphere projections,
and quantum mutual information.
"""

from __future__ import annotations
from typing import Any, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from scipy.linalg import sqrtm

if TYPE_CHECKING:
    from quantrix.core.state import QuantumState

from quantrix.core import operators as ops


def state_fidelity(state_a: QuantumState, state_b: QuantumState) -> Union[float, np.ndarray]:
    """Computes fidelity F(|psi_a>, |psi_b>) = |<psi_a | psi_b>|^2 between pure states."""
    inner_val = state_a.inner(state_b)
    return np.abs(inner_val) ** 2


def density_matrix_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Computes Uhlmann's fidelity between density matrices:

    F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2
    """
    sqrt_rho = sqrtm(rho)
    prod = sqrt_rho @ sigma @ sqrt_rho
    sqrt_prod = sqrtm(prod)
    trace_val = np.trace(sqrt_prod)
    return float(np.real(trace_val * np.conj(trace_val)))


def trace_distance(state_a: QuantumState, state_b: QuantumState) -> float:
    """Computes trace distance D(rho, sigma) = 1/2 * Tr|rho - sigma| = 1/2 * sum(|eigvals|)."""
    rho = state_a.density_matrix()
    sigma = state_b.density_matrix()
    diff = rho - sigma
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def purity(state: Any) -> float:
    """Calculates state purity gamma = Tr(rho^2). For pure states, gamma = 1.0."""
    if hasattr(state, "density_matrix"):
        rho = state.density_matrix()
    else:
        rho = state
    return float(np.real(np.trace(rho @ rho)))


def bloch_vector(state_1q: QuantumState) -> Tuple[float, float, float]:
    """Computes the 3D Bloch sphere vector (rx, ry, rz) for a 1-qubit state:

    rx = <X>, ry = <Y>, rz = <Z>.
    """
    if state_1q.n_qubits != 1:
        raise ValueError("Bloch vector only defined for single-qubit states.")
    rx = float(np.real(state_1q.expectation(ops.X, targets=0)))
    ry = float(np.real(state_1q.expectation(ops.Y, targets=0)))
    rz = float(np.real(state_1q.expectation(ops.Z, targets=0)))
    return (rx, ry, rz)


def quantum_mutual_information(
    state: QuantumState,
    subsystem_a: Sequence[int],
    subsystem_b: Sequence[int]
) -> float:
    """Calculates quantum mutual information I(A : B) = S(A) + S(B) - S(AB)."""
    s_a = state.von_neumann_entropy(subsystem_a)
    s_b = state.von_neumann_entropy(subsystem_b)
    
    subsystem_ab = sorted(list(set(subsystem_a).union(set(subsystem_b))))
    s_ab = state.von_neumann_entropy(subsystem_ab)
    
    return float(s_a + s_b - s_ab)
