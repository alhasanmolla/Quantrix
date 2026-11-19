"""Quantum-inspired operators and unitary matrix generators.

Contains standard quantum gates, parameterized rotations, multi-qubit entanglers,
Kronecker tensor products, and Hamiltonian time-evolution operators.
"""

from __future__ import annotations
import math
from typing import Sequence, Union
import numpy as np
from scipy.linalg import expm


# 1-Qubit Fundamental Gates (2x2)
I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
T = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=np.complex128)


def rx(theta: float) -> np.ndarray:
    """X-axis rotation unitary: exp(-i * theta * X / 2)."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([
        [c, -1.0j * s],
        [-1.0j * s, c]
    ], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    """Y-axis rotation unitary: exp(-i * theta * Y / 2)."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([
        [c, -s],
        [s, c]
    ], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    """Z-axis rotation unitary: exp(-i * theta * Z / 2)."""
    e_neg = np.exp(-1.0j * theta / 2.0)
    e_pos = np.exp(1.0j * theta / 2.0)
    return np.array([
        [e_neg, 0.0],
        [0.0, e_pos]
    ], dtype=np.complex128)


def phase(theta: float) -> np.ndarray:
    """Phase shift gate: diag(1, exp(i * theta))."""
    return np.array([
        [1.0, 0.0],
        [0.0, np.exp(1.0j * theta)]
    ], dtype=np.complex128)


def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """Arbitrary single-qubit unitary parameterized by 3 Euler angles."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([
        [c, -np.exp(1.0j * lam) * s],
        [np.exp(1.0j * phi) * s, np.exp(1.0j * (phi + lam)) * c]
    ], dtype=np.complex128)


# 2-Qubit Entangling Gates (4x4)
CNOT = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0]
], dtype=np.complex128)

CZ = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, -1.0]
], dtype=np.complex128)

SWAP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.complex128)


def cphase(theta: float) -> np.ndarray:
    """Controlled Phase gate."""
    diag = np.array([1.0, 1.0, 1.0, np.exp(1.0j * theta)], dtype=np.complex128)
    return np.diag(diag)


# 3-Qubit Gates (8x8)
TOFFOLI = np.eye(8, dtype=np.complex128)
TOFFOLI[6:8, 6:8] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def kron(*matrices: np.ndarray) -> np.ndarray:
    """Computes Kronecker tensor product A (x) B (x) ... across arbitrary sequence."""
    if not matrices:
        raise ValueError("At least one matrix must be provided.")
    res = matrices[0]
    for m in matrices[1:]:
        res = np.kron(res, m)
    return res


def qft_matrix(n_qubits: int) -> np.ndarray:
    """Constructs the N x N Quantum Fourier Transform unitary matrix where N = 2^n_qubits."""
    dim = 1 << n_qubits
    indices = np.arange(dim)
    # Omega = exp(2 * pi * i / N)
    omega = np.exp(2.0j * np.pi / dim)
    grid = np.outer(indices, indices)
    mat = (1.0 / np.sqrt(dim)) * (omega ** grid)
    return np.asarray(mat, dtype=np.complex128)


def random_unitary(dim: int, seed: Union[int, None] = None) -> np.ndarray:
    """Generates a Haar-random unitary matrix of dimension (dim x dim)."""
    rng = np.random.default_rng(seed)
    real_part = rng.standard_normal((dim, dim))
    imag_part = rng.standard_normal((dim, dim))
    z = (real_part + 1.0j * imag_part) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph


def hamiltonian_evolution_operator(H_mat: np.ndarray, time: float) -> np.ndarray:
    """Computes U(t) = exp(-i * H * t) via spectral decomposition."""
    eigvals, eigvecs = np.linalg.eigh(H_mat)
    diag_exp = np.exp(-1.0j * eigvals * time)
    return (eigvecs * diag_exp) @ np.conj(eigvecs.T)
