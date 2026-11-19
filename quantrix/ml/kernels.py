"""Quantum-inspired kernel estimation and feature embeddings for machine learning.

Transforms classical data into high-dimensional quantum Hilbert states via
vectorized amplitude and angle embeddings, and computes complete Gram matrices
simultaneously using batched Einstein summation.
"""

from __future__ import annotations
import math
from typing import Optional, Union, Literal
import numpy as np

from quantrix.core.state import QuantumState
from quantrix.core import operators as ops
from quantrix.ops.einsum_ops import generate_batched_inner_product_einsum


class QuantumFeatureMap:
    """Encodes classical data vectors into quantum superposition statevectors."""

    def __init__(
        self,
        n_qubits: int,
        encoding: Literal["angle", "amplitude", "zz"] = "angle",
        backend: Optional[str] = None
    ) -> None:
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.backend = backend

    def encode(self, X: np.ndarray) -> QuantumState:
        """Encodes a batch of classical vectors X of shape (N, D) into a batched QuantumState.

        Args:
            X: 2D array of shape (N, D).

        Returns:
            Batched QuantumState of shape (N, 2, ..., 2).
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        batch_size, d = X.shape

        if self.encoding == "amplitude":
            # Amplitude encoding: normalize vectors to unit length
            target_dim = 1 << self.n_qubits
            padded = np.zeros((batch_size, target_dim), dtype=np.complex128)
            limit = min(d, target_dim)
            padded[:, :limit] = X[:, :limit]
            
            # Vectorized L2 normalization across batch
            norms = np.linalg.norm(padded, axis=-1, keepdims=True)
            padded = padded / (norms + 1e-15)
            return QuantumState(
                padded,
                n_qubits=self.n_qubits,
                batch_size=batch_size,
                backend=self.backend
            )

        elif self.encoding == "angle":
            # Angle encoding: apply Ry(x_i) rotations to |0...0>
            state = QuantumState.zeros(
                n_qubits=self.n_qubits,
                batch_size=batch_size,
                backend=self.backend
            )
            # Apply rotations per feature up to n_qubits
            for q in range(min(d, self.n_qubits)):
                angles = X[:, q]
                # Batched single-qubit rotation
                # Cosine and sine arrays shaped (batch, 1, 1)
                c = np.cos(angles / 2.0)
                s = np.sin(angles / 2.0)
                # Form batched 2x2 rotation matrices: shape (batch, 2, 2)
                # Ry(theta) = [[c, -s], [s, c]]
                b_rot = np.zeros((batch_size, 2, 2), dtype=np.complex128)
                b_rot[:, 0, 0] = c
                b_rot[:, 0, 1] = -s
                b_rot[:, 1, 0] = s
                b_rot[:, 1, 1] = c

                state.apply_gate(b_rot, targets=q, inplace=True)
            return state

        elif self.encoding == "zz":
            # Quantum-inspired ZZ-feature map: H -> Rz(x_i) -> entangle -> Rz(x_i * x_j)
            state = QuantumState.zeros(
                n_qubits=self.n_qubits,
                batch_size=batch_size,
                backend=self.backend
            )
            state.apply_all(ops.H, inplace=True)
            
            # First order Rz
            for q in range(min(d, self.n_qubits)):
                angles = 2.0 * X[:, q]
                b_rz = np.zeros((batch_size, 2, 2), dtype=np.complex128)
                b_rz[:, 0, 0] = np.exp(-1.0j * angles / 2.0)
                b_rz[:, 1, 1] = np.exp(1.0j * angles / 2.0)
                state.apply_gate(b_rz, targets=q, inplace=True)

            # Entangling ZZ rotations: CNOT -> Rz(2 * (pi - x_i)(pi - x_j)) -> CNOT
            for q in range(min(d, self.n_qubits) - 1):
                state.cnot(q, q + 1, inplace=True)
                prod_angle = 2.0 * (np.pi - X[:, q]) * (np.pi - X[:, q + 1])
                b_rz = np.zeros((batch_size, 2, 2), dtype=np.complex128)
                b_rz[:, 0, 0] = np.exp(-1.0j * prod_angle / 2.0)
                b_rz[:, 1, 1] = np.exp(1.0j * prod_angle / 2.0)
                state.apply_gate(b_rz, targets=q + 1, inplace=True)
                state.cnot(q, q + 1, inplace=True)

            return state

        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")


class QuantumKernel:
    """Computes quantum kernel Gram matrices K(x_i, x_j) = |<psi(x_i) | psi(x_j)>|^2

    simultaneously across all pairs of samples using batched Einstein summation.
    """

    def __init__(
        self,
        feature_map: Optional[QuantumFeatureMap] = None,
        n_qubits: int = 4,
        encoding: Literal["angle", "amplitude", "zz"] = "angle",
        backend: Optional[str] = None
    ) -> None:
        self.feature_map = feature_map or QuantumFeatureMap(
            n_qubits=n_qubits, encoding=encoding, backend=backend
        )
        self.backend = backend

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes the complete kernel Gram matrix between X1 and X2 with zero Python sample loops."""
        state1 = self.feature_map.encode(X1)
        if X2 is None or X2 is X1:
            state2 = state1
        else:
            state2 = self.feature_map.encode(X2)

        # Get flat amplitude representations: (B1, dim) and (B2, dim)
        amps1 = state1.numpy().reshape(state1.batch_size, -1)
        amps2 = state2.numpy().reshape(state2.batch_size, -1)

        # Vectorized Gram matrix computation: K_ij = |amps1_i . conj(amps2_j)|^2
        # Zero loops: einsum calculates the full pairwise matrix contraction
        inner_matrix = np.einsum("ik,jk->ij", amps1, np.conj(amps2))
        return np.abs(inner_matrix) ** 2
