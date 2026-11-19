"""Parameterized Quantum Circuits (PQC) and exact Parameter-Shift Gradients.

Implements variational quantum-inspired layers with analytic gradients computed via
the exact quantum parameter-shift rule without numerical finite differences.
"""

from __future__ import annotations
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np

from quantrix.core.state import QuantumState
from quantrix.core import operators as ops


class ParameterizedQuantumCircuit:
    """Variational Quantum-Inspired Circuit featuring alternating parameterized

    single-qubit rotations and entangling CNOT layers.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        backend: Optional[str] = None
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        # 2 rotation parameters per qubit per layer: Ry and Rz
        self.num_params = n_layers * n_qubits * 2
        self.params = np.random.uniform(0.0, 2.0 * np.pi, size=self.num_params)

    def forward(
        self,
        params: Optional[np.ndarray] = None,
        initial_state: Optional[QuantumState] = None
    ) -> QuantumState:
        """Executes circuit on initial state with given parameters."""
        p = self.params if params is None else params
        if initial_state is None:
            state = QuantumState.zeros(n_qubits=self.n_qubits, backend=self.backend)
        else:
            state = initial_state.copy()

        param_idx = 0
        for _ in range(self.n_layers):
            # Parameterized Ry rotations
            for q in range(self.n_qubits):
                theta_y = p[param_idx]
                param_idx += 1
                state.ry(q, theta_y, inplace=True)

            # Parameterized Rz rotations
            for q in range(self.n_qubits):
                theta_z = p[param_idx]
                param_idx += 1
                state.rz(q, theta_z, inplace=True)

            # Entangling CNOT ladder
            for q in range(self.n_qubits - 1):
                state.cnot(q, q + 1, inplace=True)
            if self.n_qubits > 2:
                state.cnot(self.n_qubits - 1, 0, inplace=True)

        return state

    def expectation(
        self,
        params: Optional[np.ndarray] = None,
        observable: Optional[np.ndarray] = None,
        target_qubit: int = 0,
        initial_state: Optional[QuantumState] = None
    ) -> float:
        """Computes expectation value <psi(params)| O |psi(params)>."""
        state = self.forward(params, initial_state=initial_state)
        op = ops.Z if observable is None else observable
        val = state.expectation(op, targets=target_qubit)
        return float(np.real(val))

    def compute_gradients_parameter_shift(
        self,
        params: Optional[np.ndarray] = None,
        observable: Optional[np.ndarray] = None,
        target_qubit: int = 0,
        initial_state: Optional[QuantumState] = None
    ) -> np.ndarray:
        """Computes exact analytical gradient vector using the Quantum Parameter-Shift Rule:

        dE/d_theta_k = 0.5 * (E(theta + pi/2 * e_k) - E(theta - pi/2 * e_k))
        """
        p = np.array(self.params if params is None else params, dtype=np.float64)
        shift = math.pi / 2.0
        grads = np.zeros_like(p)

        for i in range(len(p)):
            # Positive shift
            p_plus = p.copy()
            p_plus[i] += shift
            e_plus = self.expectation(p_plus, observable, target_qubit, initial_state)

            # Negative shift
            p_minus = p.copy()
            p_minus[i] -= shift
            e_minus = self.expectation(p_minus, observable, target_qubit, initial_state)

            # Exact analytic gradient
            grads[i] = 0.5 * (e_plus - e_minus)

        return grads
