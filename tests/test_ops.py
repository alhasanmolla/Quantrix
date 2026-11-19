"""Unit tests for operations, dynamic einsum, and quantum metrics."""

import pytest
import numpy as np
import math

from quantrix import (
    QuantumState,
    operators as ops,
    state_fidelity,
    purity,
    bloch_vector,
    quantum_parallel_matrix_multiply,
    grover_search,
)
from quantrix.ops.einsum_ops import generate_gate_einsum


def test_dynamic_gate_einsum_generation():
    # 1-qubit gate on qubit 1 of 3 qubits
    subscript, shape = generate_gate_einsum(target_qubits=[1], n_qubits=3, is_batched=False)
    assert shape == (2, 2)
    # Target qubit 1 has input char 'b' and output char 'd'
    assert subscript == "db,abc->adc"

    # Batched 2-qubit gate on qubits (0, 2) of 4 qubits
    subscript, shape = generate_gate_einsum(target_qubits=[0, 2], n_qubits=4, is_batched=True)
    assert shape == (2, 2, 2, 2)
    assert subscript.startswith("efac,...abcd->...ebfd")


def test_quantum_parallel_matrix_multiply():
    np.random.seed(42)
    A = np.random.randn(8, 12)
    B = np.random.randn(12, 16)

    C_actual = quantum_parallel_matrix_multiply(A, B)
    C_expected = A @ B
    assert np.allclose(C_actual, C_expected)


def test_state_fidelity_and_purity():
    s1 = QuantumState.zeros(2)
    s2 = QuantumState.zeros(2)
    assert math.isclose(state_fidelity(s1, s2), 1.0, rel_tol=1e-6)

    s2.x(0)
    assert math.isclose(state_fidelity(s1, s2), 0.0, abs_tol=1e-6)
    assert math.isclose(purity(s1), 1.0, rel_tol=1e-6)


def test_bloch_vector():
    s = QuantumState.zeros(1)
    rx, ry, rz = bloch_vector(s)
    # Ground state |0> points to north pole (0, 0, 1)
    assert math.isclose(rx, 0.0, abs_tol=1e-6)
    assert math.isclose(ry, 0.0, abs_tol=1e-6)
    assert math.isclose(rz, 1.0, rel_tol=1e-6)

    s.hadamard(0)
    rx, ry, rz = bloch_vector(s)
    # State |+> points along +X axis (1, 0, 0)
    assert math.isclose(rx, 1.0, rel_tol=1e-6)
    assert math.isclose(ry, 0.0, abs_tol=1e-6)
    assert math.isclose(rz, 0.0, abs_tol=1e-6)


def test_grover_amplitude_amplification():
    # 3-qubit search: 8 items. Mark target index 5 (binary 101)
    n_qubits = 3
    target_idx = 5
    
    # 2 Grover iterations are optimal for N=8
    final_state = grover_search(
        oracle_fn=lambda idx: idx == target_idx,
        n_qubits=n_qubits,
        num_iterations=2
    )

    probs = final_state.probabilities()
    # Target index 5 should have high probability (> 90%)
    assert probs[target_idx] > 0.90
    assert np.argmax(probs) == target_idx
