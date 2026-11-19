"""Unit tests for QuantumState core capabilities."""

import pytest
import numpy as np
import math

from quantrix import QuantumState, operators as ops
from quantrix.core.memory import MemoryLayout


def test_state_initialization():
    # Test ground state |000>
    s0 = QuantumState.zeros(n_qubits=3)
    assert s0.n_qubits == 3
    assert s0.dim == 8
    assert not s0.is_batched
    assert math.isclose(s0.norm(), 1.0, rel_tol=1e-7)
    amps = s0.numpy().reshape(-1)
    assert np.isclose(amps[0], 1.0)
    assert np.allclose(amps[1:], 0.0)

    # Test superposition state
    s_sup = QuantumState.superposition(n_qubits=3)
    expected_amp = 1.0 / np.sqrt(8)
    assert np.allclose(s_sup.numpy().reshape(-1), expected_amp)
    assert math.isclose(s_sup.norm(), 1.0, rel_tol=1e-7)


def test_bell_state_and_entanglement():
    # Create Bell state |Phi+> = (|00> + |11>) / sqrt(2)
    bell = QuantumState.bell_state("phi_plus")
    assert bell.n_qubits == 2
    assert bell.dim == 4
    
    amps = bell.numpy().reshape(-1)
    inv_sqrt2 = 1.0 / np.sqrt(2)
    expected = np.array([inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128)
    assert np.allclose(amps, expected)

    # Reduced density matrix of qubit 0 should be maximally mixed: 0.5 * I
    rho_0 = bell.partial_trace(keep_qubits=[0])
    assert rho_0.shape == (2, 2)
    assert np.allclose(rho_0, 0.5 * np.eye(2))

    # Entanglement entropy of qubit 0 should be exactly 1.0 bit
    entropy = bell.von_neumann_entropy(subsystem=[0])
    assert math.isclose(entropy, 1.0, rel_tol=1e-5)


def test_ghz_state():
    ghz = QuantumState.ghz_state(n_qubits=4)
    assert ghz.n_qubits == 4
    assert ghz.dim == 16
    probs = ghz.probabilities()
    # Only |0000> and |1111> should have probability 0.5
    assert math.isclose(probs[0], 0.5, rel_tol=1e-5)
    assert math.isclose(probs[15], 0.5, rel_tol=1e-5)
    assert math.isclose(np.sum(probs), 1.0, rel_tol=1e-5)


def test_gates_and_evolution():
    s = QuantumState.zeros(n_qubits=1)
    # Apply Pauli X: |0> -> |1>
    s.x(0)
    assert np.allclose(s.probabilities(), [0.0, 1.0])

    # Apply Hadamard: |1> -> (|0> - |1>) / sqrt(2)
    s.hadamard(0)
    amps = s.numpy().reshape(-1)
    inv_sqrt2 = 1.0 / np.sqrt(2)
    assert np.allclose(amps, [inv_sqrt2, -inv_sqrt2])

    # Apply Z: changes phase of |1>
    s.z(0)
    assert np.allclose(s.numpy().reshape(-1), [inv_sqrt2, inv_sqrt2])


def test_inner_product_and_dunders():
    s0 = QuantumState.zeros(n_qubits=2)
    s1 = QuantumState.zeros(n_qubits=2)
    s1.x(0)  # |10>

    # Orthogonal states: <s0 | s1> == 0
    assert math.isclose(abs(s0 | s1), 0.0, abs_tol=1e-7)
    # Self inner product: <s0 | s0> == 1
    assert math.isclose(abs(s0 | s0), 1.0, rel_tol=1e-7)

    # Tensor product: |00> (x) |10> = |0010> (4 qubits)
    s_prod = s0 @ s1
    assert s_prod.n_qubits == 4
    assert s_prod.dim == 16


def test_batched_states():
    batch_size = 5
    n_qubits = 3
    s_batch = QuantumState.superposition(n_qubits=n_qubits, batch_size=batch_size)
    assert s_batch.is_batched
    assert s_batch.batch_size == batch_size
    assert s_batch.dim == 8

    # Apply gate across entire batch with zero loops
    s_batch.x(0)
    probs = s_batch.probabilities()
    assert probs.shape == (batch_size, 8)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
