"""Unit tests verifying multi-backend consistency and hardware execution hooks."""

import pytest
import numpy as np
import math

from quantrix import QuantumState, BackendManager, operators as ops
from quantrix.backends.c_hooks import CHookBackend


def test_registered_backends():
    backends = BackendManager.list_backends()
    assert "numpy" in backends
    assert backends["numpy"] is True
    assert "c_hook" in backends


def test_backend_consistency():
    # Run identical quantum operations on numpy, numba, and torch backends
    for b_name in ["numpy", "numba", "torch"]:
        backend = BackendManager.get_backend(b_name)
        if not backend.is_available:
            continue

        s = QuantumState.zeros(n_qubits=3, backend=b_name)
        s.hadamard(0)
        s.cnot(0, 1)
        s.rx(2, theta=math.pi / 4.0)

        # Norm must be 1.0
        assert math.isclose(s.norm(), 1.0, rel_tol=1e-5)

        # Expectation of Z on qubit 0
        exp_z0 = s.expectation(ops.Z, targets=0)
        # For Bell pair (|00> + |11>)/sqrt(2), <Z0> = 0.5 * (1 - 1) = 0.0
        assert math.isclose(float(np.real(exp_z0)), 0.0, abs_tol=1e-5)


def test_backend_transition():
    s_numpy = QuantumState.superposition(n_qubits=3, backend="numpy")
    s_numpy.z(1)

    # Transition to Torch
    if BackendManager.get_backend("torch").is_available:
        s_torch = s_numpy.to_backend("torch")
        assert s_torch.backend.name == "torch"
        assert np.allclose(s_numpy.numpy(), s_torch.numpy())

    # Transition to Numba
    if BackendManager.get_backend("numba").is_available:
        s_numba = s_numpy.to_backend("numba")
        assert s_numba.backend.name == "numba"
        assert np.allclose(s_numpy.numpy(), s_numba.numpy())


def test_c_hook_registration():
    c_backend = CHookBackend()
    assert c_backend.name == "c_hook"

    # Register a custom C-style hook
    def dummy_matmul_hook(ptr_a, ptr_b, ptr_c, m, n, k, is_c):
        return 0

    c_backend.bridge.register_hook("matmul", dummy_matmul_hook)
    assert "matmul" in c_backend.bridge._hooks
