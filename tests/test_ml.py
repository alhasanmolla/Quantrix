"""Unit tests for Quantum-inspired ML models, kernels, and layers."""

import pytest
import numpy as np
import math

from quantrix import (
    QuantumFeatureMap,
    QuantumKernel,
    ParameterizedQuantumCircuit,
    QuantumLinear,
    TensorContractionLayer,
    QuantumKernelClassifier,
)


def test_quantum_feature_map():
    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    qfm = QuantumFeatureMap(n_qubits=3, encoding="angle")
    state = qfm.encode(X)
    assert state.is_batched
    assert state.batch_size == 2
    assert state.n_qubits == 3
    # Norm of each state in the batch must be 1.0
    norms = state.norm()
    assert np.allclose(norms, 1.0)


def test_quantum_kernel_gram_matrix():
    X = np.random.uniform(0.0, np.pi, size=(6, 3))
    kernel = QuantumKernel(n_qubits=3, encoding="angle")
    K = kernel(X, X)

    assert K.shape == (6, 6)
    # Diagonal elements should be ~1.0 (<psi|psi> = 1)
    assert np.allclose(np.diag(K), 1.0, atol=1e-5)
    # Gram matrix must be symmetric: K = K.T
    assert np.allclose(K, K.T, atol=1e-5)
    # All kernel values between 0.0 and 1.0
    assert np.all(K >= -1e-6)
    assert np.all(K <= 1.0 + 1e-6)


def test_parameter_shift_rule():
    pqc = ParameterizedQuantumCircuit(n_qubits=2, n_layers=1)
    # Compute analytical parameter shift gradient
    grads = pqc.compute_gradients_parameter_shift()
    assert len(grads) == pqc.num_params

    # Verify against numerical central differences
    eps = 1e-5
    for i in range(min(2, len(grads))):
        p_plus = pqc.params.copy()
        p_plus[i] += eps
        e_plus = pqc.expectation(p_plus)

        p_minus = pqc.params.copy()
        p_minus[i] -= eps
        e_minus = pqc.expectation(p_minus)

        num_grad = (e_plus - e_minus) / (2.0 * eps)
        assert math.isclose(grads[i], num_grad, rel_tol=1e-3, abs_tol=1e-4)


def test_quantum_linear_orthogonality():
    ql = QuantumLinear(in_features=4, out_features=4, use_bias=False)
    W = ql.get_orthogonal_matrix()
    # W @ W.T should be identity
    assert np.allclose(W @ W.T, np.eye(4), atol=1e-5)

    X = np.random.randn(5, 4)
    Y = ql(X)
    # Orthogonal transformation preserves vector L2 norms: ||Y|| == ||X||
    norm_x = np.linalg.norm(X, axis=1)
    norm_y = np.linalg.norm(Y, axis=1)
    assert np.allclose(norm_x, norm_y, atol=1e-5)


def test_tensor_contraction_layer():
    tcl = TensorContractionLayer(tensor_modes_in=(2, 2), tensor_modes_out=(2, 2), rank=2)
    X = np.random.randn(8, 4)
    out = tcl(X)
    assert out.shape == (8, 4)


def test_quantum_kernel_classifier():
    np.random.seed(42)
    # Create simple 2D classification dataset
    X_pos = np.random.normal(loc=0.5, scale=0.2, size=(10, 2))
    X_neg = np.random.normal(loc=-0.5, scale=0.2, size=(10, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 10 + [0] * 10)

    clf = QuantumKernelClassifier(n_qubits=2, encoding="angle", alpha=0.01)
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert acc >= 0.90
