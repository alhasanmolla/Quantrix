"""Executable validation of all Quickstart examples from README.md."""

import numpy as np
import math
from quantrix import (
    QuantumState,
    BackendManager,
    operators as ops,
    QuantumKernelClassifier,
)


def run_quickstart_section1():
    print("[1] Superposition State Initialization and Gate Evolution...")
    state = QuantumState.superposition(n_qubits=4)
    assert math.isclose(state.norm(), 1.0, rel_tol=1e-6)
    assert state.dim == 16

    state.x(0)
    state.hadamard(2)
    state.cnot(control=0, target=1)

    probs = state.probabilities()
    assert math.isclose(np.sum(probs), 1.0, rel_tol=1e-6)
    print("    -> Passed.")


def run_quickstart_section2():
    print("[2] Entanglement, Partial Trace, and Von Neumann Entropy...")
    ghz = QuantumState.ghz_state(n_qubits=3)

    rho_0 = ghz.partial_trace(keep_qubits=[0])
    assert rho_0.shape == (2, 2)
    assert np.allclose(rho_0, 0.5 * np.eye(2))

    entropy = ghz.von_neumann_entropy(subsystem=[0])
    assert math.isclose(entropy, 1.0, rel_tol=1e-5)

    exp_z0 = ghz.expectation(ops.Z, targets=0)
    exp_zz = ghz.expectation(ops.kron(ops.Z, ops.Z), targets=[0, 1])
    assert math.isclose(float(np.real(exp_z0)), 0.0, abs_tol=1e-5)
    assert math.isclose(float(np.real(exp_zz)), 1.0, rel_tol=1e-5)
    print("    -> Passed.")


def run_quickstart_section3():
    print("[3] Hardware Backend Switching...")
    state = QuantumState.superposition(n_qubits=6, backend="numpy")
    
    if BackendManager.get_backend("numba").is_available:
        state_numba = state.to_backend("numba")
        assert state_numba.backend.name == "numba"

    if BackendManager.get_backend("torch").is_available:
        state_torch = state.to_backend("torch")
        assert state_torch.backend.name == "torch"
    print("    -> Passed.")


def run_quickstart_section4():
    print("[4] Quantum-Inspired Machine Learning Classification...")
    np.random.seed(42)
    X_train = np.random.uniform(-1.0, 1.0, size=(60, 2))
    y_train = np.where(np.linalg.norm(X_train, axis=1) > 0.5, 1, 0)

    clf = QuantumKernelClassifier(n_qubits=4, encoding="angle", alpha=1e-3)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_train, y_train)
    assert accuracy > 0.80
    print(f"    -> Passed with Accuracy: {accuracy * 100:.1f}%.")


if __name__ == "__main__":
    print("Validating Quickstart Suite:")
    run_quickstart_section1()
    run_quickstart_section2()
    run_quickstart_section3()
    run_quickstart_section4()
    print("All Quickstart examples validated successfully!")
