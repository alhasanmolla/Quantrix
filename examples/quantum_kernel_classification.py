"""Quantum-Inspired Machine Learning: Non-Linear Classification.

Demonstrates:
1. Embedding classical 2D non-linear points into high-dimensional quantum Hilbert states.
2. Computing the pairwise Quantum Kernel Gram matrix with zero sample loops using Einstein summation.
3. Training and evaluating a QuantumKernelClassifier on concentric non-linear circles.
"""

import math
import numpy as np
from quantrix import QuantumKernelClassifier, QuantumKernel, QuantumFeatureMap


def generate_concentric_circles(n_samples: int = 120, noise: float = 0.05, seed: int = 42):
    """Generates 2D concentric circles dataset (non-linearly separable in 2D)."""
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2

    # Inner circle: radius ~ 0.5
    angles_inner = rng.uniform(0.0, 2.0 * math.pi, n_half)
    r_inner = 0.5 + rng.normal(0.0, noise, n_half)
    X_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    y_inner = np.zeros(n_half, dtype=int)

    # Outer circle: radius ~ 1.0
    angles_outer = rng.uniform(0.0, 2.0 * math.pi, n_samples - n_half)
    r_outer = 1.0 + rng.normal(0.0, noise, n_samples - n_half)
    X_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    y_outer = np.ones(n_samples - n_half, dtype=int)

    X = np.vstack([X_inner, X_outer])
    y = np.concatenate([y_inner, y_outer])

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def main():
    print("=" * 65)
    print(" Quantrix: Quantum-Inspired Machine Learning Classification")
    print("=" * 65)

    # 1. Generate data
    X, y = generate_concentric_circles(n_samples=160, noise=0.06, seed=42)
    split = 120
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print(f"\n[+] Dataset generated:")
    print(f"    - Training samples : {len(X_train)}")
    print(f"    - Test samples     : {len(X_test)}")
    print(f"    - Problem type     : Concentric Non-Linear Circles")

    # 2. Quantum Feature Mapping and Kernel
    n_qubits = 4
    print(f"\n[+] Initializing Quantum Kernel with {n_qubits} Qubits...")
    kernel = QuantumKernel(n_qubits=n_qubits, encoding="angle")

    # 3. Fit Quantum Kernel Classifier
    clf = QuantumKernelClassifier(n_qubits=n_qubits, encoding="angle", alpha=1e-3)
    print("\n[+] Training QuantumKernelClassifier via batched einsum...")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"\n[+] Results:")
    print(f"    - Training Accuracy : {train_acc * 100:.2f}%")
    print(f"    - Test Accuracy     : {test_acc * 100:.2f}%")
    print(f"    - Kernel separates non-linear data seamlessly via quantum superposition!")


if __name__ == "__main__":
    main()
