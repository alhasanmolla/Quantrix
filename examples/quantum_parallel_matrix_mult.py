"""Quantum-Inspired Parallel Processing and Superposition Matrix Multiplications.

Demonstrates:
1. Multi-state superposition initialization with zero loops.
2. Parallel matrix-matrix transformations mapped over quantum-inspired state tensors.
3. Entanglement creation, density matrix calculations, and Von Neumann entropy.
4. Quantum expectation values computed via Einstein summation.
"""

import numpy as np
from quantrix import QuantumState, operators as ops, quantum_parallel_matrix_multiply


def main():
    print("=" * 65)
    print(" Quantrix: Quantum-Inspired Parallel Matrix & State Processing")
    print("=" * 65)

    # 1. Initialize Equal Superposition across 4 virtual qubits (dimension 16)
    n_qubits = 4
    state = QuantumState.superposition(n_qubits=n_qubits)
    print(f"\n[+] Created Uniform Superposition State: {state}")
    print(f"    - State dimension: {state.dim}")
    print(f"    - Amplitudes shape: {state.amplitudes.shape}")
    print(f"    - L2 Norm: {state.norm():.6f}")

    # 2. Parallel Unitary Transformation: Apply Hadamard to all qubits
    print("\n[+] Applying Global Hadamard Superposition Transform...")
    state.apply_all(ops.H, inplace=True)
    print(f"    - Transformation completed with zero loops.")
    print(f"    - Ground state probability: {state.probabilities()[0]:.4f}")

    # 3. Create Entangled GHZ State across all 4 qubits
    print("\n[+] Synthesizing 4-Qubit GHZ Entangled State (|0000> + |1111>)/sqrt(2)...")
    ghz = QuantumState.ghz_state(n_qubits=4)
    probs = ghz.probabilities()
    print(f"    - P(|0000>): {probs[0]:.4f}")
    print(f"    - P(|1111>): {probs[15]:.4f}")
    print(f"    - P(other basis states): {np.sum(probs[1:15]):.4f}")

    # 4. Partial Trace and Entanglement Entropy
    entropy = ghz.von_neumann_entropy(subsystem=[0])
    print(f"\n[+] Entanglement Analysis:")
    print(f"    - Von Neumann Entropy S(qubit 0): {entropy:.4f} bit (Maximal Entanglement = 1.0)")

    # 5. Quantum Expectation Values using Einstein Summation
    exp_z0 = ghz.expectation(ops.Z, targets=0)
    exp_zz = ghz.expectation(ops.kron(ops.Z, ops.Z), targets=[0, 1])
    print(f"\n[+] Observables via Einsum Contraction:")
    print(f"    - <Z_0>: {exp_z0.real:.4f} (Individual qubit is completely random)")
    print(f"    - <Z_0 (x) Z_1>: {exp_zz.real:.4f} (Strict correlation)")

    # 6. Quantum-Inspired Accelerated Matrix Multiplication
    print("\n[+] Accelerated Parallel Matrix Contraction (Einsum):")
    A = np.random.randn(64, 128)
    B = np.random.randn(128, 32)
    C = quantum_parallel_matrix_multiply(A, B)
    expected_C = A @ B
    print(f"    - Multiplied ({A.shape}) x ({B.shape}) -> ({C.shape})")
    print(f"    - Numerical accuracy verified: {np.allclose(C, expected_C)}")


if __name__ == "__main__":
    main()
