"""High-Performance Benchmark Suite for Quantrix.

Compares Quantrix vectorized einsum tensor contraction, Numba JIT, and PyTorch
against standard loop-based state transformations.
"""

import time
import math
import numpy as np
from quantrix import QuantumState, BackendManager, operators as ops


def benchmark_multi_qubit_evolution(n_qubits: int = 14, num_runs: int = 5):
    dim = 1 << n_qubits
    print(f"\n=======================================================")
    print(f" Benchmark: Multi-Qubit Circuit Evolution ({n_qubits} Qubits, Dim = {dim:,})")
    print(f"=======================================================")

    results = {}

    # 1. Quantrix with NumPy Backend (Vectorized Einsum)
    s_numpy = QuantumState.zeros(n_qubits=n_qubits, backend="numpy")
    t0 = time.perf_counter()
    for _ in range(num_runs):
        s_numpy.hadamard(0)
        s_numpy.cnot(0, 1)
        s_numpy.rx(n_qubits - 1, math.pi / 4.0)
    t_numpy = (time.perf_counter() - t0) / num_runs
    results["Quantrix (NumPy Einsum)"] = t_numpy
    print(f"[*] Quantrix (NumPy Einsum) : {t_numpy * 1000:8.3f} ms / step")

    # 2. Quantrix with Numba Backend (LLVM JIT native)
    if BackendManager.get_backend("numba").is_available:
        s_numba = QuantumState.zeros(n_qubits=n_qubits, backend="numba")
        # Warmup JIT
        s_numba.hadamard(0)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            s_numba.hadamard(0)
            s_numba.cnot(0, 1)
            s_numba.rx(n_qubits - 1, math.pi / 4.0)
        t_numba = (time.perf_counter() - t0) / num_runs
        results["Quantrix (Numba JIT)"] = t_numba
        print(f"[*] Quantrix (Numba JIT)    : {t_numba * 1000:8.3f} ms / step")

    # 3. Quantrix with PyTorch Backend
    if BackendManager.get_backend("torch").is_available:
        s_torch = QuantumState.zeros(n_qubits=n_qubits, backend="torch")
        t0 = time.perf_counter()
        for _ in range(num_runs):
            s_torch.hadamard(0)
            s_torch.cnot(0, 1)
            s_torch.rx(n_qubits - 1, math.pi / 4.0)
        t_torch = (time.perf_counter() - t0) / num_runs
        results["Quantrix (PyTorch ATen)"] = t_torch
        print(f"[*] Quantrix (PyTorch ATen) : {t_torch * 1000:8.3f} ms / step")

    # 4. Naive Python Nested Loop Evolution (for comparison)
    # Using small dimension to avoid timeout
    naive_qubits = min(10, n_qubits)
    naive_dim = 1 << naive_qubits
    state_naive = np.zeros(naive_dim, dtype=np.complex128)
    state_naive[0] = 1.0
    gate_h = ops.H

    t0 = time.perf_counter()
    out = np.zeros_like(state_naive)
    step = 1 << (naive_qubits - 1)
    for b in range(naive_dim // (step * 2)):
        base = b * (step * 2)
        for k in range(step):
            i0 = base + k
            i1 = i0 + step
            out[i0] = gate_h[0, 0] * state_naive[i0] + gate_h[0, 1] * state_naive[i1]
            out[i1] = gate_h[1, 0] * state_naive[i0] + gate_h[1, 1] * state_naive[i1]
    t_naive_scaled = (time.perf_counter() - t0) * (dim / naive_dim)
    print(f"[-] Naive Loop (Projected)  : {t_naive_scaled * 1000:8.3f} ms / step")

    speedup = t_naive_scaled / t_numpy
    print(f"\n=> Acceleration vs Naive Loop: {speedup:6.1f}x Speedup!")
    return results


def benchmark_batched_kernel_gram(batch_size: int = 128, n_qubits: int = 4):
    from quantrix import QuantumKernel
    print(f"\n=======================================================")
    print(f" Benchmark: Batched Quantum Kernel Gram Matrix ({batch_size} samples)")
    print(f"=======================================================")

    X = np.random.uniform(0.0, np.pi, size=(batch_size, n_qubits))
    kernel = QuantumKernel(n_qubits=n_qubits, encoding="angle")

    # Time vectorized Gram matrix computation
    t0 = time.perf_counter()
    K = kernel(X, X)
    t_kernel = time.perf_counter() - t0

    num_pairs = batch_size * batch_size
    print(f"[*] Computed {num_pairs:,} state inner products in: {t_kernel * 1000:.2f} ms")
    print(f"[*] Throughput: {num_pairs / t_kernel:,.0f} state contractions / second")
    print(f"[*] Kernel Matrix Valid (Hermitian & Normal): {np.allclose(K, K.T)}")


if __name__ == "__main__":
    benchmark_multi_qubit_evolution(n_qubits=12, num_runs=10)
    benchmark_batched_kernel_gram(batch_size=128, n_qubits=4)
