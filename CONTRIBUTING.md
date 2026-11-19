# Contributing to Quantrix

We welcome contributions to Quantrix! This guide explains how to set up your development environment, implement new operators or backends, interface native C-extensions, and submit pull requests.

---

## 1. Development Setup

Clone the repository and install in editable mode with development and acceleration dependencies:

```bash
git clone https://github.com/quantrix/quantrix.git
cd quantrix
pip install -e .[acceleration,dev]
```

Verify your installation by running the test suite:

```bash
python -m pytest tests/ -v
```

---

## 2. Project Architecture

The library is organized as follows:

```
quantrix/
├── core/            # QuantumState master representation, MemoryLayout, operators
├── backends/        # NumPy, Numba JIT, PyTorch, and C-extension bridge
├── ops/             # Dynamic einsum generator, QFT, Grover, measurements
└── ml/              # Quantum kernels, parameterized circuits, orthogonal layers
```

---

## 3. Implementing a New Backend

All execution backends inherit from `quantrix.backends.base.AbstractBackend`. To add a new backend (e.g. CuPy, JAX, or Triton):

1. Create `quantrix/backends/your_backend.py`.
2. Subclass `AbstractBackend` and implement:
   - `einsum(subscripts, *operands)`
   - `matmul(a, b)`
   - `tensordot(a, b, axes)`
   - `apply_gate(state_tensor, gate_matrix, target_qubits, n_qubits, is_batched)`
   - `inner_product(state_a, state_b, is_batched)`
   - `expectation_value(state_tensor, operator, target_qubits, n_qubits, is_batched)`
   - `to_backend_array(array, dtype, device)`
   - `to_numpy(array)`
3. Register the backend in `quantrix/backends/__init__.py`:
   ```python
   BackendManager.register_backend(YourBackend())
   ```

---

## 4. Writing C-Extensions and GPU Hooks

Quantrix provides a ctypes bridge (`quantrix.backends.c_hooks.CExtensionBridge`) to call compiled native shared libraries (`.so`, `.dll`, or `.dylib`):

```python
from quantrix.backends.c_hooks import CHookBackend

c_backend = CHookBackend()
# Load precompiled shared library:
c_backend.bridge.load_library("/path/to/custom_kernels.so")

# Or register a function hook directly:
c_backend.bridge.register_hook("matmul", my_ctypes_func)
```

The C-API expects standard C-contiguous pointers for zero-copy execution:
```c
int quantrix_c_matmul(
    const void* a,
    const void* b,
    void* c,
    int64_t m,
    int64_t n,
    int64_t k,
    int is_complex
);
```

---

## 5. Testing and Validation

All contributions must include unit tests under `tests/`.

- Keep tests fast and deterministic.
- Test both batched and unbatched configurations.
- Verify numerical consistency across backends.

Run tests:
```bash
python -m pytest tests/
```

Run performance benchmarks:
```bash
python examples/high_perf_benchmark.py
```

---

## 6. Pull Request Process

1. Fork the repo and create a feature branch (`git checkout -b feature/my-feature`).
2. Implement your changes following PEP 8 conventions and type annotations.
3. Ensure all tests and benchmarks pass locally.
4. Open a Pull Request with a clear summary of the changes and performance implications.
