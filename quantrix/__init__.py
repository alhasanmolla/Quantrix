"""Quantrix: High-Performance Quantum-Inspired Parallel Matrix & Tensor Library.

Features:
- QuantumState master class managing memory layout, SIMD alignment, and tensor views.
- Vectorized, broadcasting, and Einstein summation backend operations with zero loops.
- Pluggable backends: NumPy, Numba JIT, PyTorch GPU/CPU, and native C extension hooks.
- Quantum-inspired transformations: QFT, Grover search, and parallel matrix operations.
- Foundational quantum machine learning: Quantum kernels, unitary linear layers, and PQCs.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from quantrix.core.state import QuantumState
from quantrix.core.memory import MemoryLayout
from quantrix.core import operators
from quantrix.backends.base import BackendManager, AbstractBackend
from quantrix.backends.numpy_backend import NumPyBackend
from quantrix.backends.numba_backend import NumbaBackend
from quantrix.backends.torch_backend import TorchBackend
from quantrix.backends.c_hooks import CHookBackend, CExtensionBridge
from quantrix.ops.transformations import (
    apply_hadamard_transform,
    grover_diffusion_operator,
    grover_search,
    quantum_parallel_matrix_multiply,
)
from quantrix.ops.measurements import (
    state_fidelity,
    density_matrix_fidelity,
    trace_distance,
    purity,
    bloch_vector,
    quantum_mutual_information,
)
from quantrix.ml.kernels import QuantumFeatureMap, QuantumKernel
from quantrix.ml.circuits import ParameterizedQuantumCircuit
from quantrix.ml.layers import QuantumLinear, TensorContractionLayer
from quantrix.ml.classifiers import QuantumKernelClassifier, VariationalQuantumClassifier

__version__ = "0.1.0"

__all__ = [
    "QuantumState",
    "MemoryLayout",
    "BackendManager",
    "AbstractBackend",
    "NumPyBackend",
    "NumbaBackend",
    "TorchBackend",
    "CHookBackend",
    "CExtensionBridge",
    "operators",
    "apply_hadamard_transform",
    "grover_diffusion_operator",
    "grover_search",
    "quantum_parallel_matrix_multiply",
    "state_fidelity",
    "density_matrix_fidelity",
    "trace_distance",
    "purity",
    "bloch_vector",
    "quantum_mutual_information",
    "QuantumFeatureMap",
    "QuantumKernel",
    "ParameterizedQuantumCircuit",
    "QuantumLinear",
    "TensorContractionLayer",
    "QuantumKernelClassifier",
    "VariationalQuantumClassifier",
]
