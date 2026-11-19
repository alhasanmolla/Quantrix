"""Master QuantumState Class for Quantrix.

Manages memory layout, high-performance tensor views, and execution dispatch
across NumPy, Numba, PyTorch, and C-extension backends.
"""

from __future__ import annotations
import math
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

from quantrix.core.memory import MemoryLayout
from quantrix.backends.base import AbstractBackend, BackendManager
from quantrix.core import operators as ops
from quantrix.ops.einsum_ops import generate_partial_trace_einsum


class QuantumState:
    """Master representation of a quantum-inspired state vector or batch of states.

    Features:
    - Zero-loop tensor and matrix execution using Einstein summation and broadcasting.
    - Automatic zero-copy view management between tensor and flat layouts.
    - Seamless runtime backend switching (NumPy, Numba, PyTorch GPU/CPU, C hooks).
    - Comprehensive support for single-qubit, multi-qubit, parameterized, and dynamic operators.
    """

    def __init__(
        self,
        data: Any,
        n_qubits: Optional[int] = None,
        batch_size: Optional[int] = None,
        dtype: Optional[Any] = None,
        backend: Optional[Union[str, AbstractBackend]] = None,
        normalize: bool = False,
        order: str = "C",
        device: str = "cpu"
    ) -> None:
        # Resolve backend
        if backend is None:
            self._backend = BackendManager.get_backend()
        elif isinstance(backend, str):
            self._backend = BackendManager.get_backend(backend)
        else:
            self._backend = backend

        # Convert data to backend array
        arr = self._backend.to_backend_array(data, dtype=dtype, device=device)
        np_arr = self._backend.to_numpy(arr)

        # Infer dimensions if not provided
        total_elements = np_arr.size
        if batch_size is None and np_arr.ndim > 1 and not (np_arr.ndim == 2 and np_arr.shape[0] == 1):
            # Check if leading dimension is batch
            if n_qubits is not None:
                dim_per_state = 1 << n_qubits
                if np_arr.shape[0] * dim_per_state == total_elements:
                    batch_size = np_arr.shape[0]

        is_batched = batch_size is not None and batch_size > 0
        dim = (total_elements // batch_size) if is_batched else total_elements

        if n_qubits is None:
            # Check if dimension is power of 2
            if dim > 0 and (dim & (dim - 1)) == 0:
                n_qubits = int(math.log2(dim))
            else:
                raise ValueError(
                    f"State dimension {dim} is not a power of 2; n_qubits must be explicitly provided."
                )

        target_dtype = np_arr.dtype
        self._layout = MemoryLayout(
            n_qubits=n_qubits,
            batch_size=batch_size,
            dtype=target_dtype,
            order=order,
            device=device
        )

        # Reshape array to tensor view (e.g. (2, 2, ..., 2) or (batch, 2, 2, ..., 2))
        tensor_shape = self._layout.tensor_shape
        if hasattr(arr, "reshape"):
            self._data = arr.reshape(tensor_shape)
        else:
            self._data = self._backend.to_backend_array(
                np_arr.reshape(tensor_shape), dtype=target_dtype, device=device
            )

        if normalize:
            self.normalize_()

    # --- Properties ---

    @property
    def n_qubits(self) -> int:
        return self._layout.n_qubits  # type: ignore

    @property
    def dim(self) -> int:
        return self._layout.state_dim

    @property
    def batch_size(self) -> Optional[int]:
        return self._layout.batch_size

    @property
    def is_batched(self) -> bool:
        return self._layout.is_batched

    @property
    def dtype(self) -> Any:
        return self._layout.dtype

    @property
    def layout(self) -> MemoryLayout:
        return self._layout

    @property
    def backend(self) -> AbstractBackend:
        return self._backend

    @property
    def tensor(self) -> Any:
        """Zero-copy multi-qubit tensor view (2, 2, ..., 2) or (batch, 2, ...)."""
        return self._data

    @property
    def amplitudes(self) -> Any:
        """Flat amplitude vector or batch of vectors: shape (2^n,) or (batch, 2^n)."""
        flat_shape = self._layout.flat_shape
        if hasattr(self._data, "reshape"):
            return self._data.reshape(flat_shape)
        return self._backend.to_backend_array(
            self.numpy().reshape(flat_shape)
        )

    # --- Factory Initializers ---

    @classmethod
    def zeros(
        cls,
        n_qubits: int,
        batch_size: Optional[int] = None,
        dtype: Any = np.complex128,
        backend: Optional[Union[str, AbstractBackend]] = None,
        device: str = "cpu"
    ) -> QuantumState:
        """Initializes ground computational basis state |0...0>."""
        dim = 1 << n_qubits
        if batch_size is not None:
            data = np.zeros((batch_size, dim), dtype=dtype)
            data[:, 0] = 1.0
        else:
            data = np.zeros(dim, dtype=dtype)
            data[0] = 1.0
        return cls(data, n_qubits=n_qubits, batch_size=batch_size, dtype=dtype, backend=backend, device=device)

    @classmethod
    def superposition(
        cls,
        n_qubits: int,
        batch_size: Optional[int] = None,
        dtype: Any = np.complex128,
        backend: Optional[Union[str, AbstractBackend]] = None,
        device: str = "cpu"
    ) -> QuantumState:
        """Initializes equal superposition state 1/sqrt(2^n) * sum |i>."""
        dim = 1 << n_qubits
        norm_factor = 1.0 / math.sqrt(dim)
        shape = (batch_size, dim) if batch_size is not None else (dim,)
        data = np.full(shape, norm_factor, dtype=dtype)
        return cls(data, n_qubits=n_qubits, batch_size=batch_size, dtype=dtype, backend=backend, device=device)

    @classmethod
    def from_array(
        cls,
        array: Any,
        n_qubits: Optional[int] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        backend: Optional[Union[str, AbstractBackend]] = None,
        device: str = "cpu"
    ) -> QuantumState:
        """Creates a QuantumState from an existing array or tensor."""
        return cls(
            array,
            n_qubits=n_qubits,
            batch_size=batch_size,
            backend=backend,
            normalize=normalize,
            device=device
        )

    @classmethod
    def random(
        cls,
        n_qubits: int,
        batch_size: Optional[int] = None,
        backend: Optional[Union[str, AbstractBackend]] = None,
        seed: Optional[int] = None,
        dtype: Any = np.complex128,
        device: str = "cpu"
    ) -> QuantumState:
        """Initializes random normalized quantum state vector."""
        rng = np.random.default_rng(seed)
        dim = 1 << n_qubits
        shape = (batch_size, dim) if batch_size is not None else (dim,)
        real_part = rng.standard_normal(shape)
        imag_part = rng.standard_normal(shape)
        data = (real_part + 1.0j * imag_part).astype(dtype)
        return cls(
            data,
            n_qubits=n_qubits,
            batch_size=batch_size,
            dtype=dtype,
            backend=backend,
            normalize=True,
            device=device
        )

    @classmethod
    def bell_state(
        cls,
        state: str = "phi_plus",
        backend: Optional[Union[str, AbstractBackend]] = None
    ) -> QuantumState:
        """Initializes standard 2-qubit Bell states: 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'."""
        qs = cls.zeros(n_qubits=2, backend=backend)
        qs.hadamard(0)
        qs.cnot(0, 1)
        if state == "phi_minus":
            qs.z(0)
        elif state == "psi_plus":
            qs.x(1)
        elif state == "psi_minus":
            qs.z(0)
            qs.x(1)
        return qs

    @classmethod
    def ghz_state(
        cls,
        n_qubits: int = 3,
        backend: Optional[Union[str, AbstractBackend]] = None
    ) -> QuantumState:
        """Creates an N-qubit Greenberger-Horne-Zeilinger (GHZ) entangled state."""
        qs = cls.zeros(n_qubits=n_qubits, backend=backend)
        qs.hadamard(0)
        for i in range(n_qubits - 1):
            qs.cnot(i, i + 1)
        return qs

    # --- Evolution and Gate Applications ---

    def apply_gate(
        self,
        gate_matrix: Any,
        targets: Union[int, Sequence[int]],
        inplace: bool = False
    ) -> QuantumState:
        """Applies single or multi-qubit unitary gate matrix via vectorized einsum contraction."""
        target_list = [targets] if isinstance(targets, int) else list(targets)
        new_tensor = self._backend.apply_gate(
            state_tensor=self._data,
            gate_matrix=gate_matrix,
            target_qubits=target_list,
            n_qubits=self.n_qubits,
            is_batched=self.is_batched
        )

        if inplace:
            self._data = new_tensor
            return self

        return QuantumState(
            new_tensor,
            n_qubits=self.n_qubits,
            batch_size=self.batch_size,
            backend=self._backend,
            dtype=self.dtype,
            order=self._layout.order,
            device=self._layout.device
        )

    def apply_all(self, gate_1q: Any, inplace: bool = False) -> QuantumState:
        """Applies a single-qubit gate to ALL qubits simultaneously with vectorized einsum."""
        state = self if inplace else self.copy()
        for q in range(self.n_qubits):
            state.apply_gate(gate_1q, targets=q, inplace=True)
        return state

    # Convenience Gates
    def hadamard(self, target: int, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.H, targets=target, inplace=inplace)

    def x(self, target: int, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.X, targets=target, inplace=inplace)

    def y(self, target: int, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.Y, targets=target, inplace=inplace)

    def z(self, target: int, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.Z, targets=target, inplace=inplace)

    def s(self, target: int, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.S, targets=target, inplace=inplace)

    def t(self, target: int, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.T, targets=target, inplace=inplace)

    def rx(self, target: int, theta: float, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.rx(theta), targets=target, inplace=inplace)

    def ry(self, target: int, theta: float, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.ry(theta), targets=target, inplace=inplace)

    def rz(self, target: int, theta: float, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.rz(theta), targets=target, inplace=inplace)

    def cnot(self, control: int, target: int, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.CNOT, targets=[control, target], inplace=inplace)

    def cz(self, control: int, target: int, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.CZ, targets=[control, target], inplace=inplace)

    def swap(self, q1: int, q2: int, inplace: bool = True) -> QuantumState:
        return self.apply_gate(ops.SWAP, targets=[q1, q2], inplace=inplace)

    def qft(self, inplace: bool = False) -> QuantumState:
        """Applies Quantum Fourier Transform matrix to entire state space."""
        qft_mat = ops.qft_matrix(self.n_qubits)
        all_targets = list(range(self.n_qubits))
        return self.apply_gate(qft_mat, targets=all_targets, inplace=inplace)

    # --- State Normalization & Inner Products ---

    def norm(self) -> Union[float, np.ndarray]:
        """Calculates L2 Euclidean norm."""
        val = self._backend.norm(self.amplitudes, axis=-1 if self.is_batched else None)
        return self._backend.to_numpy(val)

    def normalize_(self) -> QuantumState:
        """Normalizes state amplitudes to unit norm in-place."""
        norms = self.norm()
        if self.is_batched:
            # Reshape norm for broadcasting
            if isinstance(norms, np.ndarray):
                norms = norms.reshape((-1,) + (1,) * self.n_qubits)
            self._data = self._data / (norms + 1e-15)
        else:
            self._data = self._data / (norms + 1e-15)
        return self

    def inner(self, other: QuantumState) -> Union[complex, np.ndarray]:
        """Computes inner product <self | other>."""
        if self.n_qubits != other.n_qubits:
            raise ValueError(f"State qubits mismatch: {self.n_qubits} vs {other.n_qubits}")
        res = self._backend.inner_product(self._data, other._data, is_batched=self.is_batched)
        return self._backend.to_numpy(res)

    # --- Measurements and Observables ---

    def probabilities(self) -> np.ndarray:
        """Computes probabilities |c_i|^2 across all basis states."""
        amps = self.numpy().reshape(self._layout.flat_shape)
        return np.abs(amps) ** 2

    def sample(self, shots: int = 1024, seed: Optional[int] = None) -> np.ndarray:
        """Vectorized multinomial sampling of measurement outcomes."""
        probs = self.probabilities()
        rng = np.random.default_rng(seed)
        dim = self.dim
        if not self.is_batched:
            p = probs / np.sum(probs)
            return rng.choice(dim, size=shots, p=p)
        
        # Batched sampling
        norm_probs = probs / np.sum(probs, axis=-1, keepdims=True)
        # Vectorized choice per batch row
        samples = np.array([
            rng.choice(dim, size=shots, p=norm_probs[b])
            for b in range(self.batch_size)  # type: ignore
        ])
        return samples

    def expectation(
        self,
        operator: Any,
        targets: Optional[Union[int, Sequence[int]]] = None
    ) -> Union[float, complex, np.ndarray]:
        """Computes expectation value <psi| O |psi> using vectorized einsum."""
        if targets is None:
            target_list = list(range(self.n_qubits))
        elif isinstance(targets, int):
            target_list = [targets]
        else:
            target_list = list(targets)

        val = self._backend.expectation_value(
            self._data, operator, target_list, self.n_qubits, is_batched=self.is_batched
        )
        return self._backend.to_numpy(val)

    def density_matrix(self) -> np.ndarray:
        """Computes density operator rho = |psi><psi| or batched density matrices."""
        amps = self.numpy().reshape(self._layout.flat_shape)
        if not self.is_batched:
            return np.outer(amps, np.conj(amps))
        # Batched outer product
        return np.einsum("bi,bj->bij", amps, np.conj(amps))

    def partial_trace(self, keep_qubits: Sequence[int]) -> np.ndarray:
        """Traces out unselected qubits, returning reduced density matrix of keep_qubits."""
        n = self.n_qubits
        keep = list(keep_qubits)
        for q in keep:
            if not (0 <= q < n):
                raise ValueError(f"Qubit {q} out of range [0, {n - 1}]")

        # Full density tensor
        amps_tensor = self.numpy().reshape(self._layout.tensor_shape)
        if not self.is_batched:
            rho_tensor = np.tensordot(amps_tensor, np.conj(amps_tensor), axes=0)
            einsum_str = generate_partial_trace_einsum(keep, n, is_batched=False)
            reduced_tensor = np.einsum(einsum_str, rho_tensor)
            k = len(keep)
            return reduced_tensor.reshape(1 << k, 1 << k)
        
        # Batched partial trace
        rho_tensor = np.einsum("b...,b...->b......", amps_tensor, np.conj(amps_tensor))
        einsum_str = generate_partial_trace_einsum(keep, n, is_batched=True)
        reduced_tensor = np.einsum(einsum_str, rho_tensor)
        k = len(keep)
        return reduced_tensor.reshape(self.batch_size, 1 << k, 1 << k)  # type: ignore

    def von_neumann_entropy(self, subsystem: Sequence[int]) -> Union[float, np.ndarray]:
        """Calculates Von Neumann entanglement entropy S(rho_A) = -Tr(rho_A * log2(rho_A))."""
        rho_red = self.partial_trace(subsystem)
        
        def _calc_entropy(rho: np.ndarray) -> float:
            eigvals = np.linalg.eigvalsh(rho)
            eigvals = np.maximum(eigvals.real, 0.0)
            eigvals = eigvals[eigvals > 1e-12]
            return float(-np.sum(eigvals * np.log2(eigvals)))

        if not self.is_batched:
            return _calc_entropy(rho_red)
        return np.array([_calc_entropy(rho_red[b]) for b in range(self.batch_size)])  # type: ignore

    # --- Tensor Product and Linear Algebra Overloads ---

    def tensor_product(self, other: QuantumState) -> QuantumState:
        """Forms composite quantum state |self> (x) |other> with dimension 2^(n1 + n2)."""
        a_amps = self.numpy().reshape(self._layout.flat_shape)
        b_amps = other.numpy().reshape(other._layout.flat_shape)

        if not self.is_batched and not other.is_batched:
            comp_amps = np.kron(a_amps, b_amps)
            new_batch = None
        elif self.is_batched and other.is_batched:
            if self.batch_size != other.batch_size:
                raise ValueError("Batch size mismatch for tensor product.")
            comp_amps = np.einsum("bi,bj->bij", a_amps, b_amps).reshape(self.batch_size, -1)
            new_batch = self.batch_size
        else:
            raise ValueError("Cannot tensor product batched and non-batched states without explicit broadcast.")

        return QuantumState(
            comp_amps,
            n_qubits=self.n_qubits + other.n_qubits,
            batch_size=new_batch,
            backend=self._backend,
            device=self._layout.device
        )

    def to_backend(self, backend: Union[str, AbstractBackend]) -> QuantumState:
        """Transitions state execution to another backend."""
        target_backend = BackendManager.get_backend(backend) if isinstance(backend, str) else backend
        return QuantumState(
            self.numpy(),
            n_qubits=self.n_qubits,
            batch_size=self.batch_size,
            backend=target_backend,
            dtype=self.dtype,
            order=self._layout.order,
            device=self._layout.device
        )

    def numpy(self) -> np.ndarray:
        """Exports state data to standard NumPy ndarray."""
        return self._backend.to_numpy(self._data)

    def copy(self) -> QuantumState:
        """Creates a deep copy of the QuantumState."""
        return QuantumState(
            self.numpy().copy(),
            n_qubits=self.n_qubits,
            batch_size=self.batch_size,
            backend=self._backend,
            dtype=self.dtype,
            order=self._layout.order,
            device=self._layout.device
        )

    # --- Operator Dunders ---

    def __or__(self, other: QuantumState) -> Union[complex, np.ndarray]:
        """Operator overload for inner product <self | other>."""
        return self.inner(other)

    def __matmul__(self, other: Any) -> Any:
        """Operator overload for tensor product or gate application."""
        if isinstance(other, QuantumState):
            return self.tensor_product(other)
        elif isinstance(other, np.ndarray):
            # Matrix multiplication
            return self.apply_gate(other, targets=list(range(self.n_qubits)))
        raise TypeError(f"Unsupported operand type for @: {type(other)}")

    def __add__(self, other: QuantumState) -> QuantumState:
        """Linear superposition addition: |psi1> + |psi2>."""
        res_data = self._data + other._data
        return QuantumState(
            res_data,
            n_qubits=self.n_qubits,
            batch_size=self.batch_size,
            backend=self._backend,
            dtype=self.dtype
        )

    def __sub__(self, other: QuantumState) -> QuantumState:
        """Linear superposition subtraction: |psi1> - |psi2>."""
        res_data = self._data - other._data
        return QuantumState(
            res_data,
            n_qubits=self.n_qubits,
            batch_size=self.batch_size,
            backend=self._backend,
            dtype=self.dtype
        )

    def __mul__(self, scalar: Union[float, complex, int]) -> QuantumState:
        """Scalar amplitude scaling."""
        return QuantumState(
            self._data * scalar,
            n_qubits=self.n_qubits,
            batch_size=self.batch_size,
            backend=self._backend,
            dtype=self.dtype
        )

    def __rmul__(self, scalar: Union[float, complex, int]) -> QuantumState:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        batch_info = f", batch_size={self.batch_size}" if self.is_batched else ""
        return (
            f"QuantumState(qubits={self.n_qubits}, dim={self.dim}{batch_info}, "
            f"backend='{self._backend.name}', dtype={self.dtype})"
        )
