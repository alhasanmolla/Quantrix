"""Memory layout and tensor structure management for Quantrix.

Manages contiguous memory buffers, alignment, memory orders (C-contiguous vs Fortran),
and zero-copy view transformations between flattened statevectors, multi-qubit
tensor representations, and batched density matrices.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union
import numpy as np


@dataclass(frozen=True)
class MemoryLayout:
    """Encapsulates memory layout, stride properties, and shape representations

    for quantum-inspired states and matrices.
    """
    n_qubits: Optional[int] = None
    batch_size: Optional[int] = None
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.complex128))
    order: str = "C"  # 'C' for row-major, 'F' for column-major
    device: str = "cpu"

    @property
    def is_batched(self) -> bool:
        return self.batch_size is not None and self.batch_size > 0

    @property
    def state_dim(self) -> int:
        """Total state space dimension (2^n_qubits if qubit-based)."""
        if self.n_qubits is not None:
            return 1 << self.n_qubits
        return 0

    @property
    def tensor_shape(self) -> Tuple[int, ...]:
        """Tensor representation shape: (2, 2, ..., 2) or (batch, 2, 2, ..., 2)."""
        if self.n_qubits is None:
            raise ValueError("n_qubits must be defined for tensor shape.")
        qubit_shape = (2,) * self.n_qubits
        if self.is_batched:
            return (self.batch_size,) + qubit_shape
        return qubit_shape

    @property
    def flat_shape(self) -> Tuple[int, ...]:
        """Flat vector shape: (2^n,) or (batch, 2^n)."""
        dim = self.state_dim
        if self.is_batched:
            return (self.batch_size, dim)
        return (dim,)

    @property
    def density_tensor_shape(self) -> Tuple[int, ...]:
        """Density matrix tensor shape: (2, 2, ..., 2, 2, 2, ..., 2) where

        first N indices are row qubits, second N indices are column qubits.
        """
        if self.n_qubits is None:
            raise ValueError("n_qubits must be defined for density tensor shape.")
        qubit_shape = (2,) * (2 * self.n_qubits)
        if self.is_batched:
            return (self.batch_size,) + qubit_shape
        return qubit_shape

    @property
    def density_matrix_shape(self) -> Tuple[int, ...]:
        """Density matrix 2D shape: (2^n, 2^n) or (batch, 2^n, 2^n)."""
        dim = self.state_dim
        if self.is_batched:
            return (self.batch_size, dim, dim)
        return (dim, dim)

    @staticmethod
    def align_array(
        arr: np.ndarray,
        alignment: int = 64,
        order: str = "C",
        dtype: Optional[np.dtype] = None
    ) -> np.ndarray:
        """Ensures an array is contiguous and aligned to SIMD boundary (e.g. 64-byte for AVX-512).

        Args:
            arr: Input NumPy array.
            alignment: Byte alignment (default 64 bytes).
            order: 'C' or 'F'.
            dtype: Target dtype (preserves existing if None).

        Returns:
            Contiguous, SIMD-aligned NumPy array.
        """
        target_dtype = dtype or arr.dtype
        # Convert or ensure contiguous
        if not arr.flags.c_contiguous and order == "C":
            arr = np.ascontiguousarray(arr, dtype=target_dtype)
        elif not arr.flags.f_contiguous and order == "F":
            arr = np.asfortranarray(arr, dtype=target_dtype)
        elif arr.dtype != target_dtype:
            arr = arr.astype(target_dtype, copy=False)

        # Check alignment of data pointer
        addr = arr.ctypes.data
        if addr % alignment == 0:
            return arr

        # Allocate aligned memory buffer if misaligned
        n_bytes = arr.nbytes + alignment
        raw_buffer = np.empty(n_bytes, dtype=np.uint8)
        offset = (-raw_buffer.ctypes.data) % alignment
        aligned_view = raw_buffer[offset : offset + arr.nbytes].view(target_dtype).reshape(arr.shape, order=order)
        np.copyto(aligned_view, arr)
        return aligned_view

    @staticmethod
    def to_tensor_view(data: np.ndarray, n_qubits: int, is_batched: bool = False) -> np.ndarray:
        """Returns a zero-copy tensor view shaped (2, 2, ..., 2) or (batch, 2, ...)."""
        target_shape = ((data.shape[0],) + (2,) * n_qubits) if is_batched else ((2,) * n_qubits)
        if data.shape == target_shape:
            return data
        return data.reshape(target_shape)

    @staticmethod
    def to_flat_view(data: np.ndarray, is_batched: bool = False) -> np.ndarray:
        """Returns a flat vector view of shape (2^n,) or (batch, 2^n)."""
        if is_batched:
            batch_size = data.shape[0]
            return data.reshape(batch_size, -1)
        return data.reshape(-1)
