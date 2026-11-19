"""C-Extension Hooks and Native Shared Library Interface for Quantrix.

Provides a robust ctypes-based bridge and hook architecture allowing high-performance
C, C++, or CUDA dynamic libraries (.so / .dll / .dylib) to be registered and invoked
directly with zero-copy NumPy array pointers.
"""

from __future__ import annotations
import ctypes
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from quantrix.backends.base import AbstractBackend
from quantrix.backends.numpy_backend import NumPyBackend


# Ctypes double complex struct
class ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


# Signature types for C extension hooks
C_MATMUL_FN = ctypes.CFUNCTYPE(
    ctypes.c_int,  # status code
    ctypes.c_void_p,  # ptr A
    ctypes.c_void_p,  # ptr B
    ctypes.c_void_p,  # ptr C (output)
    ctypes.c_int64,   # M
    ctypes.c_int64,   # N
    ctypes.c_int64,   # K
    ctypes.c_int      # is_complex
)

C_GATE_FN = ctypes.CFUNCTYPE(
    ctypes.c_int,  # status code
    ctypes.c_void_p,  # ptr state
    ctypes.c_void_p,  # ptr gate
    ctypes.c_int,     # n_qubits
    ctypes.c_int,     # target_qubit
    ctypes.c_void_p   # ptr out_state
)


class CExtensionBridge:
    """Manages loaded dynamic shared libraries and ctypes function bindings."""

    def __init__(self, library_path: Optional[str] = None) -> None:
        self.lib_path = library_path
        self._lib: Optional[ctypes.CDLL] = None
        self._hooks: Dict[str, Callable] = {}
        if library_path and os.path.exists(library_path):
            self.load_library(library_path)

    @property
    def is_loaded(self) -> bool:
        return self._lib is not None or len(self._hooks) > 0

    def load_library(self, path: str) -> bool:
        """Loads a shared library (.dll, .so, or .dylib) into memory."""
        try:
            self._lib = ctypes.CDLL(os.path.abspath(path))
            self.lib_path = path
            
            # Bind known C functions if present
            if hasattr(self._lib, "quantrix_c_matmul"):
                fn = self._lib.quantrix_c_matmul
                fn.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int
                ]
                fn.restype = ctypes.c_int
                self._hooks["matmul"] = fn

            if hasattr(self._lib, "quantrix_c_apply_gate"):
                fn = self._lib.quantrix_c_apply_gate
                fn.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                    ctypes.c_int, ctypes.c_void_p
                ]
                fn.restype = ctypes.c_int
                self._hooks["apply_gate"] = fn

            return True
        except Exception as e:
            self._lib = None
            return False

    def register_hook(self, name: str, func: Callable) -> None:
        """Directly registers a custom Python/ctypes callable hook."""
        self._hooks[name] = func

    def call_c_matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        out: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Calls C matmul via raw pointer passing."""
        if "matmul" not in self._hooks:
            return None

        m, k = a.shape
        _, n = b.shape
        if out is None:
            out = np.empty((m, n), dtype=a.dtype)

        is_complex = 1 if np.iscomplexobj(a) else 0
        status = self._hooks["matmul"](
            a.ctypes.data,
            b.ctypes.data,
            out.ctypes.data,
            m, n, k,
            is_complex
        )
        if status == 0:
            return out
        return None


class CHookBackend(AbstractBackend):
    """Backend utilizing registered C-extensions with fallback to NumPy."""

    def __init__(self, library_path: Optional[str] = None) -> None:
        self.bridge = CExtensionBridge(library_path)
        self._numpy_backend = NumPyBackend()

    @property
    def name(self) -> str:
        return "c_hook"

    @property
    def is_available(self) -> bool:
        return self.bridge.is_loaded

    def einsum(
        self,
        subscripts: str,
        *operands: Any,
        optimize: Union[bool, str] = "optimal"
    ) -> np.ndarray:
        return self._numpy_backend.einsum(subscripts, *operands, optimize=optimize)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        arr_a = np.ascontiguousarray(self.to_numpy(a))
        arr_b = np.ascontiguousarray(self.to_numpy(b))
        if arr_a.ndim == 2 and arr_b.ndim == 2 and self.bridge.is_loaded:
            res = self.bridge.call_c_matmul(arr_a, arr_b)
            if res is not None:
                return res
        return self._numpy_backend.matmul(arr_a, arr_b)

    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> np.ndarray:
        return self._numpy_backend.tensordot(a, b, axes=axes)

    def norm(
        self,
        a: Any,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> Union[float, np.ndarray]:
        return self._numpy_backend.norm(a, axis=axis, keepdims=keepdims)

    def apply_gate(
        self,
        state_tensor: np.ndarray,
        gate_matrix: np.ndarray,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> np.ndarray:
        # If custom C hook registered for gate, call it
        if "apply_gate" in self.bridge._hooks and not is_batched and len(target_qubits) == 1:
            state_c = np.ascontiguousarray(state_tensor.reshape(-1), dtype=np.complex128)
            gate_c = np.ascontiguousarray(gate_matrix, dtype=np.complex128)
            out_c = np.empty_like(state_c)
            status = self.bridge._hooks["apply_gate"](
                state_c.ctypes.data,
                gate_c.ctypes.data,
                n_qubits,
                target_qubits[0],
                out_c.ctypes.data
            )
            if status == 0:
                return out_c.reshape(state_tensor.shape)

        return self._numpy_backend.apply_gate(
            state_tensor, gate_matrix, target_qubits, n_qubits, is_batched
        )

    def inner_product(
        self,
        state_a: np.ndarray,
        state_b: np.ndarray,
        is_batched: bool = False
    ) -> Union[complex, np.ndarray]:
        return self._numpy_backend.inner_product(state_a, state_b, is_batched)

    def expectation_value(
        self,
        state_tensor: np.ndarray,
        operator: np.ndarray,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> Union[float, complex, np.ndarray]:
        return self._numpy_backend.expectation_value(
            state_tensor, operator, target_qubits, n_qubits, is_batched
        )

    def to_backend_array(
        self,
        array: Any,
        dtype: Optional[Any] = None,
        device: Optional[str] = None
    ) -> np.ndarray:
        return self._numpy_backend.to_backend_array(array, dtype=dtype, device=device)

    def to_numpy(self, array: Any) -> np.ndarray:
        return self._numpy_backend.to_numpy(array)
