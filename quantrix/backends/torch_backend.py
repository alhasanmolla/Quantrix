"""GPU and C++ ATen Accelerated Backend using PyTorch.

Enables massive parallel hardware acceleration (CUDA/ROCm/MPS/CPU),
automatic differentiation, and zero-overhead tensor operations for Quantrix.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np

from quantrix.backends.base import AbstractBackend
from quantrix.ops.einsum_ops import (
    generate_gate_einsum,
    generate_expectation_einsum,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TorchBackend(AbstractBackend):
    """PyTorch hardware-accelerated backend with CPU and GPU support."""

    def __init__(self, device: Optional[str] = None) -> None:
        if TORCH_AVAILABLE:
            if device is not None:
                self._device = torch.device(device)
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = None

    @property
    def name(self) -> str:
        return "torch"

    @property
    def is_available(self) -> bool:
        return TORCH_AVAILABLE

    @property
    def device(self) -> Any:
        return self._device

    def einsum(
        self,
        subscripts: str,
        *operands: Any,
        optimize: Union[bool, str] = "optimal"
    ) -> Any:
        # Convert operands to torch tensors if needed
        t_operands = [self.to_backend_array(op) for op in operands]
        return torch.einsum(subscripts, *t_operands)

    def matmul(self, a: Any, b: Any) -> Any:
        ta = self.to_backend_array(a)
        tb = self.to_backend_array(b)
        return torch.matmul(ta, tb)

    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> Any:
        ta = self.to_backend_array(a)
        tb = self.to_backend_array(b)
        return torch.tensordot(ta, tb, dims=axes)

    def norm(
        self,
        a: Any,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> Any:
        ta = self.to_backend_array(a)
        if axis is None:
            return torch.linalg.vector_norm(ta)
        return torch.linalg.vector_norm(ta, dim=axis, keepdim=keepdims)

    def apply_gate(
        self,
        state_tensor: Any,
        gate_matrix: Any,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> Any:
        t_state = self.to_backend_array(state_tensor)
        t_gate = self.to_backend_array(gate_matrix)

        k = len(target_qubits)
        gate_single_dim = 1 << (2 * k)
        gate_is_batched = t_gate.numel() > gate_single_dim and (t_gate.numel() % gate_single_dim == 0)

        if gate_is_batched:
            b_size = t_gate.numel() // gate_single_dim
            gate_tensor_shape = (b_size,) + (2,) * (2 * k)
        else:
            gate_tensor_shape = (2,) * (2 * k)

        t_gate = t_gate.reshape(gate_tensor_shape)

        einsum_str, _ = generate_gate_einsum(
            target_qubits, n_qubits, is_batched=is_batched, gate_is_batched=gate_is_batched
        )
        return torch.einsum(einsum_str, t_gate, t_state)

    def inner_product(
        self,
        state_a: Any,
        state_b: Any,
        is_batched: bool = False
    ) -> Any:
        ta = self.to_backend_array(state_a)
        tb = self.to_backend_array(state_b)
        if not is_batched:
            return torch.vdot(ta.reshape(-1), tb.reshape(-1))
        
        flat_a = ta.reshape(ta.shape[0], -1)
        flat_b = tb.reshape(tb.shape[0], -1)
        return torch.sum(torch.conj(flat_a) * flat_b, dim=-1)

    def expectation_value(
        self,
        state_tensor: Any,
        operator: Any,
        target_qubits: Sequence[int],
        n_qubits: int,
        is_batched: bool = False
    ) -> Any:
        t_state = self.to_backend_array(state_tensor)
        t_op = self.to_backend_array(operator)

        k = len(target_qubits)
        op_tensor = t_op.reshape((2,) * (2 * k))
        einsum_str, _ = generate_expectation_einsum(target_qubits, n_qubits, is_batched=is_batched)
        
        return torch.einsum(einsum_str, op_tensor, torch.conj(t_state), t_state)

    def to_backend_array(
        self,
        array: Any,
        dtype: Optional[Any] = None,
        device: Optional[str] = None
    ) -> Any:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available.")
        
        target_device = torch.device(device) if device is not None else self._device

        if isinstance(array, torch.Tensor):
            t = array
        elif isinstance(array, np.ndarray):
            t = torch.from_numpy(array)
        else:
            t = torch.tensor(array)

        if dtype is not None:
            if not isinstance(dtype, torch.dtype):
                np_dtype = np.dtype(dtype)
                dtype_map = {
                    np.dtype("complex128"): torch.complex128,
                    np.dtype("complex64"): torch.complex64,
                    np.dtype("float64"): torch.float64,
                    np.dtype("float32"): torch.float32,
                    np.dtype("int64"): torch.int64,
                    np.dtype("int32"): torch.int32,
                }
                torch_dtype = dtype_map.get(np_dtype, torch.complex128)
            else:
                torch_dtype = dtype
            t = t.to(dtype=torch_dtype)
        if target_device is not None and t.device != target_device:
            t = t.to(device=target_device)
        return t

    def to_numpy(self, array: Any) -> np.ndarray:
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return np.asarray(array)
