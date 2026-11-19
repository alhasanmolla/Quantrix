"""Backend subsystem initialization for Quantrix."""

from quantrix.backends.base import AbstractBackend, BackendManager
from quantrix.backends.numpy_backend import NumPyBackend
from quantrix.backends.numba_backend import NumbaBackend, NUMBA_AVAILABLE
from quantrix.backends.torch_backend import TorchBackend, TORCH_AVAILABLE
from quantrix.backends.c_hooks import CHookBackend, CExtensionBridge

# Register default backends
_numpy_backend = NumPyBackend()
BackendManager.register_backend(_numpy_backend)

if NUMBA_AVAILABLE:
    _numba_backend = NumbaBackend()
    BackendManager.register_backend(_numba_backend)

if TORCH_AVAILABLE:
    _torch_backend = TorchBackend()
    BackendManager.register_backend(_torch_backend)

_c_backend = CHookBackend()
BackendManager.register_backend(_c_backend)

__all__ = [
    "AbstractBackend",
    "BackendManager",
    "NumPyBackend",
    "NumbaBackend",
    "TorchBackend",
    "CHookBackend",
    "CExtensionBridge",
]
