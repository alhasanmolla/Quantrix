"""Einsum and tensor contraction utilities for Quantum-Inspired Parallel Processing.

Provides dynamic subscript generation and optimized Einstein summation routines
for applying n-qubit gates, multi-target operators, partial traces, and tensor contractions
with zero Python loops.
"""

from __future__ import annotations
import string
from typing import Sequence, Tuple, List, Union
import numpy as np


# Available index symbols for tensor modes (lowercase and uppercase letters)
INDEX_CHARS = string.ascii_lowercase + string.ascii_uppercase


def generate_gate_einsum(
    target_qubits: Sequence[int],
    n_qubits: int,
    is_batched: bool = False,
    gate_is_batched: bool = False
) -> Tuple[str, Tuple[int, ...]]:
    """Generates optimal einsum subscripts for applying an arbitrary k-qubit gate

    to specified target qubits in an n-qubit state tensor.

    Args:
        target_qubits: Tuple/List of qubit indices (0 to n_qubits - 1).
        n_qubits: Total number of qubits in the state.
        is_batched: Whether the state contains a leading batch dimension.
        gate_is_batched: Whether the gate matrix contains a leading batch dimension.

    Returns:
        einsum_str: Formatted subscript, e.g. "Bij,Bajc->Baic" or "ij,...ajc...->...aic..."
        gate_tensor_shape: Expected shape of the gate tensor.
    """
    k = len(target_qubits)
    if k > n_qubits:
        raise ValueError(f"Target qubits ({k}) cannot exceed total qubits ({n_qubits})")
    for q in target_qubits:
        if not (0 <= q < n_qubits):
            raise ValueError(f"Target qubit {q} out of range [0, {n_qubits - 1}]")

    # Allocate unique character labels
    state_in_indices = [INDEX_CHARS[i] for i in range(n_qubits)]
    gate_out_indices = [INDEX_CHARS[n_qubits + i] for i in range(k)]
    gate_in_indices = [state_in_indices[q] for q in target_qubits]

    state_out_indices = list(state_in_indices)
    for idx, q in enumerate(target_qubits):
        state_out_indices[q] = gate_out_indices[idx]

    batch_char = "Z"
    if gate_is_batched and is_batched:
        gate_subscript = batch_char + "".join(gate_out_indices) + "".join(gate_in_indices)
        state_in_str = batch_char + "".join(state_in_indices)
        state_out_str = batch_char + "".join(state_out_indices)
    elif gate_is_batched and not is_batched:
        gate_subscript = batch_char + "".join(gate_out_indices) + "".join(gate_in_indices)
        state_in_str = "".join(state_in_indices)
        state_out_str = batch_char + "".join(state_out_indices)
    elif not gate_is_batched and is_batched:
        gate_subscript = "".join(gate_out_indices) + "".join(gate_in_indices)
        state_in_str = "..." + "".join(state_in_indices)
        state_out_str = "..." + "".join(state_out_indices)
    else:
        gate_subscript = "".join(gate_out_indices) + "".join(gate_in_indices)
        state_in_str = "".join(state_in_indices)
        state_out_str = "".join(state_out_indices)

    einsum_str = f"{gate_subscript},{state_in_str}->{state_out_str}"
    gate_shape = (2,) * (2 * k)

    return einsum_str, gate_shape


def generate_expectation_einsum(
    target_qubits: Sequence[int],
    n_qubits: int,
    is_batched: bool = False
) -> Tuple[str, Tuple[int, ...]]:
    """Generates einsum subscripts to compute <psi| O |psi> in a single contraction.

    Args:
        target_qubits: Qubits the operator O acts on.
        n_qubits: Total number of qubits in the state.
        is_batched: Whether leading batch dimension is present.

    Returns:
        einsum_str: Formatted contraction string for: O, psi, conj(psi) -> scalar or (batch,)
        gate_shape: Shape of O tensor.
    """
    k = len(target_qubits)
    state_bra_indices = [INDEX_CHARS[i] for i in range(n_qubits)]
    state_ket_indices = list(state_bra_indices)
    
    # New indices for ket at target qubits
    ket_new_indices = [INDEX_CHARS[n_qubits + i] for i in range(k)]
    for idx, q in enumerate(target_qubits):
        state_ket_indices[q] = ket_new_indices[idx]

    # Operator indices: bra targets, ket targets
    op_bra_targets = [state_bra_indices[q] for q in target_qubits]
    op_subscript = "".join(op_bra_targets) + "".join(ket_new_indices)

    batch_prefix = "..." if is_batched else ""
    bra_str = batch_prefix + "".join(state_bra_indices)
    ket_str = batch_prefix + "".join(state_ket_indices)
    out_str = "..." if is_batched else ""

    einsum_str = f"{op_subscript},{bra_str},{ket_str}->{out_str}"
    gate_shape = (2,) * (2 * k)
    return einsum_str, gate_shape


def generate_partial_trace_einsum(
    keep_qubits: Sequence[int],
    n_qubits: int,
    is_batched: bool = False
) -> str:
    """Generates einsum subscripts for tracing out all qubits EXCEPT keep_qubits

    from a density tensor of shape (2, ..., 2, 2, ..., 2) [row qubits, col qubits].

    Args:
        keep_qubits: Sequence of qubit indices to preserve.
        n_qubits: Total qubits.
        is_batched: Leading batch dimension.

    Returns:
        einsum_str: e.g. "abcadc->bd" (tracing out qubits a and c)
    """
    row_indices = [INDEX_CHARS[i] for i in range(n_qubits)]
    col_indices = [INDEX_CHARS[n_qubits + i] for i in range(n_qubits)]
    
    # Traced qubits must share the same symbol in rows and cols
    for q in range(n_qubits):
        if q not in keep_qubits:
            col_indices[q] = row_indices[q]

    batch_prefix = "..." if is_batched else ""
    in_subscript = batch_prefix + "".join(row_indices) + "".join(col_indices)
    
    out_row = [row_indices[q] for q in keep_qubits]
    out_col = [col_indices[q] for q in keep_qubits]
    out_subscript = batch_prefix + "".join(out_row) + "".join(out_col)

    return f"{in_subscript}->{out_subscript}"


def generate_batched_inner_product_einsum(n_qubits: int) -> str:
    """Computes <psi_i | phi_j> across all pairs (i, j) in batches of states.

    psi shape: (B1, 2, ..., 2)
    phi shape: (B2, 2, ..., 2)
    Returns: Gram matrix of shape (B1, B2)
    """
    qubit_indices = "".join(INDEX_CHARS[i] for i in range(n_qubits))
    return f"x{qubit_indices},y{qubit_indices}->xy"
