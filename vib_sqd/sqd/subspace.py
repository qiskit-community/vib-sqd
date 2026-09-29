# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass
class ProjectedSubspace:
    basis_bitstrings: np.ndarray  # shape (k, n_qubits), uint8
    h_proj: np.ndarray  # shape (k, k)


def bitrow_to_int(bitrow: np.ndarray) -> int:
    out = 0
    for b in bitrow:
        out = (out << 1) | int(b)
    return out


def project_sparse_pauli_to_sampled_basis(
    op: SparsePauliOp,
    basis_bitstrings: np.ndarray,
) -> np.ndarray:
    """
    Small/medium-scale projector path.
    For large-scale HPC, replace with chunked/distributed application.
    """
    k = len(basis_bitstrings)
    n = op.num_qubits
    dim = 2**n

    addresses = np.array([bitrow_to_int(b) for b in basis_bitstrings], dtype=int)
    mat = op.to_matrix(sparse=False)
    return mat[np.ix_(addresses, addresses)]


def diagonalize_projected(h_proj: np.ndarray, k: int = 5):
    evals, evecs = np.linalg.eigh(h_proj)
    idx = np.argsort(evals)[:k]
    return evals[idx], evecs[:, idx]
