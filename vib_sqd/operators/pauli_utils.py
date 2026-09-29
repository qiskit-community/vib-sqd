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

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp

PAULI_INDEX_TO_CHAR = {0: "I", 1: "X", 2: "Y", 3: "Z"}
CHAR_TO_PAULI_INDEX = {v: k for k, v in PAULI_INDEX_TO_CHAR.items()}


def pauli_indices_to_label(indices: Sequence[int]) -> str:
    return "".join(PAULI_INDEX_TO_CHAR[int(i)] for i in indices)


def label_to_pauli_indices(label: str) -> tuple[int, ...]:
    return tuple(CHAR_TO_PAULI_INDEX[c] for c in label)


def merge_pauli_terms(
    pauli_terms: Iterable[tuple[complex, Sequence[int]]], tol: float = 1e-12
):
    accum = defaultdict(complex)
    for coeff, idxs in pauli_terms:
        label = pauli_indices_to_label(idxs)
        accum[label] += coeff
    merged = [
        (c, label_to_pauli_indices(label)) for label, c in accum.items() if abs(c) > tol
    ]
    return merged


def sparse_pauli_from_index_terms(
    pauli_terms: Iterable[tuple[complex, Sequence[int]]],
) -> SparsePauliOp:
    labels = []
    coeffs = []
    for coeff, idxs in pauli_terms:
        labels.append(pauli_indices_to_label(idxs))
        coeffs.append(coeff)
    if not labels:
        return SparsePauliOp(["I"], coeffs=[0.0])
    return SparsePauliOp(labels, coeffs=np.array(coeffs, dtype=complex)).simplify()


def reconstruct_matrix_from_paulis(
    pauli_terms: Iterable[tuple[complex, Sequence[int]]],
) -> np.ndarray:
    op = sparse_pauli_from_index_terms(pauli_terms)
    return op.to_matrix()
