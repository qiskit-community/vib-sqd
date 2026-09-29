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

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vib_sqd.operators.pauli_utils import sparse_pauli_from_index_terms


@dataclass
class SimpleSQDResult:
    post_sqd_probabilities: dict[str, float]
    selected_basis_bitstrings: list[str]
    spectrum_df: pd.DataFrame


def encoded_to_dense_matrix(encoded_hamiltonian) -> np.ndarray:
    """
    Convert an encoded Hamiltonian object into a dense matrix.

    Supports:
      - objects with hamiltonian_matrix
      - objects with pauli_terms
    """
    if hasattr(encoded_hamiltonian, "hamiltonian_matrix"):
        return np.asarray(encoded_hamiltonian.hamiltonian_matrix, dtype=complex)

    if hasattr(encoded_hamiltonian, "pauli_terms"):
        return (
            sparse_pauli_from_index_terms(encoded_hamiltonian.pauli_terms)
            .simplify()
            .to_matrix()
        )

    raise AttributeError(
        "Encoded Hamiltonian has neither hamiltonian_matrix nor pauli_terms."
    )


def _bitstring_to_index(bitstring: str) -> int:
    return int(bitstring, 2)


def run_simple_sqd_projection(
    encoded_hamiltonian,
    bitstring_probabilities: dict[str, float],
    threshold: float = 1e-4,
    init_bitstring: str | None = None,
    min_basis_size: int = 8,
    n_roots: int = 5,
) -> SimpleSQDResult:
    """
    Simple SQD-like postprocessing:
      1. choose basis states whose sampled probability >= threshold
      2. ensure init_bitstring is included
      3. if too few basis states, add most probable states until min_basis_size
      4. project Hamiltonian into that sampled subspace
      5. diagonalize projected Hamiltonian
      6. save ground-state amplitude probabilities over selected basis states

    This is a lightweight postprocessing layer suitable for your current
    vibrational workflows.
    """
    dense_H = encoded_to_dense_matrix(encoded_hamiltonian)

    physical = None
    if hasattr(encoded_hamiltonian, "physical_state_indices"):
        physical = set(int(x) for x in encoded_hamiltonian.physical_state_indices)

    # ---------------------------------------------------------
    # 1. thresholded basis
    # ---------------------------------------------------------
    selected = [
        str(b)
        for b, p in bitstring_probabilities.items()
        if float(p) >= float(threshold)
    ]

    # ---------------------------------------------------------
    # 2. always include initial state
    # ---------------------------------------------------------
    if init_bitstring is not None and init_bitstring not in selected:
        selected.append(init_bitstring)

    # ---------------------------------------------------------
    # 3. if too few states, top up with most probable states
    # ---------------------------------------------------------
    if len(selected) < min_basis_size:
        ranked = sorted(
            bitstring_probabilities.items(),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )
        for b, _ in ranked:
            if b not in selected:
                selected.append(str(b))
            if len(selected) >= min_basis_size:
                break

    # keep only physical states if such info exists
    filtered_selected = []
    for b in selected:
        idx = _bitstring_to_index(b)
        if physical is None or idx in physical:
            filtered_selected.append(b)
    selected = filtered_selected

    if len(selected) == 0:
        raise ValueError("No basis states selected for simple SQD projection.")

    indices = np.array([_bitstring_to_index(b) for b in selected], dtype=int)

    # ---------------------------------------------------------
    # 4. projected Hamiltonian
    # ---------------------------------------------------------
    H_sub = dense_H[np.ix_(indices, indices)]

    # ---------------------------------------------------------
    # 5. diagonalize
    # ---------------------------------------------------------
    evals, evecs = np.linalg.eigh(H_sub)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    # ---------------------------------------------------------
    # 6. ground-state probabilities on the selected basis
    # ---------------------------------------------------------
    gs = evecs[:, 0]
    probs = np.abs(gs) ** 2

    post_probs = {bitstring: float(p) for bitstring, p in zip(selected, probs)}

    # Save spectrum table (roots of projected Hamiltonian)
    rows = []
    for root in range(min(n_roots, len(evals))):
        rows.append(
            {
                "state": f"root_{root}",
                "stage": "post_sqd",
                "energy_cm_inv": float(evals[root]),
            }
        )
    spectrum_df = pd.DataFrame(rows)

    return SimpleSQDResult(
        post_sqd_probabilities=post_probs,
        selected_basis_bitstrings=selected,
        spectrum_df=spectrum_df,
    )


def save_simple_sqd_result(
    outdir: str | Path,
    result: SimpleSQDResult,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "post_sqd_probabilities.json").write_text(
        json.dumps(result.post_sqd_probabilities, indent=2)
    )

    (outdir / "selected_basis_bitstrings.json").write_text(
        json.dumps(result.selected_basis_bitstrings, indent=2)
    )

    result.spectrum_df.to_csv(outdir / "estimated_spectrum.csv", index=False)
