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
from itertools import product
from typing import Literal

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from vib_sqd.basis.harmonic import number_operator, q_power_matrix


@dataclass
class DirectEncodedHamiltonian:
    hamiltonian_matrix: np.ndarray
    n_qubits: int
    valid_bitstrings: np.ndarray
    physical_state_indices: np.ndarray
    pauli_terms: list[tuple[complex, tuple[int, ...]]]


def embed_single_mode_operator_direct(op: np.ndarray) -> np.ndarray:
    """
    Embed a modal-space operator into one-hot direct qubit encoding.

    For n_modals=2:
        modal 0 -> |10> -> index 2
        modal 1 -> |01> -> index 1
    """
    op = np.asarray(op, dtype=complex)

    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise ValueError("op must be a square matrix.")

    n_modals = op.shape[0]
    dim = 2**n_modals
    out = np.zeros((dim, dim), dtype=complex)

    one_hot_indices = [1 << (n_modals - 1 - k) for k in range(n_modals)]

    for i in range(n_modals):
        for j in range(n_modals):
            out[one_hot_indices[i], one_hot_indices[j]] = op[i, j]

    return out


def _kron_all(ops: list[np.ndarray]) -> np.ndarray:
    if not ops:
        raise ValueError("ops must not be empty.")
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _physical_indices(n_modes: int, n_modals: int) -> np.ndarray:
    valid = []

    for occ in product(range(n_modals), repeat=n_modes):
        bits = []
        for o in occ:
            block = [0] * n_modals
            block[o] = 1
            bits.extend(block)

        idx = 0
        for bit in bits:
            idx = (idx << 1) | bit
        valid.append(idx)

    return np.array(sorted(valid), dtype=int)


def _term_powers_by_mode(term, n_modes: int) -> list[int]:
    """
    Convert a ForceTerm into per-mode total powers.

    Supports both normal compressed form:
        modes=(1, 2), powers=(2, 1)

    and defensive repeated-mode form:
        modes=(1, 1), powers=(1, 1)
    """
    term_powers = getattr(term, "powers", tuple(1 for _ in term.modes))

    if len(term_powers) != len(term.modes):
        raise ValueError(f"ForceTerm powers length does not match modes length: {term}")

    powers_by_mode = [0] * n_modes

    for mode, power in zip(term.modes, term_powers):
        if mode < 1 or mode > n_modes:
            raise ValueError(f"ForceTerm {term} references mode outside 1..{n_modes}.")
        if power < 1:
            raise ValueError(f"ForceTerm powers must be positive: {term}")

        powers_by_mode[mode - 1] += int(power)

    return powers_by_mode


def _is_pure_single_mode_quadratic(term, n_modes: int) -> bool:
    """
    Identify terms like q_i^2.

    These are the only degree-2 terms skipped when using
    harmonic_source='omega_number_operator'. Bilinear couplings q_i q_j
    must be retained.
    """
    powers_by_mode = _term_powers_by_mode(term, n_modes)
    degree = sum(powers_by_mode)

    if degree != 2:
        return False

    active_modes = [i for i, p in enumerate(powers_by_mode) if p != 0]
    return len(active_modes) == 1 and powers_by_mode[active_modes[0]] == 2


def _max_required_q_power(terms, n_modes: int, maxorder: int) -> int:
    """
    Compute the largest q^p needed by any individual mode after powers are
    accumulated per mode.
    """
    max_power = 1

    for term in terms:
        if len(term.modes) > maxorder:
            continue

        powers_by_mode = _term_powers_by_mode(term, n_modes)
        max_power = max(max_power, max(powers_by_mode, default=0))

    return max_power


def _validate_forcefield_coordinate_kind(forcefield) -> None:
    """
    direct.py uses q_power_matrix, whose q is dimensionless:
        q = (a + a†) / sqrt(2)

    Therefore polynomial coefficients must already be in this convention.
    """
    coordinate_kind = getattr(forcefield, "coordinate_kind", "dimensionless")

    if coordinate_kind != "dimensionless":
        raise ValueError(
            "Direct harmonic-basis encoding requires forcefield.coordinate_kind="
            "'dimensionless'. Got "
            f"{coordinate_kind!r}. Convert forcefield coefficients before encoding."
        )


def _validate_polynomial_quadratic_terms(forcefield) -> None:
    """
    If harmonic_source='polynomial_quadratic', the harmonic oscillator part is
    not built from omega*n. Therefore explicit pure quadratic terms q_i^2 must
    be present for every mode.
    """
    n_modes = len(forcefield.omega)

    has_pure_quadratic = [False] * n_modes

    for term in forcefield.terms:
        powers_by_mode = _term_powers_by_mode(term, n_modes)
        degree = sum(powers_by_mode)

        if degree != 2:
            continue

        active_modes = [i for i, p in enumerate(powers_by_mode) if p != 0]

        if len(active_modes) == 1:
            mode_idx = active_modes[0]
            if powers_by_mode[mode_idx] == 2:
                has_pure_quadratic[mode_idx] = True

    missing = [i + 1 for i, has_it in enumerate(has_pure_quadratic) if not has_it]

    if missing:
        raise ValueError(
            "harmonic_source='polynomial_quadratic' requires explicit pure "
            "quadratic q_i^2 terms for every mode. Missing modes: "
            f"{missing}. Use harmonic_source='omega_number_operator' instead, "
            "or include degree-2 terms in forcefield.terms."
        )


def build_direct_encoded_reference_matrix(
    omega,
    terms,
    vmax: int,
    maxorder: int,
    lambda_cutoff: float | None = None,
    include_zpe: bool = True,
    harmonic_source: Literal[
        "omega_number_operator",
        "polynomial_quadratic",
    ] = "omega_number_operator",
) -> DirectEncodedHamiltonian:
    """
    Dense direct-encoded harmonic-basis Hamiltonian.

    Coordinate convention:
        q_power_matrix uses q = (a + a†)/sqrt(2).

    harmonic_source='omega_number_operator':
        H_harm = sum_i omega_i * (n_i + 1/2)
        degree-2 force terms are skipped to avoid double counting.

    harmonic_source='polynomial_quadratic':
        no omega*n harmonic term is added.
        quadratic force terms are expected in forcefield.terms.
    """
    if vmax < 0:
        raise ValueError("vmax must be non-negative.")

    n_modals = vmax + 1
    n_modes = len(omega)
    n_qubits = n_modes * n_modals

    if n_modes <= 0:
        raise ValueError("At least one vibrational mode is required.")

    dim_mode = 2**n_modals
    identity_mode = np.eye(dim_mode, dtype=complex)
    full_identity = np.eye(dim_mode**n_modes, dtype=complex)

    number_direct = embed_single_mode_operator_direct(number_operator(vmax))

    max_power = _max_required_q_power(terms, n_modes=n_modes, maxorder=maxorder)

    q_cache = {
        p: embed_single_mode_operator_direct(q_power_matrix(vmax, p))
        for p in range(1, max_power + 1)
    }

    H = np.zeros((dim_mode**n_modes, dim_mode**n_modes), dtype=complex)

    if harmonic_source == "omega_number_operator":
        for mode_idx, w in enumerate(omega):
            ops = [identity_mode] * n_modes
            ops[mode_idx] = number_direct
            H += float(w) * _kron_all(ops)

            if include_zpe:
                H += 0.5 * float(w) * full_identity

    elif harmonic_source == "polynomial_quadratic":
        pass

    else:
        raise ValueError(f"Unknown harmonic_source={harmonic_source!r}")

    for term in terms:
        if len(term.modes) > maxorder:
            continue

        if lambda_cutoff is not None and abs(complex(term.coeff)) < float(
            lambda_cutoff
        ):
            continue

        term_powers = getattr(term, "powers", tuple(1 for _ in term.modes))

        if (
            harmonic_source == "omega_number_operator"
            and _is_pure_single_mode_quadratic(term, n_modes)
        ):
            continue

        powers_by_mode = [0] * n_modes

        for mode, power in zip(term.modes, term_powers):
            if mode < 1 or mode > n_modes:
                raise ValueError(
                    f"ForceTerm {term} references mode outside 1..{n_modes}."
                )
            powers_by_mode[mode - 1] += power

        ops = [identity_mode if p == 0 else q_cache[p] for p in powers_by_mode]

        H += complex(term.coeff) * _kron_all(ops)

    valid = _physical_indices(n_modes, n_modals)

    return DirectEncodedHamiltonian(
        hamiltonian_matrix=H,
        n_qubits=n_qubits,
        valid_bitstrings=valid,
        physical_state_indices=valid,
        pauli_terms=[],
    )


def _pauli_label_to_index_tuple(label: str) -> tuple[int, ...]:
    mapping = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    return tuple(mapping[ch] for ch in label)


def dense_matrix_to_pauli_terms(
    matrix: np.ndarray,
    tol: float = 1e-10,
) -> list[tuple[complex, tuple[int, ...]]]:
    op = SparsePauliOp.from_operator(Operator(matrix))
    terms = []

    for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
        c = complex(coeff)
        if abs(c) >= tol:
            terms.append((c, _pauli_label_to_index_tuple(label)))

    return terms


def build_direct_encoded_vibrational_hamiltonian(
    forcefield,
    vmax: int,
    maxorder: int,
    tol: float = 1e-10,
    lambda_cutoff: float | None = None,
    include_zpe: bool = True,
    harmonic_source: Literal[
        "omega_number_operator",
        "polynomial_quadratic",
    ] = "omega_number_operator",
) -> DirectEncodedHamiltonian:
    # Validate coordinate convention
    _validate_forcefield_coordinate_kind(forcefield)

    # Validate polynomial quadratic terms if needed
    if harmonic_source == "polynomial_quadratic":
        _validate_polynomial_quadratic_terms(forcefield)

    enc = build_direct_encoded_reference_matrix(
        omega=forcefield.omega,
        terms=forcefield.terms,
        vmax=vmax,
        maxorder=maxorder,
        lambda_cutoff=lambda_cutoff,
        include_zpe=include_zpe,
        harmonic_source=harmonic_source,
    )

    pauli_terms = dense_matrix_to_pauli_terms(enc.hamiltonian_matrix, tol=tol)

    if lambda_cutoff is not None:
        pauli_terms = [
            (c, idxs)
            for c, idxs in pauli_terms
            if abs(complex(c)) >= float(lambda_cutoff)
        ]

    return DirectEncodedHamiltonian(
        hamiltonian_matrix=enc.hamiltonian_matrix,
        n_qubits=enc.n_qubits,
        valid_bitstrings=enc.valid_bitstrings,
        physical_state_indices=enc.physical_state_indices,
        pauli_terms=pauli_terms,
    )
