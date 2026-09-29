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

"""
Penalty Hamiltonians for constraining physical subspaces.

Implements penalty terms to discourage unphysical states in variational algorithms,
following the approach in the RSC paper Eq. 19-20.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def add_one_hot_penalty(
    hamiltonian: SparsePauliOp,
    n_modes: int,
    n_modals: int,
    mu: float,
) -> SparsePauliOp:
    """
    Add one-hot occupation penalty to Hamiltonian.

    For direct encoding, the physical subspace requires exactly one occupied
    modal per mode. This adds a penalty term:

        H' = H + μ * Σ_mode (N_mode - 1)²

    where N_mode = Σ_modal n_modal is the modal number operator for each mode.

    Since qubit number operators satisfy n² = n:
        (Σ n_i - 1)² = Σ n_i + 2 Σ_{i<j} n_i n_j - 2 Σ n_i + 1
                     = 1 - Σ n_i + 2 Σ_{i<j} n_i n_j

    Using n_q = (I - Z_q)/2 for qubits:
        n_i n_j = (I - Z_i)(I - Z_j)/4 = (I - Z_i - Z_j + Z_i Z_j)/4

    Args:
        hamiltonian: Original Hamiltonian operator
        n_modes: Number of vibrational modes
        n_modals: Number of modal levels per mode
        mu: Penalty strength (should be >> vibrational energy scale)

    Returns:
        Modified Hamiltonian with penalty terms

    Example:
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> H = SparsePauliOp(["IIII"], [1.0])
        >>> H_penalized = add_one_hot_penalty(H, n_modes=2, n_modals=2, mu=100.0)

    Reference:
        McArdle et al., "Quantum simulation of molecular vibrations with quantum computers"
        Phys. Chem. Chem. Phys., 2020, 22, 25324-25340
        Equations 19-20
    """
    n_qubits = n_modes * n_modals

    # Build penalty terms
    penalty_labels = []
    penalty_coeffs = []

    for mode in range(n_modes):
        mode_qubits = [mode * n_modals + a for a in range(n_modals)]

        # Constant term: +μ for each mode
        penalty_labels.append("I" * n_qubits)
        penalty_coeffs.append(mu)

        # Linear terms: -μ Σ n_i = -μ/2 Σ (I - Z_i)
        for q in mode_qubits:
            # -μ/2 * I term (absorbed into constant)
            penalty_labels.append("I" * n_qubits)
            penalty_coeffs.append(-mu / 2)

            # +μ/2 * Z_i term
            z_label = ["I"] * n_qubits
            z_label[n_qubits - 1 - q] = "Z"  # Qiskit big-endian label
            penalty_labels.append("".join(z_label))
            penalty_coeffs.append(mu / 2)

        # Quadratic terms: +2μ Σ_{i<j} n_i n_j
        # = +2μ Σ_{i<j} (I - Z_i - Z_j + Z_i Z_j)/4
        # = +μ/2 Σ_{i<j} (I - Z_i - Z_j + Z_i Z_j)
        for i, q_i in enumerate(mode_qubits):
            for q_j in mode_qubits[i + 1 :]:
                # +μ/2 * I term
                penalty_labels.append("I" * n_qubits)
                penalty_coeffs.append(mu / 2)

                # -μ/2 * Z_i term
                z_i_label = ["I"] * n_qubits
                z_i_label[n_qubits - 1 - q_i] = "Z"
                penalty_labels.append("".join(z_i_label))
                penalty_coeffs.append(-mu / 2)

                # -μ/2 * Z_j term
                z_j_label = ["I"] * n_qubits
                z_j_label[n_qubits - 1 - q_j] = "Z"
                penalty_labels.append("".join(z_j_label))
                penalty_coeffs.append(-mu / 2)

                # +μ/2 * Z_i Z_j term
                zz_label = ["I"] * n_qubits
                zz_label[n_qubits - 1 - q_i] = "Z"
                zz_label[n_qubits - 1 - q_j] = "Z"
                penalty_labels.append("".join(zz_label))
                penalty_coeffs.append(mu / 2)

    # Create penalty operator
    penalty_op = SparsePauliOp(penalty_labels, np.array(penalty_coeffs))

    # Simplify by combining like terms
    penalty_op = penalty_op.simplify()

    # Add to original Hamiltonian
    return hamiltonian + penalty_op


def estimate_penalty_strength(
    hamiltonian: SparsePauliOp,
    safety_factor: float = 10.0,
) -> float:
    """
    Estimate appropriate penalty strength μ.

    The penalty should be large enough to make unphysical states energetically
    unfavorable, but not so large as to cause numerical issues.

    A reasonable heuristic is:
        μ ≈ safety_factor × max(|H_ij|)

    Args:
        hamiltonian: Original Hamiltonian
        safety_factor: Multiplier for maximum coefficient (default 10.0)

    Returns:
        Recommended penalty strength
    """
    coeffs = (
        np.array(hamiltonian.coeffs)
        if hamiltonian.coeffs is not None
        else np.array([0.0])
    )
    max_coeff = np.max(np.abs(coeffs))
    return safety_factor * max_coeff


def is_physical_state(state_idx: int, n_modes: int, n_modals: int) -> bool:
    """Return True when the computational basis state is one-hot in every mode block."""
    for mode in range(n_modes):
        mode_occupation = 0
        for modal in range(n_modals):
            q = mode * n_modals + modal
            mode_occupation += (state_idx >> q) & 1
        if mode_occupation != 1:
            return False
    return True


def physical_basis_indices(n_modes: int, n_modals: int) -> np.ndarray:
    """
    Enumerate physical direct-encoding basis indices.

    Uses the repo convention that each mode block is one-hot left-to-right
    externally, matching the direct encoding helpers.
    """
    indices = []
    for occupations in product(range(n_modals), repeat=n_modes):
        bits = []
        for occ in occupations:
            mode_bits = [0] * n_modals
            mode_bits[occ] = 1
            bits.extend(mode_bits)

        idx = 0
        for bit in bits:
            idx = (idx << 1) | bit
        indices.append(idx)

    return np.array(sorted(indices), dtype=int)


def project_to_physical_subspace(
    hamiltonian: SparsePauliOp | np.ndarray,
    n_modes: int,
    n_modals: int,
) -> np.ndarray:
    """Project a full Hamiltonian into the one-hot physical subspace."""
    H_matrix = (
        hamiltonian.to_matrix()
        if isinstance(hamiltonian, SparsePauliOp)
        else np.asarray(hamiltonian)
    )
    indices = physical_basis_indices(n_modes, n_modals)
    return H_matrix[np.ix_(indices, indices)]


def check_physical_subspace_energy(
    hamiltonian: SparsePauliOp,
    n_modes: int,
    n_modals: int,
) -> tuple[float, float]:
    """
    Check ground state energies in full vs physical subspaces.

    This diagnostic helps identify if unphysical states have lower energy
    than physical ones, indicating the need for a penalty term.
    """
    H_matrix = hamiltonian.to_matrix()
    E_full = np.min(np.linalg.eigvalsh(H_matrix))
    H_physical = project_to_physical_subspace(H_matrix, n_modes, n_modals)
    E_physical = np.min(np.linalg.eigvalsh(H_physical))
    return float(E_full), float(E_physical)
