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

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from vib_sqd.basis.modal import (
    compute_modal_matrix_element_1body,
    solve_modal_basis,
)
from vib_sqd.data.forcefields.co2 import ForceTerm, load_co2_bending_only
from vib_sqd.encoding.direct import build_direct_encoded_vibrational_hamiltonian
from vib_sqd.operators.penalty import (
    add_one_hot_penalty,
    check_physical_subspace_energy,
    is_physical_state,
    physical_basis_indices,
    project_to_physical_subspace,
)


def test_is_physical_state_and_indices_match_direct_encoding_convention():
    n_modes = 2
    n_modals = 2

    indices = physical_basis_indices(n_modes, n_modals)
    assert indices.tolist() == [5, 6, 9, 10]

    for idx in indices:
        assert is_physical_state(int(idx), n_modes, n_modals)

    for idx in [0, 3, 12, 15]:
        assert not is_physical_state(idx, n_modes, n_modals)


def test_project_to_physical_subspace_extracts_expected_block():
    diagonal = np.arange(16, dtype=float)
    H = np.diag(diagonal)

    projected = project_to_physical_subspace(H, n_modes=2, n_modals=2)
    expected = np.diag([5.0, 6.0, 9.0, 10.0])

    assert projected.shape == (4, 4)
    assert np.allclose(projected, expected)


def test_check_physical_subspace_energy_detects_unphysical_lower_state():
    coeffs = np.arange(16, dtype=float) - 20.0
    coeffs[[5, 6, 9, 10]] += 30.0
    hamiltonian = SparsePauliOp.from_operator(Operator(np.diag(coeffs)))

    e_full, e_physical = check_physical_subspace_energy(
        hamiltonian, n_modes=2, n_modals=2
    )

    assert e_full < e_physical
    assert np.isclose(e_full, -20.0)
    assert np.isclose(e_physical, 15.0)


def test_harmonic_modal_matrix_elements_match_analytic_low_lying_values():
    omega = 1000.0
    basis = solve_modal_basis(
        omega=omega,
        anharmonic_terms=[],
        n_modals=3,
        n_dvr=201,
        q_range=8.0,
    )

    q01 = compute_modal_matrix_element_1body(basis, lambda q: q, 0, 1)
    q20 = compute_modal_matrix_element_1body(basis, lambda q: q**2, 0, 0)

    assert np.isclose(q01, 1 / np.sqrt(2), atol=5e-2)
    assert np.isclose(q20, 0.5, atol=5e-2)


def test_explicit_forcefield_powers_change_the_encoded_hamiltonian():
    class SimpleForceField:
        def __init__(self):
            self.omega = (100.0, 100.0)
            self.terms = (ForceTerm(1.0, (1, 2), (2, 1)),)

    class LegacyLikeForceField:
        def __init__(self):
            self.omega = (100.0, 100.0)
            self.terms = (ForceTerm(1.0, (1, 2), (1, 1)),)

    with_powers = build_direct_encoded_vibrational_hamiltonian(
        SimpleForceField(), vmax=1, maxorder=4, tol=1e-10
    )
    without_powers = build_direct_encoded_vibrational_hamiltonian(
        LegacyLikeForceField(), vmax=1, maxorder=4, tol=1e-10
    )

    assert not np.allclose(
        with_powers.hamiltonian_matrix, without_powers.hamiltonian_matrix
    )


def test_co2_bending_exact_physical_energy_matches_corrected_reference():
    ff = load_co2_bending_only()
    enc = build_direct_encoded_vibrational_hamiltonian(
        ff, vmax=1, maxorder=4, tol=1e-10
    )
    hamiltonian = SparsePauliOp.from_operator(Operator(enc.hamiltonian_matrix))

    e_full, e_physical = check_physical_subspace_energy(
        hamiltonian, n_modes=2, n_modals=2
    )

    assert np.isclose(e_physical, 669.0091609969484, atol=1e-6)
    assert e_full < e_physical


def test_one_hot_penalty_restores_the_physical_ground_sector():
    ff = load_co2_bending_only()
    enc = build_direct_encoded_vibrational_hamiltonian(
        ff, vmax=1, maxorder=4, tol=1e-10
    )
    hamiltonian = SparsePauliOp.from_operator(Operator(enc.hamiltonian_matrix))

    _, e_physical = check_physical_subspace_energy(hamiltonian, n_modes=2, n_modals=2)

    penalized = add_one_hot_penalty(
        hamiltonian,
        n_modes=2,
        n_modals=2,
        mu=2000.0,
    )
    e_full_with_penalty = np.min(np.linalg.eigvalsh(penalized.to_matrix()))

    assert np.isclose(e_full_with_penalty, e_physical, atol=1e-6)
