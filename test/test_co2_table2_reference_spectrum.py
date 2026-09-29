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

from vib_sqd.data.forcefields.co2 import (
    ForceFieldSource,
    load_co2_table2_case,
)
from vib_sqd.encoding.direct import build_direct_encoded_vibrational_hamiltonian
from vib_sqd.operators.penalty import project_to_physical_subspace


def _exact_physical_gaps_from_case(case: str) -> np.ndarray:
    spec = load_co2_table2_case(case)
    n_modals = spec.vmax + 1

    enc = build_direct_encoded_vibrational_hamiltonian(
        spec.forcefield,
        vmax=spec.vmax,
        maxorder=4,
        tol=1e-10,
    )

    h_physical = project_to_physical_subspace(
        enc.hamiltonian_matrix,
        n_modes=spec.n_modes,
        n_modals=n_modals,
    )

    eigs = np.linalg.eigvalsh(h_physical)
    n_gaps = len(spec.reference_gaps_cm)
    return eigs[1 : 1 + n_gaps] - eigs[0]


def test_case_a_current_loader_is_explicitly_marked_approximate():
    spec = load_co2_table2_case("A")
    assert spec.source == ForceFieldSource.APPROXIMATE


def test_case_a_current_loader_does_not_yet_match_table2_reference_gaps():
    spec = load_co2_table2_case("A")
    gaps = _exact_physical_gaps_from_case("A")
    target = np.array(spec.reference_gaps_cm)

    assert gaps.shape == target.shape
    assert not np.allclose(gaps, target, atol=1.0)
    assert not np.isclose(gaps[0], gaps[1], atol=5.0)


def test_case_b_current_loader_is_explicitly_marked_approximate():
    spec = load_co2_table2_case("B")
    assert spec.source == ForceFieldSource.APPROXIMATE


def test_case_b_current_loader_does_not_yet_match_table2_reference_gaps():
    spec = load_co2_table2_case("B")
    gaps = _exact_physical_gaps_from_case("B")
    target = np.array(spec.reference_gaps_cm)

    assert gaps.shape == target.shape
    assert not np.allclose(gaps, target, atol=1.0)
    assert not np.isclose(gaps[0], gaps[1], atol=5.0)


def test_case_c_current_loader_is_explicitly_marked_approximate():
    spec = load_co2_table2_case("C")
    assert spec.source == ForceFieldSource.APPROXIMATE


def test_case_c_current_loader_does_not_yet_match_table2_reference_gaps():
    spec = load_co2_table2_case("C")
    gaps = _exact_physical_gaps_from_case("C")
    target = np.array(spec.reference_gaps_cm)

    assert gaps.shape == target.shape
    assert not np.allclose(gaps, target, atol=1.0)
