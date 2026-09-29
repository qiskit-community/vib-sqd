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

"""Logical <-> absolute modal-index relabeling used to reuse the
ground-state ansatz/amplitude code for excited-state VSCF references
(vib_sqd.basis.modal_occupation)."""

import itertools

import pytest

from vib_sqd.basis.modal_occupation import (
    absolute_to_logical_occupation,
    logical_to_absolute_occupation,
    modal_permutation_for_occupation,
)


def test_ground_state_occupation_is_identity():
    n_modes, n_modals = 6, 4
    occupation = [0] * n_modes
    perm = modal_permutation_for_occupation(occupation, n_modals)
    assert perm == [list(range(n_modals))] * n_modes


def test_reference_position_is_always_logical_zero():
    occupation = [0, 2, 1, 3]
    n_modals = 4
    logical = absolute_to_logical_occupation(tuple(occupation), occupation, n_modals)
    assert logical == (0,) * len(occupation)


def test_logical_absolute_round_trip():
    n_modes, n_modals = 5, 4
    occupation = [0, 3, 1, 0, 2]
    for absolute in itertools.product(range(n_modals), repeat=n_modes):
        logical = absolute_to_logical_occupation(absolute, occupation, n_modals)
        assert logical_to_absolute_occupation(logical, occupation, n_modals) == absolute


def test_logical_zero_always_maps_to_reference_modal():
    n_modes, n_modals = 4, 5
    occupation = [4, 0, 2, 1]
    absolute = logical_to_absolute_occupation([0] * n_modes, occupation, n_modals)
    assert absolute == tuple(occupation)


def test_out_of_range_reference_raises():
    with pytest.raises(ValueError):
        modal_permutation_for_occupation([5], 4)
