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

"""Logical <-> absolute modal-index relabeling for excited-state VSCF
references.

The ground-state pipeline always encodes qubit (mode, modal=0) as the
occupied "reference" one-hot bit for every mode, and every other piece of
code (ansatz construction, amplitude arrays, configuration recovery) is
written in terms of that convention: logical index 0 is "the reference",
logical index a>=1 is "the a-th excited modal".

For an excited-state reference where mode i is occupied in some other
absolute modal r_i != 0 (e.g. built via MidasCpp's ``#3 Occup``), we reuse
every downstream piece of code UNCHANGED by relabeling modal indices per
mode so that logical position 0 always means "this mode's reference
occupation", regardless of which absolute modal that is. Logical positions
1..n_modals-1 are the remaining absolute modals in ascending order. This
reduces exactly to the identity map when occupation[mode] == 0 for every
mode, i.e. the ordinary ground-state case.

The actual reference *circuit* is therefore always
``build_vscf_reference_state(n_modes, n_modals, 0)`` (logical position 0),
independent of which excited state is being represented -- only classical
bookkeeping (amplitude extraction, decoding measured/recovered
configurations for Hamiltonian lookups) needs to know the relabeling.
"""

from __future__ import annotations

from collections.abc import Sequence


def modal_permutation_for_occupation(
    occupation: Sequence[int], n_modals: int
) -> list[list[int]]:
    """Per-mode logical-position -> absolute-modal-index map.

    ``perm[mode][0] == occupation[mode]`` (the reference); ``perm[mode][1:]``
    are the remaining absolute modals in ascending order.
    """
    for occ in occupation:
        if not (0 <= occ < n_modals):
            raise ValueError(
                f"occupation entry {occ} out of range for n_modals={n_modals}"
            )
    return [[occ] + sorted(set(range(n_modals)) - {occ}) for occ in occupation]


def logical_to_absolute_occupation(
    logical_indices: Sequence[int], occupation: Sequence[int], n_modals: int
) -> tuple[int, ...]:
    """Per-mode logical modal indices (0 = that mode's reference occupation)
    -> absolute modal indices."""
    perm = modal_permutation_for_occupation(occupation, n_modals)
    return tuple(perm[mode][idx] for mode, idx in enumerate(logical_indices))


def absolute_to_logical_occupation(
    absolute_indices: Sequence[int], occupation: Sequence[int], n_modals: int
) -> tuple[int, ...]:
    """Inverse of :func:`logical_to_absolute_occupation`."""
    perm = modal_permutation_for_occupation(occupation, n_modals)
    return tuple(perm[mode].index(idx) for mode, idx in enumerate(absolute_indices))
