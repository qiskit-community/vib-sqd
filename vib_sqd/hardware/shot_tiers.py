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

"""Qubit-count-tiered shot budget for the ibm_kingston hardware campaign.

Wider circuits need more shots to keep the one-hot-retained sample count
usable after post-selection (retention drops roughly exponentially with
circuit depth/qubit count under a fixed per-qubit error rate), so shots scale
with qubit count in three tiers rather than using one flat value across every
molecule/modal-count cell."""


def shots_for_qubit_count(nq: int) -> int:
    """nq <= 24 -> 100_000; 24 < nq <= 56 -> 500_000; 56 < nq <= 72 -> 1_000_000.
    Boundary qubit counts (24, 56) belong to the lower tier."""
    if nq <= 24:
        return 100_000
    if nq <= 56:
        return 500_000
    if nq <= 72:
        return 1_000_000
    raise ValueError(f"no shot tier defined for nq={nq} (>72 qubits)")
