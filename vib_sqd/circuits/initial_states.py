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

from qiskit import QuantumCircuit


def build_vscf_reference_state(
    n_modes: int,
    n_modals: int,
    occupied_modal: int = 0,
) -> QuantumCircuit:
    """
    Direct encoding VSCF reference:
        qubit = mode * n_modals + occupied_modal
    """
    if n_modes <= 0:
        raise ValueError("n_modes must be positive.")
    if n_modals <= 0:
        raise ValueError("n_modals must be positive.")
    if occupied_modal < 0 or occupied_modal >= n_modals:
        raise ValueError("occupied_modal out of range.")

    qc = QuantumCircuit(n_modes * n_modals)
    for mode in range(n_modes):
        qc.x(mode * n_modals + occupied_modal)
    return qc
