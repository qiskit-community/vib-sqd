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
from qiskit.circuit import ParameterVector

from vib_sqd.circuits.ansatz_enhanced import append_givens_rotation_native


def build_chc_ansatz(
    initial_state: QuantumCircuit,
    n_modes: int,
    n_modals: int,
    layers: int = 1,
) -> QuantumCircuit:
    """
    Compact heuristic chemistry-inspired ansatz for direct vibrational encoding.

    Important:
        This is a compact CHC-like comparison ansatz, not a guaranteed exact
        reproduction of the phase-optimized CHC blocks from Fig. 1 of the paper.

    Structure per layer:
        1. Singles-like intramode compact Givens blocks.
        2. Compact pair-correlation phase blocks between excited modals.
        3. Second singles-like intramode compact Givens blocks.

    Why this is useful:
        - It is much shallower than full UVCCSD.
        - It is parameterized.
        - It preserves a useful comparison point against VLUCJ and UVCCSD.
        - It exposes the same API expected by the runner:
              build_chc_ansatz(initial_state, n_modes, n_modals, layers)
    """
    if n_modes <= 0:
        raise ValueError("n_modes must be positive.")
    if n_modals <= 1:
        raise ValueError("n_modals must be at least 2.")

    n_qubits = n_modes * n_modals

    n_single_per_block = n_modes * (n_modals - 1)
    n_pair_phase = (n_modes * (n_modes - 1) // 2) * (n_modals - 1) ** 2
    params_per_layer = 2 * n_single_per_block + n_pair_phase

    params = ParameterVector("chc", layers * params_per_layer)

    qc = QuantumCircuit(n_qubits)
    qc.compose(initial_state, inplace=True)

    mode_qubits = [
        [mode * n_modals + modal for modal in range(n_modals)]
        for mode in range(n_modes)
    ]

    p = 0

    for _layer in range(layers):
        # First compact singles block.
        for mode in range(n_modes):
            q_ref = mode_qubits[mode][0]
            for modal in range(1, n_modals):
                q_exc = mode_qubits[mode][modal]
                append_givens_rotation_native(
                    qc,
                    q_ref=q_ref,
                    q_exc=q_exc,
                    theta=params[p],
                )
                p += 1

        # Compact pair-correlation phase block.
        #
        # This is not the exact 8-term UVCC double excitation. It is a compact
        # CHC-like pair correlation intended for comparison.
        for mode1 in range(n_modes):
            for mode2 in range(mode1 + 1, n_modes):
                for modal1 in range(1, n_modals):
                    for modal2 in range(1, n_modals):
                        q1 = mode_qubits[mode1][modal1]
                        q2 = mode_qubits[mode2][modal2]
                        qc.cp(params[p], q1, q2)
                        p += 1

        # Second compact singles block.
        for mode in range(n_modes):
            q_ref = mode_qubits[mode][0]
            for modal in range(1, n_modals):
                q_exc = mode_qubits[mode][modal]
                append_givens_rotation_native(
                    qc,
                    q_ref=q_ref,
                    q_exc=q_exc,
                    theta=params[p],
                )
                p += 1

    return qc
