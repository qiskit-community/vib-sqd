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

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from vib_sqd.circuits.ansatz import get_ansatz_builder
from vib_sqd.circuits.trotter import build_initial_state_from_bitstring


def test_trotter_ansatz_builds():
    builder = get_ansatz_builder("trotter")
    ham = SparsePauliOp(["I"], coeffs=np.array([1.0]))
    psi0 = build_initial_state_from_bitstring("0")
    cfg = {"dt": 1.0, "n_steps": 1, "reps": 1, "order": 2}
    circs = builder.build(ham, psi0, cfg)
    assert isinstance(circs, list)
    assert len(circs) >= 1


def test_vucj_direct_builder_smoke():
    builder = get_ansatz_builder("vucj", encoding="direct")
    # build a dummy Hamiltonian with 12 qubits (3 modes x 4 modals)
    labels = ["I" * 12]
    ham = SparsePauliOp(labels, coeffs=np.array([0.0]))
    psi0 = build_initial_state_from_bitstring("100010001000")
    cfg = {"layers": 1, "n_modes": 3, "n_modals": 4}
    circs = builder.build(ham, psi0, cfg)
    assert isinstance(circs, list)
    assert len(circs) == 1
    qc = circs[0]
    assert qc.num_qubits == 12
    # measured circuits should have classical registers
    assert qc.num_clbits == qc.num_qubits
