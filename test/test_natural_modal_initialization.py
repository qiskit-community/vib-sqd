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
from qiskit.quantum_info import Statevector

from vib_sqd.initialization.natural_modal import (
    chain_angles_from_dominant_vector,
    insert_adjacent_same_modal_jastrow_angles_into_vlucj_x0,
    jastrow_angles_from_two_mode_statistics,
    joint_probabilities_from_two_mode_rdms,
    natural_modal_and_jastrow_initialization_from_statevector,
    natural_modal_initialization_from_statevector,
    one_mode_rdms_from_statevector,
    qiskit_one_hot_index_from_occupation,
    two_mode_rdms_from_statevector,
    vlucj_x0_from_chain_angles,
)


def test_qiskit_one_hot_index_from_occupation_one_mode_two_modals():
    assert qiskit_one_hot_index_from_occupation((0,), n_modals=2) == 1
    assert qiskit_one_hot_index_from_occupation((1,), n_modals=2) == 2


def test_one_mode_rdm_for_single_mode_superposition():
    theta = 0.3
    n_modes = 1
    n_modals = 2
    n_qubits = n_modes * n_modals

    state = np.zeros(2**n_qubits, dtype=complex)
    state[qiskit_one_hot_index_from_occupation((0,), n_modals)] = np.cos(theta)
    state[qiskit_one_hot_index_from_occupation((1,), n_modals)] = np.sin(theta)

    rdms = one_mode_rdms_from_statevector(
        Statevector(state),
        n_modes=n_modes,
        n_modals=n_modals,
    )

    expected = np.array(
        [
            [np.cos(theta) ** 2, np.cos(theta) * np.sin(theta)],
            [np.cos(theta) * np.sin(theta), np.sin(theta) ** 2],
        ],
        dtype=complex,
    )

    assert len(rdms) == 1
    assert np.allclose(rdms[0], expected)


def test_chain_angles_from_dominant_vector_two_modals():
    theta = 0.37
    vec = np.array([np.cos(theta), np.sin(theta)])

    angles = chain_angles_from_dominant_vector(vec)

    assert angles.shape == (1,)
    assert np.isclose(abs(angles[0]), theta, atol=1e-8)


def test_vlucj_x0_shape_for_three_modes_four_modals_one_layer():
    n_modes = 3
    n_modals = 4
    layers = 1

    chain_angles = tuple(np.zeros(n_modals - 1) for _ in range(n_modes))

    x0 = vlucj_x0_from_chain_angles(
        chain_angles=chain_angles,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        include_rz=True,
    )

    n_intramode_per_block = n_modes * (n_modals - 1)
    n_jastrow = (n_modes - 1) * n_modals
    n_rz = n_modes * n_modals
    expected_len = 2 * n_intramode_per_block + n_jastrow + n_rz

    assert x0.shape == (expected_len,)


def test_natural_modal_initialization_pipeline_two_modes_two_modals():
    theta0 = 0.2
    theta1 = 0.4
    n_modes = 2
    n_modals = 2
    n_qubits = n_modes * n_modals

    state = np.zeros(2**n_qubits, dtype=complex)

    # Product state:
    # mode 0: cos(theta0)|0> + sin(theta0)|1>
    # mode 1: cos(theta1)|0> + sin(theta1)|1>
    for m0 in range(n_modals):
        for m1 in range(n_modals):
            amp0 = np.cos(theta0) if m0 == 0 else np.sin(theta0)
            amp1 = np.cos(theta1) if m1 == 0 else np.sin(theta1)
            idx = qiskit_one_hot_index_from_occupation((m0, m1), n_modals)
            state[idx] = amp0 * amp1

    init = natural_modal_initialization_from_statevector(
        Statevector(state),
        n_modes=n_modes,
        n_modals=n_modals,
        vlucj_layers=1,
        include_rz=True,
    )

    assert len(init.rdms) == 2
    assert len(init.chain_angles) == 2
    assert np.isclose(abs(init.chain_angles[0][0]), theta0, atol=1e-8)
    assert np.isclose(abs(init.chain_angles[1][0]), theta1, atol=1e-8)
    assert init.vlucj_x0.ndim == 1


def test_two_mode_rdm_joint_probabilities_product_state():
    theta0 = 0.2
    theta1 = 0.4
    n_modes = 2
    n_modals = 2
    n_qubits = n_modes * n_modals

    state = np.zeros(2**n_qubits, dtype=complex)

    for m0 in range(n_modals):
        for m1 in range(n_modals):
            amp0 = np.cos(theta0) if m0 == 0 else np.sin(theta0)
            amp1 = np.cos(theta1) if m1 == 0 else np.sin(theta1)
            idx = qiskit_one_hot_index_from_occupation((m0, m1), n_modals)
            state[idx] = amp0 * amp1

    two_rdms = two_mode_rdms_from_statevector(
        Statevector(state),
        n_modes=n_modes,
        n_modals=n_modals,
    )

    joint = joint_probabilities_from_two_mode_rdms(two_rdms)
    p01 = joint[(0, 1)]

    expected = np.array(
        [
            [
                np.cos(theta0) ** 2 * np.cos(theta1) ** 2,
                np.cos(theta0) ** 2 * np.sin(theta1) ** 2,
            ],
            [
                np.sin(theta0) ** 2 * np.cos(theta1) ** 2,
                np.sin(theta0) ** 2 * np.sin(theta1) ** 2,
            ],
        ]
    )

    assert np.allclose(p01, expected)


def test_jastrow_angles_zero_for_uncorrelated_product_state():
    theta0 = 0.2
    theta1 = 0.4
    n_modes = 2
    n_modals = 2
    n_qubits = n_modes * n_modals

    state = np.zeros(2**n_qubits, dtype=complex)

    for m0 in range(n_modals):
        for m1 in range(n_modals):
            amp0 = np.cos(theta0) if m0 == 0 else np.sin(theta0)
            amp1 = np.cos(theta1) if m1 == 0 else np.sin(theta1)
            idx = qiskit_one_hot_index_from_occupation((m0, m1), n_modals)
            state[idx] = amp0 * amp1

    rdms = one_mode_rdms_from_statevector(
        Statevector(state),
        n_modes=n_modes,
        n_modals=n_modals,
    )
    two_rdms = two_mode_rdms_from_statevector(
        Statevector(state),
        n_modes=n_modes,
        n_modals=n_modals,
    )

    angles = jastrow_angles_from_two_mode_statistics(
        one_rdms=rdms,
        two_rdms=two_rdms,
        n_modes=n_modes,
        n_modals=n_modals,
        method="connected",
        scale=1.0,
        clip=1.0,
    )

    assert np.allclose(angles[(0, 1)], 0.0, atol=1e-10)


def test_insert_jastrow_angles_into_vlucj_x0_two_modes_three_modals():
    n_modes = 2
    n_modals = 3
    layers = 1

    chain_angles = tuple(np.zeros(n_modals - 1) for _ in range(n_modes))
    x0 = vlucj_x0_from_chain_angles(
        chain_angles=chain_angles,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        include_rz=True,
    )

    mat = np.zeros((n_modals, n_modals))
    mat[0, 0] = 0.1
    mat[1, 1] = 0.2
    mat[2, 2] = 0.3

    x1 = insert_adjacent_same_modal_jastrow_angles_into_vlucj_x0(
        x0=x0,
        jastrow_angles={(0, 1): mat},
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        include_rz=True,
    )

    n_intramode_per_block = n_modes * (n_modals - 1)
    jastrow_start = n_intramode_per_block

    assert np.isclose(x1[jastrow_start + 0], 0.1)
    assert np.isclose(x1[jastrow_start + 1], 0.2)
    assert np.isclose(x1[jastrow_start + 2], 0.3)


def test_natural_modal_and_jastrow_init_shape():
    n_modes = 2
    n_modals = 3
    layers = 1
    n_qubits = n_modes * n_modals

    state = np.zeros(2**n_qubits, dtype=complex)
    state[qiskit_one_hot_index_from_occupation((0, 0), n_modals)] = 0.9
    state[qiskit_one_hot_index_from_occupation((1, 1), n_modals)] = 0.3
    state = state / np.linalg.norm(state)

    init = natural_modal_and_jastrow_initialization_from_statevector(
        Statevector(state),
        n_modes=n_modes,
        n_modals=n_modals,
        vlucj_layers=layers,
        include_rz=True,
        jastrow_method="connected",
        jastrow_scale=0.1,
        jastrow_clip=0.3,
    )

    n_intramode_per_block = n_modes * (n_modals - 1)
    n_jastrow = (n_modes - 1) * n_modals
    n_rz = n_modes * n_modals
    expected = 2 * n_intramode_per_block + n_jastrow + n_rz

    assert init.vlucj_x0.shape == (expected,)
