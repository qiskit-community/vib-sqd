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
Core tests for direct encoding functionality.

Tests:
- q_power_matrix construction
- number_operator construction
- embed_single_mode_operator_direct
"""

import numpy as np
import pytest

from vib_sqd.basis.harmonic import (
    number_operator,
    q_power_matrix,
)
from vib_sqd.encoding.direct import (
    embed_single_mode_operator_direct,
)


def test_q_power_matrix_shape_and_symmetry():
    """Test q_power_matrix returns correct shape and is symmetric."""
    vmax = 3
    n_modals = vmax + 1

    # Power 1 (position operator)
    q1 = q_power_matrix(vmax, power=1)
    assert q1.shape == (n_modals, n_modals)
    assert np.allclose(q1, q1.T), "q should be symmetric"

    # Power 2
    q2 = q_power_matrix(vmax, power=2)
    assert q2.shape == (n_modals, n_modals)
    assert np.allclose(q2, q2.T), "q^2 should be symmetric"


def test_q_power_matrix_analytic_values():
    """Test q_power_matrix matches known analytic values for low-lying states."""
    vmax = 3

    # q = (a + a†)/sqrt(2)
    # <0|q|1> = 1/sqrt(2), <1|q|0> = 1/sqrt(2)
    # <1|q|2> = sqrt(2)/sqrt(2) = 1
    q1 = q_power_matrix(vmax, power=1)

    assert np.isclose(q1[0, 1], 1.0 / np.sqrt(2)), "q matrix element <0|q|1> incorrect"
    assert np.isclose(q1[1, 0], 1.0 / np.sqrt(2)), "q matrix element <1|q|0> incorrect"
    assert np.isclose(q1[1, 2], 1.0), "q matrix element <1|q|2> incorrect"


def test_number_operator_diagonal():
    """Test number_operator is diagonal with correct eigenvalues."""
    vmax = 3
    n_modals = vmax + 1

    n_op = number_operator(vmax)
    assert n_op.shape == (n_modals, n_modals)

    # Should be diagonal
    assert np.allclose(n_op, np.diag(np.diag(n_op))), "n should be diagonal"

    # Diagonal should be [0, 1, 2, 3]
    expected = np.arange(n_modals)
    assert np.allclose(np.diag(n_op), expected), "n eigenvalues should be [0,1,2,3]"


def test_embed_single_mode_operator_direct_identity():
    """Test embedding identity operator in a single mode."""
    vmax = 1
    n_modals = vmax + 1  # 2 modals per mode

    # Identity matrix for one mode
    identity = np.eye(n_modals)

    # Embed to qubit space
    embedded = embed_single_mode_operator_direct(identity)

    # Should be numpy array
    assert isinstance(embedded, np.ndarray)

    # Should be 2^n_modals x 2^n_modals
    expected_dim = 2**n_modals
    assert embedded.shape == (expected_dim, expected_dim)

    # For identity on one-hot states, should preserve those states
    # One-hot indices for n_modals=2: [2, 1] (binary 10, 01)
    one_hot_indices = [2, 1]
    for idx in one_hot_indices:
        assert np.isclose(
            embedded[idx, idx], 1.0
        ), f"Identity should preserve state {idx}"


def test_embed_single_mode_operator_direct_simple_matrix():
    """Test embedding a simple 2x2 matrix."""
    n_modals = 2

    # Simple 2x2 matrix
    mat = np.array([[1.0, 0.5], [0.5, 2.0]])

    # Embed to qubit space
    embedded = embed_single_mode_operator_direct(mat)

    # Should be 4x4 (2^2)
    assert embedded.shape == (4, 4)

    # One-hot indices: modal 0 -> |10> -> 2, modal 1 -> |01> -> 1
    # Check that mat[0,0] maps to embedded[2,2]
    assert np.isclose(embedded[2, 2], mat[0, 0])
    # Check that mat[0,1] maps to embedded[2,1]
    assert np.isclose(embedded[2, 1], mat[0, 1])
    # Check that mat[1,0] maps to embedded[1,2]
    assert np.isclose(embedded[1, 2], mat[1, 0])
    # Check that mat[1,1] maps to embedded[1,1]
    assert np.isclose(embedded[1, 1], mat[1, 1])


def test_q_power_matrix_power_zero_is_identity():
    """Test that q^0 returns identity matrix."""
    vmax = 2
    n_modals = vmax + 1

    q0 = q_power_matrix(vmax, power=0)
    expected = np.eye(n_modals)

    assert np.allclose(q0, expected), "q^0 should be identity"


def test_number_operator_commutes_with_itself():
    """Test that [n, n] = 0 (trivial but good sanity check)."""
    vmax = 2
    n_op = number_operator(vmax)

    commutator = n_op @ n_op - n_op @ n_op
    assert np.allclose(commutator, 0), "[n, n] should be zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
