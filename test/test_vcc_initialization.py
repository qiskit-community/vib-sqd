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
Tests for VCC amplitude to quantum parameter mapping.

These tests verify mapper shapes, scaling behavior, dimension validation,
and edge cases. Separate circuit-integration tests should verify that
len(x0) == circuit.num_parameters for each ansatz builder.
"""

import numpy as np
import pytest

from vib_sqd.classical.vcc_solver import VCCAmplitudes
from vib_sqd.initialization.vcc_initialization import (
    _scale_amplitudes,
    _validate_system_size,
    chc_num_params_per_layer,
    map_vccsd_to_chc_params,
    map_vccsd_to_uvccsd_initial_point,
    map_vccsd_to_uvccsd_params,
    map_vccsd_to_vlucj_params,
    vlucj_num_params_per_layer,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_amplitudes():
    """Create simple test amplitudes for 2 modes, 3 modals."""
    rng = np.random.default_rng(123)
    n_modes = 2
    n_modals = 3
    n_exc = n_modals - 1

    t1 = rng.normal(scale=0.1, size=(n_modes, n_exc))
    t2 = rng.normal(scale=0.05, size=(n_modes, n_modes, n_exc, n_exc))

    return VCCAmplitudes(
        t1=t1,
        t2=t2,
        energy=-10.5,
        converged=True,
        iterations=15,
    )


@pytest.fixture
def larger_amplitudes():
    """Create test amplitudes for 3 modes, 4 modals."""
    rng = np.random.default_rng(456)
    n_modes = 3
    n_modals = 4
    n_exc = n_modals - 1

    t1 = rng.normal(scale=0.1, size=(n_modes, n_exc))
    t2 = rng.normal(scale=0.05, size=(n_modes, n_modes, n_exc, n_exc))

    return VCCAmplitudes(
        t1=t1,
        t2=t2,
        energy=-15.2,
        converged=True,
        iterations=20,
    )


@pytest.fixture
def deterministic_amplitudes():
    """Create deterministic amplitudes for exact ordering tests."""
    n_modes = 2
    n_modals = 3
    n_exc = n_modals - 1

    t1 = np.array(
        [
            [0.2, -0.4],
            [0.6, -0.8],
        ]
    )

    t2 = np.zeros((n_modes, n_modes, n_exc, n_exc))
    t2[0, 1, 0, 0] = 0.1
    t2[0, 1, 0, 1] = 0.2
    t2[0, 1, 1, 0] = 0.3
    t2[0, 1, 1, 1] = 0.4

    return VCCAmplitudes(t1=t1, t2=t2, energy=-1.0, converged=True, iterations=10)


# ============================================================================
# Test _validate_system_size
# ============================================================================


def test_validate_system_size_valid():
    """Valid system sizes should not raise."""
    _validate_system_size(n_modes=1, n_modals=2)
    _validate_system_size(n_modes=3, n_modals=4)
    _validate_system_size(n_modes=10, n_modals=5)


def test_validate_system_size_invalid_modes():
    """n_modes <= 0 should raise."""
    with pytest.raises(ValueError, match="n_modes must be positive"):
        _validate_system_size(n_modes=0, n_modals=3)

    with pytest.raises(ValueError, match="n_modes must be positive"):
        _validate_system_size(n_modes=-1, n_modals=3)


def test_validate_system_size_invalid_modals():
    """n_modals < 2 should raise."""
    with pytest.raises(ValueError, match="n_modals must be at least 2"):
        _validate_system_size(n_modes=3, n_modals=1)

    with pytest.raises(ValueError, match="n_modals must be at least 2"):
        _validate_system_size(n_modes=3, n_modals=0)


# ============================================================================
# Test _scale_amplitudes
# ============================================================================


def test_scale_amplitudes_direct():
    """Direct scaling should return copy."""
    arr = np.array([0.1, -0.2, 0.3])
    scaled = _scale_amplitudes(arr, strategy="direct")
    np.testing.assert_allclose(scaled, arr)
    assert scaled is not arr  # Should be a copy


def test_scale_amplitudes_sqrt():
    """Sqrt scaling should preserve sign and reduce magnitude."""
    arr = np.array([0.04, -0.09, 0.16])
    scaled = _scale_amplitudes(arr, strategy="sqrt")
    expected = np.array([0.2, -0.3, 0.4])
    np.testing.assert_allclose(scaled, expected)


def test_scale_amplitudes_tanh():
    """Tanh scaling should bound large values."""
    arr = np.array([0.5, -2.0, 5.0])
    scaled = _scale_amplitudes(arr, strategy="tanh")
    assert np.all(np.abs(scaled) <= 1.0)
    np.testing.assert_allclose(scaled, np.tanh(arr))


def test_scale_amplitudes_normalized():
    """Normalized scaling should divide by max abs."""
    arr = np.array([0.1, -0.5, 0.3])
    scaled = _scale_amplitudes(arr, strategy="normalized")
    expected = arr / 0.5
    np.testing.assert_allclose(scaled, expected)


def test_scale_amplitudes_normalized_zero():
    """Normalized scaling with all zeros should return zeros."""
    arr = np.zeros(5)
    scaled = _scale_amplitudes(arr, strategy="normalized")
    np.testing.assert_allclose(scaled, arr)


def test_scale_amplitudes_none():
    """None strategy should return zeros."""
    arr = np.array([0.1, -0.2, 0.3])
    scaled = _scale_amplitudes(arr, strategy="none")
    np.testing.assert_allclose(scaled, np.zeros_like(arr))


def test_scale_amplitudes_clip():
    """Clipping should bound output."""
    arr = np.array([0.1, -0.5, 0.8])
    scaled = _scale_amplitudes(arr, strategy="direct", clip=0.3)
    expected = np.array([0.1, -0.3, 0.3])
    np.testing.assert_allclose(scaled, expected)


def test_scale_amplitudes_invalid_strategy():
    """Unknown strategy should raise."""
    arr = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="Unknown scaling strategy"):
        _scale_amplitudes(arr, strategy="invalid")  # type: ignore[arg-type]


def test_scale_amplitudes_non_finite():
    """Non-finite values should raise."""
    arr = np.array([0.1, np.nan, 0.3])
    with pytest.raises(ValueError, match="non-finite values"):
        _scale_amplitudes(arr, strategy="direct")

    arr = np.array([0.1, np.inf, 0.3])
    with pytest.raises(ValueError, match="non-finite values"):
        _scale_amplitudes(arr, strategy="direct")


def test_scale_amplitudes_negative_clip():
    """Negative clip should raise."""
    arr = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="clip must be non-negative"):
        _scale_amplitudes(arr, strategy="direct", clip=-0.5)


# ============================================================================
# Test UVCCSD mapping
# ============================================================================


def test_map_vccsd_to_uvccsd_params(simple_amplitudes):
    """UVCCSD params should return dict with t1 and t2."""
    params = map_vccsd_to_uvccsd_params(simple_amplitudes)

    assert "t1" in params
    assert "t2" in params
    assert params["t1"].shape == simple_amplitudes.t1.shape
    assert params["t2"].shape == simple_amplitudes.t2.shape


def test_map_vccsd_to_uvccsd_initial_point_shape(simple_amplitudes):
    """UVCCSD initial point should have correct length."""
    n_modes = 2
    n_modals = 3
    n_exc = n_modals - 1

    x0 = map_vccsd_to_uvccsd_initial_point(
        simple_amplitudes, n_modes=n_modes, n_modals=n_modals
    )

    n_singles = n_modes * n_exc
    n_doubles = (n_modes * (n_modes - 1) // 2) * n_exc * n_exc
    expected_len = n_singles + n_doubles

    assert len(x0) == expected_len


def test_uvccsd_initial_point_ordering():
    """UVCCSD initial point should have correct parameter ordering."""
    n_modes = 2
    n_modals = 3
    n_exc = 2

    t1 = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    t2 = np.zeros((n_modes, n_modes, n_exc, n_exc))
    t2[0, 1, 0, 0] = 10.0
    t2[0, 1, 0, 1] = 20.0
    t2[0, 1, 1, 0] = 30.0
    t2[0, 1, 1, 1] = 40.0

    amps = VCCAmplitudes(t1=t1, t2=t2, energy=-10.0, converged=True, iterations=10)

    x0 = map_vccsd_to_uvccsd_initial_point(
        amps,
        n_modes=n_modes,
        n_modals=n_modals,
        scaling_strategy="direct",
        clip=None,
    )

    # Expected order: all singles first, then all doubles
    expected = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,  # Singles
            10.0,
            20.0,
            30.0,
            40.0,  # Doubles
        ]
    )

    np.testing.assert_allclose(x0, expected)


def test_map_vccsd_to_uvccsd_initial_point_dimension_mismatch(simple_amplitudes):
    """Dimension mismatch should raise."""
    with pytest.raises(ValueError, match="n_modes mismatch"):
        map_vccsd_to_uvccsd_initial_point(simple_amplitudes, n_modes=3, n_modals=3)

    with pytest.raises(ValueError, match="n_modals mismatch"):
        map_vccsd_to_uvccsd_initial_point(simple_amplitudes, n_modes=2, n_modals=4)


def test_map_vccsd_to_uvccsd_initial_point_invalid_system_size(simple_amplitudes):
    """Invalid system size should raise."""
    with pytest.raises(ValueError, match="n_modes must be positive"):
        map_vccsd_to_uvccsd_initial_point(simple_amplitudes, n_modes=0, n_modals=3)

    with pytest.raises(ValueError, match="n_modals must be at least 2"):
        map_vccsd_to_uvccsd_initial_point(simple_amplitudes, n_modes=2, n_modals=1)


def test_map_vccsd_to_uvccsd_initial_point_tanh_scaling_bounds(simple_amplitudes):
    """Tanh scaling should bound all values to [-1, 1]."""
    x0_tanh = map_vccsd_to_uvccsd_initial_point(
        simple_amplitudes,
        n_modes=2,
        n_modals=3,
        scaling_strategy="tanh",
    )

    assert np.all(np.abs(x0_tanh) <= 1.0)


# ============================================================================
# Test VLUCJ mapping
# ============================================================================


def test_vlucj_num_params_per_layer():
    """VLUCJ parameter count should be correct."""
    n_modes = 3
    n_modals = 4

    n_singles = n_modes * (n_modals - 1)
    n_jastrow = (n_modes - 1) * n_modals
    n_rz = n_modes * n_modals

    # With RZ
    n_params = vlucj_num_params_per_layer(n_modes, n_modals, include_rz=True)
    assert n_params == 2 * n_singles + n_jastrow + n_rz

    # Without RZ
    n_params = vlucj_num_params_per_layer(n_modes, n_modals, include_rz=False)
    assert n_params == 2 * n_singles + n_jastrow


def test_vlucj_num_params_per_layer_invalid_system_size():
    """Invalid system size should raise."""
    with pytest.raises(ValueError, match="n_modes must be positive"):
        vlucj_num_params_per_layer(n_modes=0, n_modals=3)

    with pytest.raises(ValueError, match="n_modals must be at least 2"):
        vlucj_num_params_per_layer(n_modes=3, n_modals=1)


def test_map_vccsd_to_vlucj_params_shape(larger_amplitudes):
    """VLUCJ initial point should have correct shape."""
    n_modes = 3
    n_modals = 4
    layers = 4

    x0 = map_vccsd_to_vlucj_params(
        larger_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        include_rz=True,
    )

    params_per_layer = vlucj_num_params_per_layer(n_modes, n_modals, include_rz=True)
    expected_len = layers * params_per_layer

    assert len(x0) == expected_len


def test_vlucj_splits_t1_across_two_givens_blocks(deterministic_amplitudes):
    """VLUCJ should split T1 amplitudes across both Givens blocks (0.5x each)."""
    n_modes = 2
    n_modals = 3
    layers = 1

    x0 = map_vccsd_to_vlucj_params(
        deterministic_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        include_rz=False,
        distribute_over_layers=True,
        scaling_strategy="direct",
        clip=None,
    )

    n_singles = n_modes * (n_modals - 1)
    n_jastrow = (n_modes - 1) * n_modals

    first_givens = x0[:n_singles]
    second_start = n_singles + n_jastrow
    second_givens = x0[second_start : second_start + n_singles]

    # Both blocks should have 0.5x T1
    expected = 0.5 * deterministic_amplitudes.t1.reshape(-1)

    np.testing.assert_allclose(first_givens, expected)
    np.testing.assert_allclose(second_givens, expected)


def test_vlucj_jastrow_diagonal_ordering(deterministic_amplitudes):
    """VLUCJ Jastrow block should use diagonal adjacent T2 terms."""
    n_modes = 2
    n_modals = 3
    layers = 1

    x0 = map_vccsd_to_vlucj_params(
        deterministic_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        include_rz=False,
        distribute_over_layers=True,
        scaling_strategy="direct",
        clip=None,
    )

    n_singles = n_modes * (n_modals - 1)
    n_jastrow = (n_modes - 1) * n_modals

    jastrow_block = x0[n_singles : n_singles + n_jastrow]

    # For adjacent modes 0-1, modals 0,1,2:
    # modal 0 (reference) → 0.0
    # modal 1 → t2[0, 1, 0, 0] = 0.1
    # modal 2 → t2[0, 1, 1, 1] = 0.4
    expected = np.array([0.0, 0.1, 0.4])

    np.testing.assert_allclose(jastrow_block, expected)


def test_map_vccsd_to_vlucj_params_zero_layers(larger_amplitudes):
    """Zero layers should raise."""
    with pytest.raises(ValueError, match="layers must be positive"):
        map_vccsd_to_vlucj_params(larger_amplitudes, n_modes=3, n_modals=4, layers=0)


def test_map_vccsd_to_vlucj_params_negative_layers(larger_amplitudes):
    """Negative layers should raise."""
    with pytest.raises(ValueError, match="layers must be positive"):
        map_vccsd_to_vlucj_params(larger_amplitudes, n_modes=3, n_modals=4, layers=-1)


def test_map_vccsd_to_vlucj_params_dimension_mismatch(larger_amplitudes):
    """Dimension mismatch should raise."""
    with pytest.raises(ValueError, match="n_modes mismatch"):
        map_vccsd_to_vlucj_params(larger_amplitudes, n_modes=2, n_modals=4, layers=2)

    with pytest.raises(ValueError, match="n_modals mismatch"):
        map_vccsd_to_vlucj_params(larger_amplitudes, n_modes=3, n_modals=3, layers=2)


def test_map_vccsd_to_vlucj_params_distribute_over_layers(larger_amplitudes):
    """Distributing over layers should scale amplitudes."""
    n_modes = 3
    n_modals = 4
    layers = 4

    x0_distributed = map_vccsd_to_vlucj_params(
        larger_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        distribute_over_layers=True,
    )

    x0_not_distributed = map_vccsd_to_vlucj_params(
        larger_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        distribute_over_layers=False,
    )

    # Distributed should have smaller magnitude
    assert np.max(np.abs(x0_distributed)) < np.max(np.abs(x0_not_distributed))


def test_map_vccsd_to_vlucj_params_rz_block_zero(larger_amplitudes):
    """RZ block should remain zero."""
    n_modes = 3
    n_modals = 4
    layers = 2

    x0 = map_vccsd_to_vlucj_params(
        larger_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        include_rz=True,
    )

    params_per_layer = vlucj_num_params_per_layer(n_modes, n_modals, include_rz=True)
    n_singles = n_modes * (n_modals - 1)
    n_jastrow = (n_modes - 1) * n_modals
    n_rz = n_modes * n_modals

    # Check RZ block in each layer
    for layer in range(layers):
        base = layer * params_per_layer
        rz_start = base + 2 * n_singles + n_jastrow
        rz_end = rz_start + n_rz

        rz_block = x0[rz_start:rz_end]
        np.testing.assert_allclose(rz_block, np.zeros(n_rz))


# ============================================================================
# Test CHC mapping
# ============================================================================


def test_chc_num_params_per_layer():
    """CHC parameter count should be correct."""
    n_modes = 3
    n_modals = 4

    n_singles = n_modes * (n_modals - 1)
    n_pairs = n_modes * (n_modes - 1) // 2
    n_pair_corr = n_pairs * (n_modals - 1) ** 2

    n_params = chc_num_params_per_layer(n_modes, n_modals)
    assert n_params == 2 * n_singles + n_pair_corr


def test_chc_num_params_per_layer_invalid_system_size():
    """Invalid system size should raise."""
    with pytest.raises(ValueError, match="n_modes must be positive"):
        chc_num_params_per_layer(n_modes=0, n_modals=3)

    with pytest.raises(ValueError, match="n_modals must be at least 2"):
        chc_num_params_per_layer(n_modes=3, n_modals=1)


def test_map_vccsd_to_chc_params_shape(larger_amplitudes):
    """CHC initial point should have correct shape."""
    n_modes = 3
    n_modals = 4
    layers = 4

    x0 = map_vccsd_to_chc_params(
        larger_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
    )

    params_per_layer = chc_num_params_per_layer(n_modes, n_modals)
    expected_len = layers * params_per_layer

    assert len(x0) == expected_len


def test_chc_splits_t1_across_two_singles_blocks(deterministic_amplitudes):
    """CHC should split T1 amplitudes across both singles blocks (0.5x each)."""
    n_modes = 2
    n_modals = 3
    layers = 1

    x0 = map_vccsd_to_chc_params(
        deterministic_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        distribute_over_layers=True,
        scaling_strategy="direct",
        clip=None,
    )

    n_singles = n_modes * (n_modals - 1)
    n_pair_corr = (n_modes * (n_modes - 1) // 2) * (n_modals - 1) ** 2

    first_singles = x0[:n_singles]
    second_start = n_singles + n_pair_corr
    second_singles = x0[second_start : second_start + n_singles]

    # Both blocks should have 0.5x T1
    expected = 0.5 * deterministic_amplitudes.t1.reshape(-1)

    np.testing.assert_allclose(first_singles, expected)
    np.testing.assert_allclose(second_singles, expected)


def test_chc_pair_correlation_ordering(deterministic_amplitudes):
    """CHC pair-correlation block should have correct T2 ordering."""
    n_modes = 2
    n_modals = 3
    layers = 1

    x0 = map_vccsd_to_chc_params(
        deterministic_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        distribute_over_layers=True,
        scaling_strategy="direct",
        clip=None,
    )

    n_singles = n_modes * (n_modals - 1)
    n_pair_corr = (n_modes * (n_modes - 1) // 2) * (n_modals - 1) ** 2

    pair_block = x0[n_singles : n_singles + n_pair_corr]

    # For mode pair (0,1), excitations (0,0), (0,1), (1,0), (1,1):
    # t2[0, 1, 0, 0] = 0.1
    # t2[0, 1, 0, 1] = 0.2
    # t2[0, 1, 1, 0] = 0.3
    # t2[0, 1, 1, 1] = 0.4
    expected_pair_block = np.array([0.1, 0.2, 0.3, 0.4])

    np.testing.assert_allclose(pair_block, expected_pair_block)


def test_map_vccsd_to_chc_params_zero_layers(larger_amplitudes):
    """Zero layers should raise."""
    with pytest.raises(ValueError, match="layers must be positive"):
        map_vccsd_to_chc_params(larger_amplitudes, n_modes=3, n_modals=4, layers=0)


def test_map_vccsd_to_chc_params_negative_layers(larger_amplitudes):
    """Negative layers should raise."""
    with pytest.raises(ValueError, match="layers must be positive"):
        map_vccsd_to_chc_params(larger_amplitudes, n_modes=3, n_modals=4, layers=-1)


def test_map_vccsd_to_chc_params_dimension_mismatch(larger_amplitudes):
    """Dimension mismatch should raise."""
    with pytest.raises(ValueError, match="n_modes mismatch"):
        map_vccsd_to_chc_params(larger_amplitudes, n_modes=2, n_modals=4, layers=2)

    with pytest.raises(ValueError, match="n_modals mismatch"):
        map_vccsd_to_chc_params(larger_amplitudes, n_modes=3, n_modals=3, layers=2)


def test_map_vccsd_to_chc_params_distribute_over_layers(larger_amplitudes):
    """Distributing over layers should scale amplitudes."""
    n_modes = 3
    n_modals = 4
    layers = 4

    x0_distributed = map_vccsd_to_chc_params(
        larger_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        distribute_over_layers=True,
    )

    x0_not_distributed = map_vccsd_to_chc_params(
        larger_amplitudes,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        distribute_over_layers=False,
    )

    # Distributed should have smaller magnitude
    assert np.max(np.abs(x0_distributed)) < np.max(np.abs(x0_not_distributed))


# ============================================================================
# Integration tests: Verify parameter counts match expected circuit sizes
# ============================================================================


def test_uvccsd_parameter_count_consistency():
    """UVCCSD parameter count should match expected circuit size."""
    rng = np.random.default_rng(789)
    n_modes = 3
    n_modals = 4
    n_exc = n_modals - 1

    # Create dummy amplitudes
    t1 = rng.normal(scale=0.1, size=(n_modes, n_exc))
    t2 = rng.normal(scale=0.05, size=(n_modes, n_modes, n_exc, n_exc))
    amps = VCCAmplitudes(t1=t1, t2=t2, energy=-10.0, converged=True, iterations=10)

    x0 = map_vccsd_to_uvccsd_initial_point(amps, n_modes=n_modes, n_modals=n_modals)

    # Expected parameter count
    n_singles = n_modes * n_exc
    n_doubles = (n_modes * (n_modes - 1) // 2) * n_exc * n_exc
    expected_params = n_singles + n_doubles

    assert len(x0) == expected_params


def test_vlucj_parameter_count_consistency():
    """VLUCJ parameter count should match expected circuit size."""
    rng = np.random.default_rng(101112)
    n_modes = 3
    n_modals = 4
    layers = 4
    n_exc = n_modals - 1

    # Create dummy amplitudes
    t1 = rng.normal(scale=0.1, size=(n_modes, n_exc))
    t2 = rng.normal(scale=0.05, size=(n_modes, n_modes, n_exc, n_exc))
    amps = VCCAmplitudes(t1=t1, t2=t2, energy=-10.0, converged=True, iterations=10)

    x0 = map_vccsd_to_vlucj_params(
        amps, n_modes=n_modes, n_modals=n_modals, layers=layers, include_rz=True
    )

    # Expected parameter count
    params_per_layer = vlucj_num_params_per_layer(n_modes, n_modals, include_rz=True)
    expected_params = layers * params_per_layer

    assert len(x0) == expected_params


def test_chc_parameter_count_consistency():
    """CHC parameter count should match expected circuit size."""
    rng = np.random.default_rng(131415)
    n_modes = 3
    n_modals = 4
    layers = 4
    n_exc = n_modals - 1

    # Create dummy amplitudes
    t1 = rng.normal(scale=0.1, size=(n_modes, n_exc))
    t2 = rng.normal(scale=0.05, size=(n_modes, n_modes, n_exc, n_exc))
    amps = VCCAmplitudes(t1=t1, t2=t2, energy=-10.0, converged=True, iterations=10)

    x0 = map_vccsd_to_chc_params(
        amps, n_modes=n_modes, n_modals=n_modals, layers=layers
    )

    # Expected parameter count
    params_per_layer = chc_num_params_per_layer(n_modes, n_modals)
    expected_params = layers * params_per_layer

    assert len(x0) == expected_params


# ============================================================================
# Edge cases
# ============================================================================


def test_single_mode_system():
    """Single mode system should work."""
    rng = np.random.default_rng(161718)
    n_modes = 1
    n_modals = 3
    n_exc = n_modals - 1

    t1 = rng.normal(scale=0.1, size=(n_modes, n_exc))
    t2 = rng.normal(scale=0.05, size=(n_modes, n_modes, n_exc, n_exc))
    amps = VCCAmplitudes(t1=t1, t2=t2, energy=-5.0, converged=True, iterations=10)

    # UVCCSD should work
    x0_uvccsd = map_vccsd_to_uvccsd_initial_point(
        amps, n_modes=n_modes, n_modals=n_modals
    )
    assert len(x0_uvccsd) == n_exc  # Only singles, no doubles

    # VLUCJ should work (no Jastrow terms for single mode)
    x0_vlucj = map_vccsd_to_vlucj_params(
        amps, n_modes=n_modes, n_modals=n_modals, layers=2, include_rz=False
    )
    expected_vlucj = 2 * 2 * n_exc  # 2 layers * 2 singles blocks * n_exc
    assert len(x0_vlucj) == expected_vlucj

    # CHC should work (no pair correlations for single mode)
    x0_chc = map_vccsd_to_chc_params(amps, n_modes=n_modes, n_modals=n_modals, layers=2)
    expected_chc = 2 * 2 * n_exc  # 2 layers * 2 singles blocks * n_exc
    assert len(x0_chc) == expected_chc


def test_two_modal_system():
    """Two modal system (minimal) should work."""
    rng = np.random.default_rng(192021)
    n_modes = 2
    n_modals = 2
    n_exc = n_modals - 1

    t1 = rng.normal(scale=0.1, size=(n_modes, n_exc))
    t2 = rng.normal(scale=0.05, size=(n_modes, n_modes, n_exc, n_exc))
    amps = VCCAmplitudes(t1=t1, t2=t2, energy=-5.0, converged=True, iterations=10)

    # All mappings should work
    x0_uvccsd = map_vccsd_to_uvccsd_initial_point(
        amps, n_modes=n_modes, n_modals=n_modals
    )
    assert len(x0_uvccsd) > 0

    x0_vlucj = map_vccsd_to_vlucj_params(
        amps, n_modes=n_modes, n_modals=n_modals, layers=1, include_rz=False
    )
    assert len(x0_vlucj) > 0

    x0_chc = map_vccsd_to_chc_params(amps, n_modes=n_modes, n_modals=n_modals, layers=1)
    assert len(x0_chc) > 0
