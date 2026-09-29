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
Map classical VCC/VCCSD amplitudes to quantum circuit parameters.

This module provides functions to map classical vibrational coupled cluster
amplitudes to initial parameters for quantum ansätze (UVCCSD, VLUCJ, CHC).

The mapping strategies allow different scaling approaches to handle the
difference between classical amplitude magnitudes and quantum circuit parameters.

Amplitude Convention
--------------------
Input amplitudes follow the convention from vcc_solver.py:
    t1[i, a]: Mode i, reference modal 0 → excited modal a+1
    t2[i, j, a, b]: Modes i,j, both 0 → a+1, b+1

Examples
--------
Map to UVCCSD initial point:
>>> from vib_sqd.classical.amplitude_io import load_vcc_amplitudes_npz
>>> amps = load_vcc_amplitudes_npz("h2o_vccsd.npz")
>>> x0 = map_vccsd_to_uvccsd_initial_point(amps, n_modes=3, n_modals=4)
>>> # Use x0 as initial_point in VQE

Map to VLUCJ parameters:
>>> x0_vlucj = map_vccsd_to_vlucj_params(amps, n_modes=3, n_modals=4, layers=4)

Map to CHC parameters:
>>> x0_chc = map_vccsd_to_chc_params(amps, n_modes=3, n_modals=4, layers=4)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from vib_sqd.classical.vcc_solver import VCCAmplitudes

ScalingStrategy = Literal["direct", "sqrt", "tanh", "normalized", "none"]


def _validate_system_size(n_modes: int, n_modals: int) -> None:
    """
    Validate system size parameters.

    Parameters
    ----------
    n_modes : int
        Number of vibrational modes.
    n_modals : int
        Number of modals per mode.

    Raises
    ------
    ValueError
        If n_modes <= 0 or n_modals < 2.
    """
    if n_modes <= 0:
        raise ValueError(f"n_modes must be positive, got {n_modes}")
    if n_modals < 2:
        raise ValueError(f"n_modals must be at least 2, got {n_modals}")


def _scale_amplitudes(
    arr: np.ndarray,
    strategy: ScalingStrategy = "direct",
    clip: float | None = None,
) -> np.ndarray:
    """
    Scale VCC amplitudes before using them as circuit parameters.

    Different scaling strategies can be used to handle the difference
    between classical amplitude magnitudes and quantum circuit parameters.

    Parameters
    ----------
    arr : np.ndarray
        Input amplitudes.
    strategy : ScalingStrategy, default="direct"
        Scaling strategy to use:
        - "direct": Use amplitudes as-is (most physical)
        - "sqrt": sign(x) * sqrt(abs(x)) - enhance small amplitudes
        - "tanh": tanh(x) - bound large amplitudes
        - "normalized": Divide by max abs value (useful for diagnostics, not physically direct)
        - "none": Return zeros
    clip : float or None, default=None
        If provided, clip output to [-clip, clip]. Must be non-negative.

    Returns
    -------
    np.ndarray
        Scaled amplitudes.

    Raises
    ------
    ValueError
        If amplitudes contain non-finite values or clip is negative.

    Examples
    --------
    >>> scaled = _scale_amplitudes(t1, strategy="tanh", clip=0.5)
    """
    arr = np.asarray(arr, dtype=float)

    # Validate finite values
    if not np.all(np.isfinite(arr)):
        raise ValueError("Cannot scale amplitudes containing non-finite values")

    # Validate clip
    if clip is not None and clip < 0:
        raise ValueError(f"clip must be non-negative or None, got {clip}")

    if strategy == "direct":
        out = arr.copy()
    elif strategy == "sqrt":
        out = np.sign(arr) * np.sqrt(np.abs(arr))
    elif strategy == "tanh":
        out = np.tanh(arr)
    elif strategy == "normalized":
        max_abs = np.max(np.abs(arr))
        out = arr / max_abs if max_abs > 0 else arr.copy()
    elif strategy == "none":
        out = np.zeros_like(arr)
    else:
        raise ValueError(f"Unknown scaling strategy: {strategy}")

    if clip is not None:
        out = np.clip(out, -clip, clip)

    return out


def map_vccsd_to_uvccsd_params(
    vcc_amplitudes: VCCAmplitudes,
    scaling_strategy: ScalingStrategy = "direct",
    clip: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Map VCCSD amplitudes to UVCCSD ansatz parameters (for fixed mode).

    This returns a dictionary suitable for cfg["ansatz_params"] when
    using UVCCSDAnsatz in non-parameterized mode (for state preparation
    or initial energy evaluation).

    For VQE optimization, use map_vccsd_to_uvccsd_initial_point() instead.

    Parameters
    ----------
    vcc_amplitudes : VCCAmplitudes
        Classical VCC/VCCSD amplitudes.
    scaling_strategy : ScalingStrategy, default="direct"
        Scaling strategy to apply.
    clip : float or None, default=None
        Optional clipping threshold.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys "t1" and "t2" ready for cfg["ansatz_params"].

    Examples
    --------
    >>> params = map_vccsd_to_uvccsd_params(amps, scaling_strategy="direct")
    >>> cfg = {"parameterized": False, "ansatz_params": params, ...}
    """
    vcc_amplitudes.validate()

    t1 = _scale_amplitudes(vcc_amplitudes.t1, scaling_strategy, clip)
    t2 = _scale_amplitudes(vcc_amplitudes.t2, scaling_strategy, clip)

    return {
        "t1": t1,
        "t2": t2,
    }


def map_vccsd_to_uvccsd_initial_point(
    vcc_amplitudes: VCCAmplitudes,
    n_modes: int,
    n_modals: int,
    scaling_strategy: ScalingStrategy = "direct",
    clip: float | None = None,
) -> np.ndarray:
    """
    Map VCCSD amplitudes to UVCCSD initial point for VQE.

    This flattens T1 and T2 amplitudes into a 1D parameter vector
    matching the order used by UVCCSDAnsatz(parameterized=True).

    Parameter order:
        1. Singles: for each mode, for each excited modal
        2. Doubles: for each mode pair (i < j), for each excited modal pair

    Parameters
    ----------
    vcc_amplitudes : VCCAmplitudes
        Classical VCC/VCCSD amplitudes.
    n_modes : int
        Number of vibrational modes.
    n_modals : int
        Number of modals per mode.
    scaling_strategy : ScalingStrategy, default="direct"
        Scaling strategy to apply.
    clip : float or None, default=None
        Optional clipping threshold.

    Returns
    -------
    np.ndarray
        Flat parameter vector for VQE initial_point.

    Raises
    ------
    ValueError
        If dimensions don't match amplitude shapes or system size is invalid.

    Examples
    --------
    >>> x0 = map_vccsd_to_uvccsd_initial_point(amps, n_modes=3, n_modals=4)
    >>> result = run_vqe(ansatz=uvccsd_circuit, initial_point=x0)
    """
    _validate_system_size(n_modes, n_modals)
    vcc_amplitudes.validate()

    if n_modes != vcc_amplitudes.n_modes:
        raise ValueError(
            f"n_modes mismatch: got {n_modes}, "
            f"amplitudes have {vcc_amplitudes.n_modes}"
        )

    if n_modals != vcc_amplitudes.n_modals:
        raise ValueError(
            f"n_modals mismatch: got {n_modals}, "
            f"amplitudes have {vcc_amplitudes.n_modals}"
        )

    t1 = _scale_amplitudes(vcc_amplitudes.t1, scaling_strategy, clip)
    t2 = _scale_amplitudes(vcc_amplitudes.t2, scaling_strategy, clip)

    n_exc = n_modals - 1
    n_singles = n_modes * n_exc
    n_doubles = (n_modes * (n_modes - 1) // 2) * n_exc * n_exc

    x0 = np.zeros(n_singles + n_doubles, dtype=float)

    # Singles block
    p = 0
    for mode in range(n_modes):
        for exc_idx in range(n_exc):
            x0[p] = t1[mode, exc_idx]
            p += 1

    # Doubles block
    for mode1 in range(n_modes):
        for mode2 in range(mode1 + 1, n_modes):
            for exc1_idx in range(n_exc):
                for exc2_idx in range(n_exc):
                    x0[p] = t2[mode1, mode2, exc1_idx, exc2_idx]
                    p += 1

    # Sanity check
    if p != len(x0):
        raise RuntimeError(
            f"UVCCSD parameter fill mismatch: filled {p}, expected {len(x0)}"
        )

    return x0


def vlucj_num_params_per_layer(
    n_modes: int,
    n_modals: int,
    include_rz: bool = True,
) -> int:
    """
    Return expected number of VLUCJ parameters per layer.

    Assumed order per layer:
        1. First intramode Givens: n_modes * (n_modals - 1)
        2. Adjacent-mode Jastrow: (n_modes - 1) * n_modals
        3. Second intramode Givens: n_modes * (n_modals - 1)
        4. Optional RZ: n_modes * n_modals

    Parameters
    ----------
    n_modes : int
        Number of vibrational modes.
    n_modals : int
        Number of modals per mode.
    include_rz : bool, default=True
        Whether to include RZ phase block.

    Returns
    -------
    int
        Number of parameters per layer.

    Raises
    ------
    ValueError
        If system size is invalid.

    Examples
    --------
    >>> n_params = vlucj_num_params_per_layer(n_modes=3, n_modals=4, include_rz=True)
    >>> print(f"VLUCJ with 4 layers needs {4 * n_params} parameters")
    """
    _validate_system_size(n_modes, n_modals)

    n_singles = n_modes * (n_modals - 1)
    n_jastrow = max(n_modes - 1, 0) * n_modals
    n_rz = n_modes * n_modals if include_rz else 0

    return 2 * n_singles + n_jastrow + n_rz


def map_vccsd_to_vlucj_params(
    vcc_amplitudes: VCCAmplitudes,
    n_modes: int,
    n_modals: int,
    layers: int,
    include_rz: bool = True,
    scaling_strategy: ScalingStrategy = "direct",
    clip: float | None = np.pi,
    distribute_over_layers: bool = True,
) -> np.ndarray:
    """
    Map VCCSD amplitudes to VLUCJ initial parameters.

    Mapping strategy:
        T1 → intramode Givens rotations (split 0.5x across both blocks)
        T2 → adjacent-mode Jastrow phases (diagonal terms only)
        RZ → zero by default (no VCC equivalent)

    Important:
        VLUCJ only has adjacent-mode Jastrow terms in this implementation.
        From full T2, we use adjacent mode correlations: t2[i, i+1, a, a]

    Parameters
    ----------
    vcc_amplitudes : VCCAmplitudes
        Classical VCC/VCCSD amplitudes.
    n_modes : int
        Number of vibrational modes.
    n_modals : int
        Number of modals per mode.
    layers : int
        Number of VLUCJ layers (must be positive).
    include_rz : bool, default=True
        Whether to include RZ phase block.
    scaling_strategy : ScalingStrategy, default="direct"
        Scaling strategy to apply.
    clip : float or None, default=np.pi
        Optional clipping threshold.
    distribute_over_layers : bool, default=True
        If True, divide amplitudes equally across layers.

    Returns
    -------
    np.ndarray
        Flat VLUCJ parameter vector for VQE initial_point.

    Raises
    ------
    ValueError
        If dimensions don't match amplitude shapes, system size is invalid,
        or layers <= 0.

    Examples
    --------
    >>> x0 = map_vccsd_to_vlucj_params(amps, n_modes=3, n_modals=4, layers=4)
    >>> result = run_vqe(ansatz=vlucj_circuit, initial_point=x0)
    """
    _validate_system_size(n_modes, n_modals)

    if layers <= 0:
        raise ValueError(f"layers must be positive, got {layers}")

    vcc_amplitudes.validate()

    if n_modes != vcc_amplitudes.n_modes:
        raise ValueError(
            f"n_modes mismatch: got {n_modes}, "
            f"amplitudes have {vcc_amplitudes.n_modes}"
        )

    if n_modals != vcc_amplitudes.n_modals:
        raise ValueError(
            f"n_modals mismatch: got {n_modals}, "
            f"amplitudes have {vcc_amplitudes.n_modals}"
        )

    t1 = _scale_amplitudes(vcc_amplitudes.t1, scaling_strategy, clip)
    t2 = _scale_amplitudes(vcc_amplitudes.t2, scaling_strategy, clip)

    n_singles = n_modes * (n_modals - 1)
    n_jastrow = max(n_modes - 1, 0) * n_modals
    n_rz = n_modes * n_modals if include_rz else 0
    params_per_layer = 2 * n_singles + n_jastrow + n_rz

    x0 = np.zeros(layers * params_per_layer, dtype=float)

    layer_scale = 1.0 / layers if distribute_over_layers else 1.0
    # Split T1 across both singles blocks (0.5x each)
    singles_block_scale = 0.5 * layer_scale
    # Full T2 for correlation block
    correlation_block_scale = layer_scale

    for layer in range(layers):
        base = layer * params_per_layer

        # 1. First intramode block (0.5x T1)
        p = base
        for mode in range(n_modes):
            for exc in range(n_modals - 1):
                x0[p] = singles_block_scale * t1[mode, exc]
                p += 1

        # 2. Adjacent-mode Jastrow block
        p = base + n_singles
        for mode in range(n_modes - 1):
            next_mode = mode + 1
            for modal in range(n_modals):
                if modal == 0:
                    # Reference modal phase; keep zero
                    x0[p] = 0.0
                else:
                    exc = modal - 1
                    x0[p] = correlation_block_scale * t2[mode, next_mode, exc, exc]
                p += 1

        # 3. Second intramode block (0.5x T1)
        p = base + n_singles + n_jastrow
        for mode in range(n_modes):
            for exc in range(n_modals - 1):
                x0[p] = singles_block_scale * t1[mode, exc]
                p += 1

        # 4. RZ block remains zero (no VCC equivalent)
        # Sanity check: we filled up to the RZ block
        if p != base + 2 * n_singles + n_jastrow:
            raise RuntimeError("VLUCJ non-RZ block fill mismatch")

    # Final shape check
    if x0.shape != (layers * params_per_layer,):
        raise RuntimeError("VLUCJ initial point shape mismatch")

    return x0


def chc_num_params_per_layer(
    n_modes: int,
    n_modals: int,
) -> int:
    """
    Return expected number of CHC parameters per layer.

    Assumed order per layer:
        1. First singles: n_modes * (n_modals - 1)
        2. Pair correlation: n_pairs * (n_modals - 1)^2
        3. Second singles: n_modes * (n_modals - 1)

    Parameters
    ----------
    n_modes : int
        Number of vibrational modes.
    n_modals : int
        Number of modals per mode.

    Returns
    -------
    int
        Number of parameters per layer.

    Raises
    ------
    ValueError
        If system size is invalid.

    Examples
    --------
    >>> n_params = chc_num_params_per_layer(n_modes=3, n_modals=4)
    >>> print(f"CHC with 4 layers needs {4 * n_params} parameters")
    """
    _validate_system_size(n_modes, n_modals)

    n_singles = n_modes * (n_modals - 1)
    n_pairs = n_modes * (n_modes - 1) // 2
    n_pair_corr = n_pairs * (n_modals - 1) ** 2

    return 2 * n_singles + n_pair_corr


def map_vccsd_to_chc_params(
    vcc_amplitudes: VCCAmplitudes,
    n_modes: int,
    n_modals: int,
    layers: int,
    scaling_strategy: ScalingStrategy = "direct",
    clip: float | None = np.pi,
    distribute_over_layers: bool = True,
) -> np.ndarray:
    """
    Map VCCSD amplitudes to CHC initial parameters.

    Mapping strategy:
        T1 → first and second singles blocks (split 0.5x across both)
        T2 → pair correlation block (all mode pairs)

    This is more expressive than VLUCJ if CHC includes all mode pairs.

    Parameters
    ----------
    vcc_amplitudes : VCCAmplitudes
        Classical VCC/VCCSD amplitudes.
    n_modes : int
        Number of vibrational modes.
    n_modals : int
        Number of modals per mode.
    layers : int
        Number of CHC layers (must be positive).
    scaling_strategy : ScalingStrategy, default="direct"
        Scaling strategy to apply.
    clip : float or None, default=np.pi
        Optional clipping threshold.
    distribute_over_layers : bool, default=True
        If True, divide amplitudes equally across layers.

    Returns
    -------
    np.ndarray
        Flat CHC parameter vector for VQE initial_point.

    Raises
    ------
    ValueError
        If dimensions don't match amplitude shapes, system size is invalid,
        or layers <= 0.

    Examples
    --------
    >>> x0 = map_vccsd_to_chc_params(amps, n_modes=3, n_modals=4, layers=4)
    >>> result = run_vqe(ansatz=chc_circuit, initial_point=x0)
    """
    _validate_system_size(n_modes, n_modals)

    if layers <= 0:
        raise ValueError(f"layers must be positive, got {layers}")

    vcc_amplitudes.validate()

    if n_modes != vcc_amplitudes.n_modes:
        raise ValueError(
            f"n_modes mismatch: got {n_modes}, "
            f"amplitudes have {vcc_amplitudes.n_modes}"
        )

    if n_modals != vcc_amplitudes.n_modals:
        raise ValueError(
            f"n_modals mismatch: got {n_modals}, "
            f"amplitudes have {vcc_amplitudes.n_modals}"
        )

    t1 = _scale_amplitudes(vcc_amplitudes.t1, scaling_strategy, clip)
    t2 = _scale_amplitudes(vcc_amplitudes.t2, scaling_strategy, clip)

    n_singles = n_modes * (n_modals - 1)
    n_pairs = n_modes * (n_modes - 1) // 2
    n_pair_corr = n_pairs * (n_modals - 1) ** 2
    params_per_layer = 2 * n_singles + n_pair_corr

    x0 = np.zeros(layers * params_per_layer, dtype=float)

    layer_scale = 1.0 / layers if distribute_over_layers else 1.0
    # Split T1 across both singles blocks (0.5x each)
    singles_block_scale = 0.5 * layer_scale
    # Full T2 for correlation block
    correlation_block_scale = layer_scale

    for layer in range(layers):
        base = layer * params_per_layer

        # 1. First singles block (0.5x T1)
        p = base
        for mode in range(n_modes):
            for exc in range(n_modals - 1):
                x0[p] = singles_block_scale * t1[mode, exc]
                p += 1

        # 2. Pair-correlation block
        p = base + n_singles
        for mode_i in range(n_modes):
            for mode_j in range(mode_i + 1, n_modes):
                for exc_i in range(n_modals - 1):
                    for exc_j in range(n_modals - 1):
                        x0[p] = (
                            correlation_block_scale * t2[mode_i, mode_j, exc_i, exc_j]
                        )
                        p += 1

        # 3. Second singles block (0.5x T1)
        p = base + n_singles + n_pair_corr
        for mode in range(n_modes):
            for exc in range(n_modals - 1):
                x0[p] = singles_block_scale * t1[mode, exc]
                p += 1

        # Sanity check for this layer
        if p != base + params_per_layer:
            raise RuntimeError(
                f"CHC parameter fill mismatch in layer {layer}: "
                f"filled until {p}, expected {base + params_per_layer}"
            )

    return x0
