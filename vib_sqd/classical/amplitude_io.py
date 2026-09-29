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
Load and save VCC/VCCSD amplitudes in standardized NPZ format.

This module provides functions to save and load VCCAmplitudes objects
using NumPy's compressed NPZ format. This allows manual export from
MidasCpp or other VCC codes and reuse across experiments.

NPZ Format Specification
------------------------
Required keys:
    t1 : np.ndarray, shape (n_modes, n_modals-1)
    t2 : np.ndarray, shape (n_modes, n_modes, n_modals-1, n_modals-1)
    format_version : str, currently "1.0"

Optional keys:
    energy : float or np.nan
    converged : bool
    iterations : int or -1 (indicates not available)
    method : str (e.g., "vccsd", "vci")
    molecule : str (e.g., "h2o", "co2")
    n_modes : int
    n_modals : int

Examples
--------
Save amplitudes:
>>> from vib_sqd.classical.vcc_solver import VCCAmplitudes
>>> amps = VCCAmplitudes(t1=t1, t2=t2, energy=-76.0, converged=True)
>>> save_vcc_amplitudes_npz(amps, "h2o_vccsd_3m_4modal.npz")

Load amplitudes:
>>> amps = load_vcc_amplitudes_npz("h2o_vccsd_3m_4modal.npz")
>>> print(f"Loaded {amps.n_modes} modes, {amps.n_modals} modals")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vib_sqd.classical.vcc_solver import VCCAmplitudes

# Current NPZ format version for future-proofing
_FORMAT_VERSION = "1.0"


def _extract_scalar_string(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    """
    Safely extract a scalar string from NPZ data.

    Parameters
    ----------
    data : NpzFile
        Loaded NPZ file.
    key : str
        Key to extract.

    Returns
    -------
    str or None
        Extracted string value, or None if key not present.
    """
    if key not in data.files:
        return None

    value = data[key]
    if value.ndim == 0:
        # Scalar array
        return str(value.item())
    else:
        # Non-scalar, convert first element
        return str(value.flat[0])


def _extract_scalar_float(
    data: np.lib.npyio.NpzFile, key: str, default: float = np.nan
) -> float:
    """
    Safely extract a scalar float from NPZ data.

    Parameters
    ----------
    data : NpzFile
        Loaded NPZ file.
    key : str
        Key to extract.
    default : float
        Default value if key not present.

    Returns
    -------
    float
        Extracted float value.
    """
    if key not in data.files:
        return default

    value = data[key]
    if value.ndim == 0:
        return float(value.item())
    else:
        return float(value.flat[0])


def _extract_scalar_int(data: np.lib.npyio.NpzFile, key: str, default: int = -1) -> int:
    """
    Safely extract a scalar int from NPZ data.

    Parameters
    ----------
    data : NpzFile
        Loaded NPZ file.
    key : str
        Key to extract.
    default : int
        Default value if key not present.

    Returns
    -------
    int
        Extracted int value.
    """
    if key not in data.files:
        return default

    value = data[key]
    if value.ndim == 0:
        return int(value.item())
    else:
        return int(value.flat[0])


def _extract_scalar_bool(
    data: np.lib.npyio.NpzFile, key: str, default: bool = False
) -> bool:
    """
    Safely extract a scalar bool from NPZ data.

    Parameters
    ----------
    data : NpzFile
        Loaded NPZ file.
    key : str
        Key to extract.
    default : bool
        Default value if key not present.

    Returns
    -------
    bool
        Extracted bool value.
    """
    if key not in data.files:
        return default

    value = data[key]
    if value.ndim == 0:
        return bool(value.item())
    else:
        return bool(value.flat[0])


def save_vcc_amplitudes_npz(
    amplitudes: VCCAmplitudes,
    path: str | Path,
    enforce_suffix: bool = True,
) -> None:
    """
    Save VCC amplitudes in a standardized NPZ format.

    This function validates the amplitudes before saving and creates
    the parent directory if it doesn't exist.

    Parameters
    ----------
    amplitudes : VCCAmplitudes
        Amplitudes to save.
    path : str or Path
        Output file path (should end with .npz).
    enforce_suffix : bool, default=True
        If True, ensure path ends with .npz suffix.

    Raises
    ------
    ValueError
        If amplitude validation fails or path doesn't end with .npz
        (when enforce_suffix=True).

    Examples
    --------
    >>> amps = VCCAmplitudes(t1=t1, t2=t2, energy=-76.0, converged=True)
    >>> save_vcc_amplitudes_npz(amps, "data/h2o_vccsd.npz")
    """
    # Validate before saving
    amplitudes.validate()

    path = Path(path)

    # Enforce .npz suffix if requested
    if enforce_suffix and path.suffix != ".npz":
        raise ValueError(
            f"Path must end with .npz suffix, got: {path}. "
            f"Set enforce_suffix=False to override."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data dictionary
    data: dict[str, Any] = {
        "format_version": np.array(_FORMAT_VERSION, dtype=str),
        "t1": amplitudes.t1,
        "t2": amplitudes.t2,
        "energy": np.array(
            amplitudes.energy if amplitudes.energy is not None else np.nan
        ),
        "converged": np.array(amplitudes.converged),
        "iterations": np.array(
            amplitudes.iterations if amplitudes.iterations is not None else -1
        ),
    }

    # Add optional metadata if present
    if "method" in amplitudes.metadata:
        data["method"] = np.array(amplitudes.metadata["method"], dtype=str)

    if "molecule" in amplitudes.metadata:
        data["molecule"] = np.array(amplitudes.metadata["molecule"], dtype=str)

    # Add system size for convenience
    data["n_modes"] = np.array(amplitudes.n_modes)
    data["n_modals"] = np.array(amplitudes.n_modals)

    # Save as compressed NPZ
    np.savez_compressed(path, **data)


def load_vcc_amplitudes_npz(
    path: str | Path,
    enforce_suffix: bool = True,
) -> VCCAmplitudes:
    """
    Load VCC amplitudes from NPZ format.

    This function loads amplitudes and validates them before returning.

    Parameters
    ----------
    path : str or Path
        Input file path.
    enforce_suffix : bool, default=True
        If True, ensure path ends with .npz suffix.

    Returns
    -------
    VCCAmplitudes
        Loaded and validated amplitudes.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    KeyError
        If required keys (t1, t2) are missing.
    ValueError
        If amplitude validation fails or path doesn't end with .npz
        (when enforce_suffix=True).

    Examples
    --------
    >>> amps = load_vcc_amplitudes_npz("data/h2o_vccsd.npz")
    >>> print(f"Energy: {amps.energy:.6f} Ha")
    >>> print(f"Converged: {amps.converged}")
    """
    path = Path(path)

    # Enforce .npz suffix if requested
    if enforce_suffix and path.suffix != ".npz":
        raise ValueError(
            f"Path must end with .npz suffix, got: {path}. "
            f"Set enforce_suffix=False to override."
        )

    if not path.exists():
        raise FileNotFoundError(f"Amplitude file not found: {path}")

    # Load NPZ file (no pickle for security)
    with np.load(path, allow_pickle=False) as data:
        # Check format version if present
        format_version = _extract_scalar_string(data, "format_version")
        if format_version is not None and format_version != _FORMAT_VERSION:
            # For now, just warn; in future could handle version migration
            import warnings

            warnings.warn(
                f"NPZ format version mismatch: file has {format_version}, "
                f"expected {_FORMAT_VERSION}. Attempting to load anyway.",
                UserWarning,
            )

        # Required keys
        if "t1" not in data.files:
            raise KeyError(f"Required key 't1' not found in {path}")
        if "t2" not in data.files:
            raise KeyError(f"Required key 't2' not found in {path}")

        t1 = np.asarray(data["t1"], dtype=float)
        t2 = np.asarray(data["t2"], dtype=float)

        # Optional keys with safe extraction
        energy_raw = _extract_scalar_float(data, "energy", default=np.nan)
        energy = None if np.isnan(energy_raw) else energy_raw

        converged = _extract_scalar_bool(data, "converged", default=False)

        iterations_raw = _extract_scalar_int(data, "iterations", default=-1)
        iterations = None if iterations_raw < 0 else iterations_raw

        # Build metadata dictionary
        metadata: dict[str, Any] = {"source": str(path)}

        method = _extract_scalar_string(data, "method")
        if method is not None:
            metadata["method"] = method

        molecule = _extract_scalar_string(data, "molecule")
        if molecule is not None:
            metadata["molecule"] = molecule

        n_modes_stored = _extract_scalar_int(data, "n_modes", default=-1)
        if n_modes_stored >= 0:
            metadata["n_modes_stored"] = n_modes_stored

        n_modals_stored = _extract_scalar_int(data, "n_modals", default=-1)
        if n_modals_stored >= 0:
            metadata["n_modals_stored"] = n_modals_stored

    # Create VCCAmplitudes object
    amplitudes = VCCAmplitudes(
        t1=t1,
        t2=t2,
        energy=energy,
        converged=converged,
        iterations=iterations,
        metadata=metadata,
    )

    # Validate before returning
    amplitudes.validate()

    # Cross-check stored dimensions if available
    if "n_modes_stored" in metadata:
        if amplitudes.n_modes != metadata["n_modes_stored"]:
            raise ValueError(
                f"Dimension mismatch: stored n_modes={metadata['n_modes_stored']}, "
                f"but t1 shape implies n_modes={amplitudes.n_modes}"
            )

    if "n_modals_stored" in metadata:
        if amplitudes.n_modals != metadata["n_modals_stored"]:
            raise ValueError(
                f"Dimension mismatch: stored n_modals={metadata['n_modals_stored']}, "
                f"but t1 shape implies n_modals={amplitudes.n_modals}"
            )

    return amplitudes
