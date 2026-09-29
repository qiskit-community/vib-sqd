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
Parameter I/O utilities for saving and loading optimized VQE parameters.

Allows reusing optimized parameters across runs without re-optimization.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit


def save_optimized_parameters(
    params: np.ndarray,
    filepath: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save optimized VQE parameters to JSON file.

    Parameters
    ----------
    params : np.ndarray
        Optimized parameter values
    filepath : str
        Path to save JSON file
    metadata : dict, optional
        Additional metadata (energy, config, etc.)

    Example
    -------
    >>> save_optimized_parameters(
    ...     params=best_params,
    ...     filepath="outputs/vlucj_params.json",
    ...     metadata={
    ...         "energy": -55.0,
    ...         "n_modes": 3,
    ...         "n_modals": 4,
    ...         "n_layers": 6,
    ...         "gap_closed_pct": 98.5,
    ...     }
    ... )
    """
    # Create output directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Prepare data
    data = {
        "parameters": params.tolist(),
        "n_parameters": len(params),
        "metadata": metadata or {},
    }

    # Save to JSON
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved {len(params)} parameters to {filepath}")


def load_optimized_parameters(
    filepath: str,
    verify_count: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load optimized VQE parameters from JSON file.

    Parameters
    ----------
    filepath : str
        Path to JSON file
    verify_count : int, optional
        Expected number of parameters (raises error if mismatch)

    Returns
    -------
    params : np.ndarray
        Parameter values
    metadata : dict
        Metadata dictionary

    Example
    -------
    >>> params, meta = load_optimized_parameters(
    ...     filepath="outputs/vlucj_params.json",
    ...     verify_count=152,
    ... )
    >>> print(f"Loaded {len(params)} params, energy={meta['energy']}")
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    params = np.array(data["parameters"])
    metadata = data.get("metadata", {})

    # Verify parameter count if requested
    if verify_count is not None and len(params) != verify_count:
        raise ValueError(
            f"Parameter count mismatch: expected {verify_count}, "
            f"got {len(params)} from {filepath}"
        )

    print(f"✓ Loaded {len(params)} parameters from {filepath}")
    if "energy" in metadata:
        print(f"  Saved energy: {metadata['energy']:.2f} cm⁻¹")

    return params, metadata


def bind_parameters_to_circuit(
    circuit: QuantumCircuit,
    parameters,  # ParameterVector or List[Parameter]
    values: np.ndarray,
) -> QuantumCircuit:
    """
    Bind parameter values to a parameterized circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized circuit
    parameters : ParameterVector or List[Parameter]
        Parameter vector or list used in circuit
    values : np.ndarray
        Parameter values to bind

    Returns
    -------
    bound_circuit : QuantumCircuit
        Circuit with parameters bound to values

    Example
    -------
    >>> # Build parameterized circuit
    >>> circuit_param, params = vlucj.build_parameterized(...)
    >>>
    >>> # Load optimized values
    >>> opt_values, _ = load_optimized_parameters("params.json")
    >>>
    >>> # Bind to get executable circuit
    >>> circuit_bound = bind_parameters_to_circuit(
    ...     circuit_param, params, opt_values
    ... )
    """
    if len(values) != len(parameters):
        raise ValueError(
            f"Value count mismatch: {len(values)} values for "
            f"{len(parameters)} parameters"
        )

    # Create binding dictionary
    param_dict = {param: float(val) for param, val in zip(parameters, values)}

    # Bind and return
    return circuit.assign_parameters(param_dict)


def create_checkpoint(
    params: np.ndarray,
    energy: float,
    iteration: int,
    filepath: str,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Create a checkpoint during optimization.

    Useful for long-running optimizations to save intermediate results.

    Parameters
    ----------
    params : np.ndarray
        Current parameter values
    energy : float
        Current energy
    iteration : int
        Current iteration number
    filepath : str
        Path to checkpoint file
    config : dict, optional
        Configuration dictionary

    Example
    -------
    >>> # In VQE callback
    >>> if iteration % 100 == 0:
    ...     create_checkpoint(
    ...         params=current_params,
    ...         energy=current_energy,
    ...         iteration=iteration,
    ...         filepath=f"checkpoints/iter_{iteration}.json",
    ...         config={"n_layers": 6, "method": "COBYLA"},
    ...     )
    """
    metadata = {
        "energy": float(energy),
        "iteration": int(iteration),
        "config": config or {},
    }

    save_optimized_parameters(params, filepath, metadata)


def load_latest_checkpoint(
    checkpoint_dir: str,
    pattern: str = "iter_*.json",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load the latest checkpoint from a directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints
    pattern : str
        Glob pattern for checkpoint files

    Returns
    -------
    params : np.ndarray
        Parameter values from latest checkpoint
    metadata : dict
        Metadata including iteration and energy

    Example
    -------
    >>> params, meta = load_latest_checkpoint("checkpoints/")
    >>> print(f"Resuming from iteration {meta['iteration']}")
    """
    checkpoint_path = Path(checkpoint_dir)

    # Find all checkpoint files
    checkpoints = list(checkpoint_path.glob(pattern))

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir} matching {pattern}"
        )

    # Sort by modification time (most recent first)
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    print(f"Loading latest checkpoint: {latest.name}")
    return load_optimized_parameters(str(latest))
