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
VCC/VCCSD amplitude container and abstract solver interface.

This module provides a standard container for classical vibrational coupled cluster
amplitudes and an abstract interface for VCC solver backends (MidasCpp, PySCF, etc.).

Amplitude Convention
--------------------
t1[i, a]: Excitation in mode i from reference modal 0 to modal a+1
    - i: mode index (0 to n_modes-1)
    - a: excited modal index (0 to n_modals-2)
    - Represents: |0⟩_i → |a+1⟩_i

t2[i, j, a, b]: Double excitation in modes i and j
    - i, j: mode indices (0 to n_modes-1)
    - a, b: excited modal indices (0 to n_modals-2)
    - Represents: |0⟩_i|0⟩_j → |a+1⟩_i|b+1⟩_j

T2 Symmetry (Optional)
----------------------
For symmetric VCC implementations:
    t2[i, j, a, b] should equal t2[j, i, b, a]

This symmetry is not enforced by default but can be validated using
the validate_t2_symmetry() method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VCCAmplitudes:
    """
    Standard container for classical VCC/VCCSD amplitudes.

    This dataclass provides a consistent interface for storing and validating
    vibrational coupled cluster amplitudes from any backend (MidasCpp, PySCF, etc.).

    Attributes
    ----------
    t1 : np.ndarray
        Single excitation amplitudes, shape (n_modes, n_modals - 1)
    t2 : np.ndarray
        Double excitation amplitudes, shape (n_modes, n_modes, n_modals - 1, n_modals - 1)
    energy : float | None
        Ground state energy (if available)
    converged : bool
        Whether the VCC calculation converged
    iterations : int | None
        Number of iterations to convergence (if available)
    metadata : dict[str, Any]
        Additional metadata (method, molecule, source, etc.)

    Examples
    --------
    >>> import numpy as np
    >>> t1 = np.random.randn(3, 3)  # 3 modes, 4 modals
    >>> t2 = np.random.randn(3, 3, 3, 3)
    >>> amps = VCCAmplitudes(t1=t1, t2=t2, energy=-76.0, converged=True)
    >>> amps.validate()
    >>> print(f"System: {amps.n_modes} modes, {amps.n_modals} modals")
    """

    t1: np.ndarray
    t2: np.ndarray
    energy: float | None = None
    converged: bool = False
    iterations: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce t1 and t2 to float arrays after initialization."""
        self.t1 = np.asarray(self.t1, dtype=float)
        self.t2 = np.asarray(self.t2, dtype=float)

    @property
    def n_modes(self) -> int:
        """Number of vibrational modes."""
        return int(self.t1.shape[0])

    @property
    def n_modals(self) -> int:
        """Number of modals (levels) per mode."""
        return int(self.t1.shape[1] + 1)

    def validate(self) -> None:
        """
        Validate amplitude shapes and values.

        Raises
        ------
        ValueError
            If amplitudes have incorrect shapes or contain non-finite values.
        """
        # Validate t1 shape
        if self.t1.ndim != 2:
            raise ValueError(f"t1 must be rank-2, got shape {self.t1.shape}")

        # Validate t2 shape
        if self.t2.ndim != 4:
            raise ValueError(f"t2 must be rank-4, got shape {self.t2.shape}")

        n_modes, n_exc = self.t1.shape
        expected_t2_shape = (n_modes, n_modes, n_exc, n_exc)

        if self.t2.shape != expected_t2_shape:
            raise ValueError(
                f"t2 shape mismatch. Expected {expected_t2_shape}, "
                f"got {self.t2.shape}"
            )

        # Validate positive n_modes
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}")

        # Validate at least one excited modal
        if n_exc <= 0:
            raise ValueError(
                f"Must have at least one excited modal (n_modals >= 2), "
                f"got n_exc={n_exc} (n_modals={n_exc + 1})"
            )

        # Validate finite amplitudes
        if not np.all(np.isfinite(self.t1)):
            raise ValueError("t1 contains non-finite values")

        if not np.all(np.isfinite(self.t2)):
            raise ValueError("t2 contains non-finite values")

        # Validate energy if provided
        if self.energy is not None and not np.isfinite(self.energy):
            raise ValueError(f"energy must be finite, got {self.energy}")

        # Validate iterations if provided
        if self.iterations is not None and self.iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {self.iterations}")

    def validate_t2_symmetry(self, atol: float = 1e-10) -> None:
        """
        Validate T2 symmetry: t2[i, j, a, b] == t2[j, i, b, a].

        This is an optional validation for symmetric VCC implementations.
        Some parsers may return only the upper triangular part of T2,
        in which case this validation should not be used.

        Parameters
        ----------
        atol : float, default=1e-10
            Absolute tolerance for symmetry check.

        Raises
        ------
        ValueError
            If T2 symmetry is violated beyond tolerance or if basic
            validation fails.

        Examples
        --------
        >>> amps.validate_t2_symmetry(atol=1e-8)
        """
        # First validate basic amplitude properties
        self.validate()

        n_modes = self.t2.shape[0]

        for i in range(n_modes):
            for j in range(n_modes):
                # Use rtol=0.0 for pure absolute tolerance check
                if not np.allclose(self.t2[i, j], self.t2[j, i].T, rtol=0.0, atol=atol):
                    max_diff = np.max(np.abs(self.t2[i, j] - self.t2[j, i].T))
                    raise ValueError(
                        f"T2 symmetry violation for mode pair ({i}, {j}): "
                        f"max difference = {max_diff:.2e} > atol = {atol:.2e}"
                    )


class VCCSolver(ABC):
    """
    Abstract interface for vibrational coupled cluster solvers.

    This abstract base class defines the interface that all VCC solver
    backends (MidasCpp, PySCF, etc.) should implement.

    Examples
    --------
    >>> class MyVCCSolver(VCCSolver):
    ...     def compute_ground_state(self, forcefield, n_modes, n_modals, **kwargs):
    ...         # Implementation here
    ...         return VCCAmplitudes(t1=..., t2=...)
    """

    @abstractmethod
    def compute_ground_state(
        self,
        forcefield,
        n_modes: int,
        n_modals: int,
        **kwargs,
    ) -> VCCAmplitudes:
        """
        Compute VCC/VCCSD ground state amplitudes.

        Parameters
        ----------
        forcefield : VibrationalForceField
            Force field defining the vibrational Hamiltonian.
        n_modes : int
            Number of vibrational modes.
        n_modals : int
            Number of modals (levels) per mode.
        **kwargs
            Additional solver-specific parameters.

        Returns
        -------
        VCCAmplitudes
            Computed amplitudes with metadata.

        Raises
        ------
        RuntimeError
            If the VCC calculation fails to converge.
        """
        raise NotImplementedError
