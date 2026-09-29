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

"""Molecule-agnostic force-field data types.

``ForceTerm``/``VibrationalForceField`` describe an arbitrary polynomial
vibrational potential (harmonic frequencies + anharmonic coupling terms);
nothing here is specific to any one molecule. These previously lived in
``data.forcefields.co2`` (re-exported from ``data.forcefields.h2o``) purely
because that was the first molecule module written -- every consumer that
only needs the generic types, including the real ``.mop`` parser
(``classical.mop_forcefield``) and the generic encoders
(``encoding.binary``, ``encoding.direct``), imports from here instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ForceTerm:
    coeff: float
    modes: tuple[int, ...]
    powers: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.powers is None:
            object.__setattr__(self, "powers", tuple(1 for _ in self.modes))
        elif len(self.powers) != len(self.modes):
            raise ValueError(
                f"powers length {len(self.powers)} must match modes length {len(self.modes)}"
            )
        if any(m < 1 for m in self.modes):
            raise ValueError("ForceTerm modes must be 1-indexed positive integers.")
        assert self.powers is not None, "powers should be set by now"
        if any(p < 1 for p in self.powers):
            raise ValueError("ForceTerm powers must be positive integers.")


@dataclass(frozen=True)
class VibrationalForceField:
    omega: tuple[float, ...]
    terms: tuple[ForceTerm, ...]
    coordinate_kind: str = "dimensionless"
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if any(w <= 0.0 for w in self.omega):
            raise ValueError("All harmonic frequencies must be positive.")
        n_modes = len(self.omega)
        for term in self.terms:
            if any(m > n_modes for m in term.modes):
                raise ValueError(
                    f"ForceTerm {term} references mode outside omega length {n_modes}."
                )
