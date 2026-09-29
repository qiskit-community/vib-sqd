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

"""Convergence decisions for a basis-size sweep: given a trajectory of
(basis_size, energy_cm) pairs from re-diagonalizing an increasingly large
subspace, decide which point is "converged."

The two cases are genuinely different, not the same check applied twice:

* **Ground state** (:func:`converged_point_variational`) has a real
  variational guarantee -- a larger subspace can never make the estimate
  worse -- so the correct answer is simply the GLOBAL MINIMUM over the
  whole trajectory. An earlier single-consecutive-step plateau heuristic
  was found to produce a real bug here: a coincidental early flat spot
  (e.g. two adjacent steps agreeing to within the threshold) can lock in a
  value before the trajectory resumes descending toward a genuinely lower
  energy later in the same schedule.

* **Excited state** (:func:`converged_point_plateau`) has no such
  guarantee (energy selection is by maximum overlap with a reference
  configuration, not a variational minimum), so "always take the largest
  basis tried" is not a real convergence check, and neither is a
  single-step plateau test -- a still-descending trajectory can pass
  through a brief flat spot before resuming a multi-cm^-1 descent. The fix
  used here requires a plateau candidate to hold for the REST of the
  schedule (every later point within ``plateau_tol`` of the candidate),
  not just the next step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict


class BasisSizePoint(TypedDict):
    basis_size: int
    energy_cm: float


def converged_point_variational(
    energies: Sequence[BasisSizePoint],
) -> tuple[int, float]:
    """Ground-state convergence: the global minimum over the trajectory.

    Returns ``(converged_basis_size, converged_energy_cm)``.
    """
    best = min(energies, key=lambda e: e["energy_cm"])
    return int(best["basis_size"]), float(best["energy_cm"])


def converged_point_plateau(
    energies: Sequence[BasisSizePoint],
    plateau_tol: float,
) -> tuple[int, float, bool]:
    """Excited-state convergence: the earliest point after which every
    later point in the trajectory stays within ``plateau_tol`` of it.

    Returns ``(converged_basis_size, converged_energy_cm, plateau_found)``.
    Note the loop's own plateau test is trivially satisfied at the last
    index (a single-element "rest of the schedule" always agrees with
    itself), so a real, non-trivial plateau is distinguished from "no
    plateau, just reporting the last point tried" by whether the accepted
    basis_size differs from the schedule's last one -- matching this
    module's original inline implementation exactly, not a redefinition.
    """
    plateau_basis_size = None
    plateau_energy = None
    for i in range(1, len(energies)):
        candidate = energies[i]["energy_cm"]
        rest = energies[i:]
        if all(abs(e["energy_cm"] - candidate) < plateau_tol for e in rest):
            plateau_basis_size = int(energies[i]["basis_size"])
            plateau_energy = float(candidate)
            break
    last = energies[-1]
    if plateau_basis_size is None:
        plateau_basis_size = int(last["basis_size"])
        plateau_energy = float(last["energy_cm"])
    plateau_found = plateau_basis_size != int(last["basis_size"])
    return plateau_basis_size, plateau_energy, plateau_found
