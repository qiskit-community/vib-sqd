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

"""H2O (water) molecule description: force field path, mode assignments,
and the classical reference energy table used throughout the
run_h2o_sweep.py sweep and report generation.

Water already has a full worked example in report.tex/pipeline_walkthrough.tex
(the original single-molecule proof of the MidasCpp -> VCC -> SQD pipeline),
but no dedicated forcefield module like ch2o_forcefield.py/nh3_forcefield.py
existed yet -- this factors out exactly the same shape, sourced from numbers
already reported there rather than invented:
  - ground-state ZPE ~4614.95/4614.93 cm^-1 (OccGroundState vs MidasCpp VCI,
    pipeline_walkthrough.tex / report.tex)
  - per-mode fundamental excitation energies via fundamentals.py's #3 Occup
    method vs OccAllFund, agreeing to ~0.1 cm^-1: 3671.0/1552.3/3752.4 vs
    3670.8/1552.2/3752.3 cm^-1 (pipeline_walkthrough.tex line ~610)
No VHCI-ladder/VibCC comparison table (like CH2O's) has been run for water in
this repo -- REFERENCE_TABLE below is ground-state + fundamentals only,
labeled by method, not a convergence ladder."""

from __future__ import annotations

from pathlib import Path

from vib_sqd.basis.ho_hamiltonian import (  # noqa: F401
    compact_index,
    ho_diagonal_element,
    ho_submatrix,
    ho_hamiltonian_sparse,
)

_HERE = Path(__file__).resolve().parent
H2O_MOP = _HERE.parents[1] / "data" / "mop_files" / "h2o_dimless_qff_numeric.mop"

N_MODES_FULL = 3
N_HO = 11

# (mode_name_in_mop, physical assignment, harmonic/fundamental frequency cm^-1)
# Mode-to-frequency assignment verified via ScaleFactors ordering (larger
# scale factor -> higher frequency, direction cross-checked against CH2O's
# known ScaleFactor/frequency ordering): Q1 has the smallest scale factor
# (0.08616) -> lowest frequency -> bend; Q2 the largest (0.13341) -> highest
# frequency -> antisym stretch; Q0 (0.13166) is in between -> sym stretch.
# Frequencies themselves are the fundamentals.py-computed values (Occup
# method), not a separate harmonic-only calculation.
MODE_ASSIGNMENTS = {
    0: ("Q0", "sym OH stretch", 3671.0),
    1: ("Q1", "HOH bend", 1552.3),
    2: ("Q2", "antisym OH stretch", 3752.4),
}

DEGENERATE_MODE_PAIRS: list[tuple[int, int]] = []

# Lowest-frequency fundamental (used as the "excited state" example for this
# molecule): mode 1 (HOH bend, 1552.3 cm^-1).
LOWEST_FUNDAMENTAL_MODE = 1

# Ground state (OccGroundState, MidasCpp VCC[2]) + per-mode fundamentals (via
# fundamentals.py's #3 Occup method), both already reported in
# pipeline_walkthrough.tex. No VHCI/VibCC convergence ladder exists for water
# in this repo (unlike CH2O) -- do not add rows here without an actual run.
REFERENCE_TABLE = {
    "v_labels": ["000", "100", "010", "001"],
    "VCC[2] (OccGroundState/Occup)": [
        4614.95,
        4614.95 + 3671.0,
        4614.95 + 1552.3,
        4614.95 + 3752.4,
    ],
}
