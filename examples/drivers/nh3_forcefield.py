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

"""NH3 (ammonia) molecule description: force field path, mode assignments,
and the classical reference energy table used throughout the run_nh3.py
sweep and report generation.

NH3 has C3v symmetry: modes Q2/Q3 (asymmetric N-H stretch) and Q4/Q5
(asymmetric bend) are exactly degenerate (E-symmetry pairs, confirmed
bit-identical ScaleFactors in the .mop). This pipeline applies NO symmetry
reduction anywhere -- all 6 modes are treated uniformly, exactly like CH2O,
matching how the user's own classical VHCI reference calculation was run.
The degeneracy is recorded here (DEGENERATE_MODE_PAIRS) purely for the
presentation writeup; a symmetry-aware MidasCpp calculation (which could
shrink the effective configuration space for these modes) is a flagged
future direction, not implemented."""

from __future__ import annotations

from pathlib import Path

from vib_sqd.basis.ho_hamiltonian import (
    compact_index,
    ho_diagonal_element,
    ho_hamiltonian_sparse,
    ho_submatrix,
)

_HERE = Path(__file__).resolve().parent
NH3_MOP = _HERE.parents[1] / "data" / "mop_files" / "NH3_Gaussian.mop"

N_MODES_FULL = 6
N_HO = 11

# (mode_name_in_mop, physical assignment, harmonic frequency cm^-1)
MODE_ASSIGNMENTS = {
    0: ("Q0", "sym N-H stretch (a1)", 3469.3),
    1: ("Q1", "umbrella/inversion bend (a1)", 1025.3),
    2: ("Q2", "asym N-H stretch, degenerate pair (e)", 3588.3),
    3: ("Q3", "asym N-H stretch, degenerate pair (e)", 3588.3),
    4: ("Q4", "asym bend, degenerate pair (e)", 1663.6),
    5: ("Q5", "asym bend, degenerate pair (e)", 1663.6),
}

# C3v E-symmetry degenerate mode pairs, exact in the .mop's ScaleFactors
# (verified bit-identical). No symmetry reduction is applied anywhere in
# this pipeline -- documented here for the presentation writeup only.
DEGENERATE_MODE_PAIRS = [(2, 3), (4, 5)]

# User-supplied reference table, cm^-1: GROUND STATE ONLY (no per-fundamental
# literature numbers were supplied for NH3, unlike CH2O). Reproduced verbatim
# for README/RESULTS. See run_nh3.py's --fundamentals flag and
# make_nh3_report.py's computed-fundamentals table for the (as-computed, not
# literature-validated) singly-excited fundamental energies.
REFERENCE_TABLE = {
    "v_labels": ["000000"],
    "HO": [7499.3],
    "VibSCF": [7390.1],
    "VHCI(28)": [7398.4],
    "VHCI(310)": [7358.5],
    "VHCI(1255)": [7357.7],
    "VibCC(VCC[3])": [7358.1],
    "Exact": [7357.7],
}
