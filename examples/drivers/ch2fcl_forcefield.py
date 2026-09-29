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

"""CH2ClF (fluorochloromethane) molecule description: force field path,
mode assignments, and the classical reference energy table used throughout
the run_ch2fcl.py sweep and report generation.

Replaces C2H2/CNH3 in the manuscript's molecule set: classical VCI energies
for those two systems fail to converge as the HO-basis size grows (the
Taylor-expanded potential is not bounded from below, so large-quantum-number
HO basis functions probe spurious low-energy regions of the fit) -- a real
force-field limitation, not a pipeline bug. CH2ClF is the replacement third
molecule.

REFERENCE_TABLE below is transcribed verbatim from the manuscript's
tab:classicalrefs (report/manuscript.tex, sourced from Erik Lotstedt's
01-Sep-2026 note reproduced in the comment block just below that table):
ZPE and the lowest fundamental excitation Delta E1 only (not a full
per-mode fundamental table like CH2O's) -- CH2ClF's VFCI energy was not
computed (the 11^9 x 11^9 Hamiltonian is intractable), so only HO/VSCF/
VCC(M=2,4,6,8)/VHCI(5 basis sizes) rows exist."""

from __future__ import annotations

from pathlib import Path

from vib_sqd.basis.ho_hamiltonian import (  # noqa: F401
    compact_index,
    ho_diagonal_element,
    ho_submatrix,
    ho_hamiltonian_sparse,
)

_HERE = Path(__file__).resolve().parent
CH2FCL_MOP = _HERE.parents[1] / "data" / "mop_files" / "CH2FCl_gaussian.mop"

N_MODES_FULL = 9
N_HO = 11

# (mode_name_in_mop, physical assignment, harmonic frequency cm^-1)
# Only mode 5 has a confirmed physical assignment (manuscript: "the nu_6 ClCF
# bending mode for CH2ClF", 1-indexed nu_6 = 0-indexed mode 5, which also has
# the smallest #1 ScaleFactors entry in the .mop -- consistent with "lowest
# fundamental"). The other 8 modes' physical assignments are not given
# anywhere in this repo; left unnamed rather than guessed.
MODE_ASSIGNMENTS = {
    5: ("Q5", "ClCF bending", None),
}

# Lowest-frequency fundamental (used as the "excited state" example for this
# molecule, same convention as H2O/CH2O/C2H2/CNH3's LOWEST_FUNDAMENTAL_MODE):
# mode 5, ClCF bending -- confirmed both by the manuscript prose (nu_6) and by
# the .mop's #1 ScaleFactors block (mode 5 has the smallest scale factor,
# 4.1416e-2, of all 9 modes).
LOWEST_FUNDAMENTAL_MODE = 5

# User-supplied reference table, cm^-1: ground state (ZPE) + the lowest
# fundamental (Delta E1, the nu_6 ClCF-bending excitation) only -- this
# molecule's manuscript table does not include all 9 fundamentals like
# CH2O's REFERENCE_TABLE does. Reproduced verbatim from tab:classicalrefs.
REFERENCE_TABLE = {
    "v_labels": ["000000000", "000001000"],
    "HO": [6768.27, 6768.27 + 376.46],
    "VSCF": [6708.12, 6708.12 + 372.27],
    "VCC (M=2)": [6701.56, 6701.56 + 372.19],
    "VCC (M=4)": [6682.03, 6682.03 + 371.23],
    "VCC (M=6)": [6681.77, 6681.77 + 371.22],
    "VCC (M=8)": [6681.77, 6681.77 + 371.22],
    "VHCI (|S|=55)": [6714.83, 6714.83 + 373.60],
    "VHCI (|S|=1840)": [6681.94, 6681.94 + 372.46],
    "VHCI (|S|=24749)": [6681.60, 6681.60 + 371.29],
    "VHCI (|S|=190167)": [6681.59, 6681.59 + 371.21],
    "VHCI (|S|=1006515)": [6681.59, 6681.59 + 371.20],
}
