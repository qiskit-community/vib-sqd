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

"""CH2O (formaldehyde) molecule description: force field path, mode
assignments, and the classical reference energy table used throughout the
run_ch2o.py sweep and report generation."""

from __future__ import annotations

from pathlib import Path

from vib_sqd.basis.ho_hamiltonian import (  # noqa: F401
    compact_index,
    ho_diagonal_element,
    ho_submatrix,
    ho_hamiltonian_sparse,
)

_HERE = Path(__file__).resolve().parent
CH2O_MOP = _HERE.parents[1] / "data" / "mop_files" / "CH2O_Gaussian.mop"

N_MODES_FULL = 6
N_HO = 11

# (mode_name_in_mop, physical assignment, harmonic frequency cm^-1)
MODE_ASSIGNMENTS = {
    0: ("Q0", "asym CH stretch", 2939.3),
    1: ("Q1", "sym CH stretch", 2884.6),
    2: ("Q2", "CO stretch", 1813.1),
    3: ("Q3", "CH2 bending", 1530.2),
    4: ("Q4", "CH2 rocking", 1262.9),
    5: ("Q5", "CH2 wagging", 1198.2),
}

# Lowest-frequency fundamental (used as the "excited state" example for this
# molecule, same convention as C2H2/CNH3/H2O's LOWEST_FUNDAMENTAL_MODE).
LOWEST_FUNDAMENTAL_MODE = 5

# User-supplied reference table, cm^-1: ground state + 6 singly-excited
# fundamentals, across several levels of classical theory. "Exact" is the
# converged VHCI(2708) result. Reproduced verbatim for README/RESULTS.
REFERENCE_TABLE = {
    "v_labels": ["000000", "100000", "010000", "001000", "000100", "000010", "000001"],
    "HO": [5814.1, 8753.5, 8698.7, 7627.2, 7344.3, 7077.0, 7012.3],
    "VibSCF": [5755.7, 8543.8, 8554.3, 7551.1, 7254.2, 6996.3, 6921.8],
    "VHCI(28)": [5761.3, 8626.8, 8554.0, 7557.2, 7259.6, 7002.5, 6930.5],
    "VHCI(189)": [5740.3, 8562.5, 8508.3, 7533.6, 7234.5, 6976.9, 6904.1],
    "VHCI(724)": [5738.3, 8555.7, 8505.2, 7530.2, 7229.5, 6972.1, 6900.0],
    "VHCI(1726)": [5738.3, 8553.6, 8505.0, 7530.1, 7229.3, 6971.9, 6899.9],
    "VHCI(2708)": [5738.3, 8553.5, 8505.0, 7530.0, 7229.3, 6971.9, 6899.9],
    "VibCC(VCC[3])": [5738.4, 8558.0, 8507.4, 7530.7, 7230.5, 6974.8, 6904.2],
    "Exact": [5738.3, 8553.5, 8505.0, 7530.0, 7229.3, 6971.9, 6899.9],
}
