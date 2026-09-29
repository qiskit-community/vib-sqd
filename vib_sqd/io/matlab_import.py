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

from __future__ import annotations

import numpy as np
from scipy.io import loadmat


def import_matlab_pauli_list(mat_path: str, key: str = "H_Paulilist"):
    data = loadmat(mat_path)
    arr = np.array(data[key])
    pauli_terms = []
    for row in arr:
        coeff = complex(row[0])
        idxs = tuple(int(x) for x in row[1:])
        pauli_terms.append((coeff, idxs))
    return pauli_terms, data


def import_state_labels(mat_path: str, key: str = "v3v2v1"):
    data = loadmat(mat_path)
    return np.array(data[key])
