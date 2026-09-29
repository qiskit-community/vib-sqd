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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class DiagResult:
    evals: np.ndarray
    evecs: np.ndarray


def diagonalize_local_dense(h_proj: np.ndarray, n_roots: int = 3) -> DiagResult:
    evals, evecs = np.linalg.eigh(h_proj)
    idx = np.argsort(evals)[:n_roots]
    return DiagResult(evals=evals[idx], evecs=evecs[:, idx])


def diagonalize_local_sparse(h_proj: sp.spmatrix, n_roots: int = 3) -> DiagResult:
    evals, evecs = spla.eigsh(h_proj, k=n_roots, which="SA")
    idx = np.argsort(evals)
    return DiagResult(evals=evals[idx], evecs=evecs[:, idx])


def save_projected_problem(path: Path, h_proj: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, h_proj)
