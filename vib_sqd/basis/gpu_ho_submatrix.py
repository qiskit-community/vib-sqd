#!/usr/bin/env python3

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

r"""
GPU (cupy) drop-in replacement for each molecule forcefield module's own
ho_submatrix(ff, n_ho, cfgs) -- identical physics (verified: same
element-by-element Kronecker-factor evaluation, see e.g.
ch2fcl_forcefield.py's ho_submatrix docstring), only the array backend
changes (numpy -> cupy) so the O(n_terms * n_cfg^2) elementwise multiply
loop runs on GPU instead of CPU. This is the real bottleneck the HO-basis
sweep hit locally: measured 0.36s at 200 configs but 2.73s at 500 (worse
than quadratic) for one real CH2ClF cell on CPU, with ~300 terms each
doing an (n_cfg x n_cfg) elementwise multiply -- exactly the kind of
embarrassingly-parallel workload cupy accelerates well.

The final returned matrix is always a numpy array (converted back via
cupy.asnumpy) so callers (np.linalg.eigh in sweep_basis_size_ground.py)
never need to know GPU was involved.

Usage: import this module and pass gpu_ho_submatrix.make_ho_submatrix_gpu(mod)
as the ho_submatrix_fn for a molecule module `mod` (h2o_forcefield,
ch2o_forcefield, or ch2fcl_forcefield), in place of that module's own
CPU ho_submatrix.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def _ho_submatrix_gpu(ff, n_ho, cfgs):
    from vib_sqd.basis.harmonic import number_operator, q_power_matrix

    n_modes = len(ff.omega)
    n_cfg = len(cfgs)
    cfg_arr = cp.asarray(np.asarray(cfgs, dtype=np.int64))  # (n_cfg, n_modes)

    num_op = cp.asarray(number_operator(n_ho - 1).astype(complex).real)
    qp_op = {
        p: cp.asarray(q_power_matrix(n_ho - 1, p).astype(complex).real)
        for p in range(1, 5)
    }

    H = cp.zeros((n_cfg, n_cfg), dtype=complex)

    diag_const = 0.5 * float(sum(ff.omega))
    idx = cp.arange(n_cfg)
    for i, w in enumerate(ff.omega):
        H[idx, idx] += float(w) * num_op[cfg_arr[:, i], cfg_arr[:, i]]
    H[idx, idx] += diag_const

    for term in ff.terms:
        powers = [0] * n_modes
        for m, p in zip(term.modes, term.powers):
            powers[m - 1] += p
        deg = sum(powers)
        act = [k for k, p in enumerate(powers) if p]
        if deg == 2 and len(act) == 1 and powers[act[0]] == 2:
            continue
        elem = cp.ones((n_cfg, n_cfg), dtype=complex)
        for k, p in enumerate(powers):
            if p == 0:
                elem = elem * (cfg_arr[:, k][:, None] == cfg_arr[:, k][None, :])
            else:
                op = qp_op[p]
                elem = elem * op[cfg_arr[:, k][:, None], cfg_arr[:, k][None, :]]
        H += float(term.coeff) * elem

    return cp.asnumpy(H)


def make_ho_submatrix_gpu(mod):
    """Returns a ho_submatrix(ff, n_ho, cfgs) callable that runs on GPU if
    cupy is available, falling back to `mod`'s own CPU implementation
    otherwise (never silently fabricates a result -- just uses the slower
    but correct path)."""
    if cp is None:
        return mod.ho_submatrix
    return _ho_submatrix_gpu


def gpu_available():
    if cp is None:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False
