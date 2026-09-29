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

"""Molecule-agnostic harmonic-oscillator (HO) product-basis Hamiltonian.

These four functions were previously copy-pasted, verbatim or near-verbatim,
into six separate per-molecule ``*_forcefield.py`` modules (h2o, ch2o,
ch2fcl, c2h2, nh3, cnh3). They depend only on a generic force-field object
(``ff.omega``, ``ff.terms``) and integer sizes (``n_ho``, ``n_modes``,
``cfg``/``cfgs``) -- nothing molecule-specific. Each per-molecule module now
just supplies its own force field and calls these directly.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from vib_sqd.basis.harmonic import number_operator, q_power_matrix


def compact_index(cfg, n_modals):
    """Row-major index of an occupation tuple into the implicit
    product(range(n_modals), repeat=n_modes) ordering used by
    ho_hamiltonian_full / ho_hamiltonian_sparse (matches np.kron's index
    convention: first mode is the most-significant digit)."""
    idx = 0
    for c in cfg:
        idx = idx * n_modals + c
    return idx


def ho_diagonal_element(ff, n_ho, cfg):
    """A single diagonal element H[idx,idx] of the compact HO Hamiltonian
    (same physics as ho_hamiltonian_sparse), computed DIRECTLY from each
    single-mode operator's own diagonal entry at `cfg` -- never forms the
    full dim=n_ho**n_modes sparse matrix. A Kronecker product's diagonal
    entry at multi-index cfg=(c_0,...,c_{n-1}) is just the product of each
    factor's (c_i,c_i) diagonal entry, so this is exact, not an
    approximation (verified against ho_hamiltonian_sparse's full matrix to
    1e-10 for CH2ClF, 2026-09-06; identical formula, molecule-agnostic).
    Much faster than building the full matrix just to read one diagonal
    entry (the reference-config energy)."""
    n_modes = len(ff.omega)
    num_diag = np.diag(number_operator(n_ho - 1).astype(complex).real)
    qp_diag = {
        p: np.diag(q_power_matrix(n_ho - 1, p).astype(complex).real)
        for p in range(1, 5)
    }

    e = 0.0
    for i, w in enumerate(ff.omega):
        e += float(w) * num_diag[cfg[i]]
    e += 0.5 * float(sum(ff.omega))

    for term in ff.terms:
        powers = [0] * n_modes
        for m, p in zip(term.modes, term.powers):
            powers[m - 1] += p
        deg = sum(powers)
        act = [k for k, p in enumerate(powers) if p]
        if deg == 2 and len(act) == 1 and powers[act[0]] == 2:
            continue
        val = 1.0
        for k, p in enumerate(powers):
            if p:
                val *= qp_diag[p][cfg[k]]
        e += float(term.coeff) * val
    return e.real if hasattr(e, "real") else e


def ho_submatrix(ff, n_ho, cfgs):
    """The dense (len(cfgs) x len(cfgs)) matrix of ho_hamiltonian_sparse's
    Hamiltonian RESTRICTED to the given list of configurations -- computed
    directly from each single-mode operator's matrix elements at the
    relevant config pairs, WITHOUT ever forming the full sparse
    dim=n_ho**n_modes matrix.

    A Kronecker-product operator's matrix element between two multi-index
    configs cfg_a=(a_0,...,a_{n-1}) and cfg_b=(b_0,...,b_{n-1}) is the
    PRODUCT of each factor's (a_i, b_i) matrix element -- exact, not an
    approximation. This matters because recovery_loop only ever needs a
    basis_size x basis_size (e.g. 200x200) submatrix of H_ho, indexed by the
    sampled configs each iteration; building the FULL sparse matrix just to
    slice out this tiny submatrix is not just wasteful but a real memory-
    crash risk at large n_ho**n_modes (confirmed: CH2ClF n_modals=6 -> dim
    6**9 ~ 1.0e7, two independent OOM kills building the full sparse matrix
    via scipy.sparse.kron, 2026-09-06). This function replaces that full
    build with direct element-by-element evaluation, using the same
    single-mode operators (number_operator, q_power_matrix) as
    ho_hamiltonian_sparse/ho_diagonal_element -- same physics, verified
    against the full matrix to machine precision for small test cases."""
    n_modes = len(ff.omega)
    n_cfg = len(cfgs)
    cfg_arr = np.asarray(cfgs, dtype=np.int64)  # (n_cfg, n_modes)

    num_op = number_operator(n_ho - 1).astype(complex).real
    qp_op = {p: q_power_matrix(n_ho - 1, p).astype(complex).real for p in range(1, 5)}

    H = np.zeros((n_cfg, n_cfg), dtype=complex)

    # Harmonic diagonal: sum_i omega_i (n_i + 1/2), diagonal in the HO basis
    # (number operator is diagonal), so only contributes to H[k,k].
    diag_const = 0.5 * float(sum(ff.omega))
    for i, w in enumerate(ff.omega):
        H[np.arange(n_cfg), np.arange(n_cfg)] += (
            float(w) * num_op[cfg_arr[:, i], cfg_arr[:, i]]
        )
    H[np.arange(n_cfg), np.arange(n_cfg)] += diag_const

    for term in ff.terms:
        powers = [0] * n_modes
        for m, p in zip(term.modes, term.powers):
            powers[m - 1] += p
        deg = sum(powers)
        act = [k for k, p in enumerate(powers) if p]
        if deg == 2 and len(act) == 1 and powers[act[0]] == 2:
            continue  # already counted in the harmonic diagonal above
        # This term's matrix element between cfg_a, cfg_b is the product,
        # over every mode, of that mode's operator's (a_i,b_i) entry
        # (identity, i.e. Kronecker delta, for modes this term doesn't touch).
        elem = np.ones((n_cfg, n_cfg), dtype=complex)
        for k, p in enumerate(powers):
            if p == 0:
                # identity factor: delta(a_k, b_k)
                elem = elem * (cfg_arr[:, k][:, None] == cfg_arr[:, k][None, :])
            else:
                op = qp_op[p]
                elem = elem * op[cfg_arr[:, k][:, None], cfg_arr[:, k][None, :]]
        H += float(term.coeff) * elem
    return H


def ho_hamiltonian_sparse(ff, n_ho):
    """Sparse HO product Hamiltonian (dim n_ho**n_modes), same physics as
    vib_sqd.basis.modal_transform.ho_hamiltonian_full but built with
    scipy.sparse.kron throughout so it stays tractable at n_ho**n_modes >
    ~10^4. Each single-mode operator (identity, number operator, q^p) is
    tiny and dense; only the multi-mode Kronecker products are kept sparse."""
    n_modes = len(ff.omega)
    ident = sparse.identity(n_ho, format="csr", dtype=complex)
    num = sparse.csr_matrix(number_operator(n_ho - 1).astype(complex))
    qp = {
        p: sparse.csr_matrix(q_power_matrix(n_ho - 1, p).astype(complex))
        for p in range(1, 5)
    }
    dim = n_ho**n_modes

    def kron_all(ops):
        out = ops[0]
        for o in ops[1:]:
            out = sparse.kron(out, o, format="csr")
        return out

    H = sparse.csr_matrix((dim, dim), dtype=complex)
    for i, w in enumerate(ff.omega):
        ops = [ident] * n_modes
        ops[i] = num
        H = H + float(w) * kron_all(ops)
    H = H + 0.5 * float(sum(ff.omega)) * sparse.identity(
        dim, format="csr", dtype=complex
    )

    for term in ff.terms:
        powers = [0] * n_modes
        for m, p in zip(term.modes, term.powers):
            powers[m - 1] += p
        deg = sum(powers)
        act = [k for k, p in enumerate(powers) if p]
        if deg == 2 and len(act) == 1 and powers[act[0]] == 2:
            continue
        ops = [ident if p == 0 else qp[p] for p in powers]
        H = H + float(term.coeff) * kron_all(ops)
    return H.tocsr()
