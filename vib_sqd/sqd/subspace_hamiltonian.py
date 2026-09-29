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

"""Build the modal-basis vibrational Hamiltonian RESTRICTED to a set of sampled
configurations -- without ever forming the full prod(M) tensor product.

This is what makes large-mode SQD tractable: for N modes at M modals the full
modal Hamiltonian is M^N-dimensional (4^8 = 65536, dense ~34 GB), but SQD only
needs H projected onto the (few hundred) configurations that appear in the
sample. Each matrix element <a|H|b> factorizes over modes:

    <a|H|b> = sum_omega  omega_i (a_i + 1/2) [diagonal, a==b]
            + sum_terms  coeff * prod_{modes in term} <a_i| Q^{k}_i |b_i>
                                * prod_{other modes}  delta(a_j, b_j)

where Q^{k}_i = C_i q^k C_i^dagger is the per-mode MODAL q^k operator (a small
M x M matrix; C_i is the VSCF HO->modal rotation, q^k from
basis.harmonic.q_power_matrix). We precompute Q^{k}_i once per (mode, power) and
assemble the sparse subspace matrix by iterating force-field terms.

Vectorized: for a term touching modes ``act``, two configs can only produce a
nonzero element if they agree on every OTHER ("untouched") mode -- so instead of
a Python double loop over all n_cfg^2 pairs, configs are grouped by their values
on the untouched-mode columns (one ``np.unique(axis=0)`` pass) and each group's
small g x g block is filled via NumPy broadcasting. Real force fields only have
terms of arity <=3 even at 20+ modes, so most groups are tiny and the O(n_cfg^2)
cost collapses to close to O(n_cfg) in practice.

H is Hermitian by construction (each Q^{k}_i / number-operator factor is
Hermitian, so H[b,a] = conj(H[a,b])); this is asserted in the test suite.

Verified against the dense modal Hamiltonian (tests/test_subspace_hamiltonian.py:
random subsets and the full product space, both exact to 1e-8).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vib_sqd.basis.harmonic import number_operator, q_power_matrix


def _modal_qpow(rot, n_ho: int, max_power: int) -> dict[tuple[int, int], np.ndarray]:
    """Precompute Q^k_i = C_i q^k C_i^dagger for every mode i and power k in
    1..max_power. rot[i] is the (M_i, n_ho) VSCF rotation for mode i."""
    cache: dict[tuple[int, int], np.ndarray] = {}
    qk_ho = {k: q_power_matrix(n_ho - 1, k) for k in range(1, max_power + 1)}
    for i, C in enumerate(rot):
        Ci = np.asarray(C)  # (M, n_ho)
        for k in range(1, max_power + 1):
            cache[(i, k)] = Ci @ qk_ho[k] @ Ci.conj().T  # (M, M) modal q^k
    return cache


def build_subspace_hamiltonian(
    configs: Sequence[tuple[int, ...]],
    forcefield,
    rot,
    n_ho: int,
    harmonic_omega: Sequence[float] | None = None,
):
    """Return the dense (n_configs x n_configs) modal Hamiltonian projected onto
    ``configs`` (each a length-N occupation tuple of modal indices).

    ``forcefield.terms`` are (coeff, modes[1-based], powers); ``rot`` is the list
    of per-mode VSCF rotations; ``harmonic_omega`` (defaults to forcefield.omega)
    supplies the Sum omega_i (n_i + 1/2) harmonic diagonal (omega in a.u.).
    """
    configs = [tuple(int(x) for x in c) for c in configs]
    n_cfg = len(configs)
    n_modes = len(configs[0])

    # max aggregated per-mode power across all terms (a term may repeat a mode)
    def _agg_max(t):
        pm: dict[int, int] = {}
        for m, p in zip(t.modes, t.powers):
            pm[m] = pm.get(m, 0) + p
        return max(pm.values(), default=0)

    max_power = max((_agg_max(t) for t in forcefield.terms), default=2)
    Q = _modal_qpow(rot, n_ho, max_power)

    omega = (
        list(harmonic_omega)
        if harmonic_omega is not None
        else list(getattr(forcefield, "omega", []))[:n_modes]
    )

    H = np.zeros((n_cfg, n_cfg), dtype=complex)
    cfg_arr = np.asarray(configs, dtype=np.int64)  # (n_cfg, n_modes)

    # Harmonic term:  sum_i omega_i (n_i + 1/2).  CRITICAL: the VSCF modals are
    # NOT number eigenstates, so in the MODAL basis this is NOT diagonal -- it is
    # sum_i omega_i (C_i n C_i^dagger)  plus the 0.5*omega_i constant on the
    # diagonal.  Build the per-mode modal number operator like the modal q^k ops.
    if omega:
        num_ho = number_operator(n_ho - 1)
        Nmod = {
            i: np.asarray(rot[i]) @ num_ho @ np.asarray(rot[i]).conj().T
            for i in range(n_modes)
        }
        half = sum(0.5 * omega[i] for i in range(n_modes))
        H[np.arange(n_cfg), np.arange(n_cfg)] += half
        for i in range(n_modes):
            # configs sharing this group differ, at most, on mode i -- i.e. agree
            # on every "untouched" (here: every OTHER) mode -- exactly the
            # `len(diff) <= 1` condition of the original per-pair loop.
            _add_grouped_term(
                H,
                cfg_arr,
                act=[i],
                per_mode={i: None},
                coeff=omega[i],
                factor_lookup=lambda mi, _pm: Nmod[mi],
            )

    # Anharmonic / coupling terms.  Match ho_hamiltonian_full EXACTLY: aggregate
    # powers per mode within a term, then skip a term iff its aggregated action is
    # a single-mode q^2 (already in the harmonic omega diagonal). Each touched
    # mode contributes its modal q^(aggregated power) element; untouched modes
    # must match (delta) for a nonzero <a|H|b>.
    for term in forcefield.terms:
        per_mode: dict[int, int] = {}
        for m, p in zip(term.modes, term.powers):
            per_mode[m - 1] = per_mode.get(m - 1, 0) + p
        act = [mi for mi, p in per_mode.items() if p]
        deg = sum(per_mode.values())
        if omega and deg == 2 and len(act) == 1 and per_mode[act[0]] == 2:
            continue  # pure single-mode q^2 -> in omega
        coeff = float(term.coeff)
        _add_grouped_term(
            H,
            cfg_arr,
            act=act,
            per_mode=per_mode,
            coeff=coeff,
            factor_lookup=lambda mi, pm: Q[(mi, pm[mi])],
        )
    return H


def _add_grouped_term(H, cfg_arr, act, per_mode, coeff, factor_lookup):
    """Add one term's contribution to ``H`` in place, vectorized:
    ``H[a,b] += coeff * prod_{mi in act} factor[mi][ca_mi, cb_mi]``.

    Two configs can only produce a nonzero matrix element for this term if
    they agree on every mode NOT in ``act`` (the "untouched" modes) -- so
    group configs by their values on the untouched-mode columns
    (``np.unique(..., axis=0)``, one vectorized pass, no per-pair Python
    loop) and do the small per-group ``g x g`` multiply/add via NumPy
    broadcasting on each touched mode's small (M, M) per-mode matrix.
    """
    n_cfg, n_modes = cfg_arr.shape
    act_arr = np.asarray(act, dtype=np.int64)
    untouched_mask = np.ones(n_modes, dtype=bool)
    untouched_mask[act_arr] = False
    untouched_idx = np.nonzero(untouched_mask)[0]

    key_cols = (
        cfg_arr[:, untouched_idx]
        if len(untouched_idx)
        else np.zeros((n_cfg, 0), dtype=np.int64)
    )
    _, group_id = np.unique(key_cols, axis=0, return_inverse=True)
    group_id = group_id.reshape(-1)
    order = np.argsort(group_id, kind="stable")
    sorted_gid = group_id[order]
    starts = np.searchsorted(sorted_gid, np.arange(sorted_gid[-1] + 1))
    ends = np.r_[starts[1:], len(order)]

    for g_start, g_end in zip(starts, ends):
        members = order[g_start:g_end]
        val = np.full((len(members), len(members)), coeff, dtype=complex)
        for mi in act_arr:
            factor = factor_lookup(int(mi), per_mode)
            a_idx = cfg_arr[members, mi]
            val *= factor[np.ix_(a_idx, a_idx)]
        H[np.ix_(members, members)] += val
