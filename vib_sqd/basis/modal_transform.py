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

"""
Modal-basis machinery for the vibrational-SQD circuit.

The repo's direct encoding runs in the raw HO basis (each qubit-block level k =
HO function phi_k). MidasCpp's VCC output, however, is in the VSCF *modal* basis
(psi_v = sum_n C_vn phi_n). This module builds the modal-basis Hamiltonian and
amplitudes so the circuit can be initialized/optimized in the same basis
MidasCpp uses, and provides the HO-basis counterparts for a controlled
comparison.

Modal-basis Hamiltonian (verified to reproduce the HO spectrum as M -> n_ho):
    U = kron_i C_i[:M, :n_ho]            (per-mode VSCF rotation, truncated to M)
    H_modal = U H_HO U^dagger            (dim M^N, compact modal product space)
The circuit still uses the one-hot 2^{N*M} qubit space; embed H_modal onto the
physical (one-hot) indices for the state-vector VQE.

Coordinate/qubit conventions match the rest of the pipeline
(q=(a+a+)/sqrt2; energies vs H use Statevector.reverse_qargs()).
"""

from __future__ import annotations

from itertools import product

import numpy as np

from vib_sqd.basis.harmonic import number_operator, q_power_matrix
from vib_sqd.basis.vscf_modals import _kron_all, parse_vscf_modals
from vib_sqd.classical.mop_forcefield import CM2AU


def ho_hamiltonian_full(ff, n_ho):
    """Dense HO product Hamiltonian (dim n_ho**n_modes), harmonic =
    sum_i omega_i (n_i+1/2), plus the .mop anharmonic terms (pure q_i^2 skipped
    to avoid double counting). Same physics as
    build_direct_encoded_vibrational_hamiltonian(harmonic_source=
    'omega_number_operator') but in the compact n_ho product space."""
    n_modes = len(ff.omega)
    ident = np.eye(n_ho)
    num = number_operator(n_ho - 1)
    qp = {p: q_power_matrix(n_ho - 1, p) for p in range(1, 5)}
    dim = n_ho**n_modes
    H = np.zeros((dim, dim), dtype=complex)
    for i, w in enumerate(ff.omega):
        ops = [ident] * n_modes
        ops[i] = num
        H += float(w) * _kron_all(ops)
        H += 0.5 * float(w) * np.eye(dim)
    for term in ff.terms:
        powers = [0] * n_modes
        for m, p in zip(term.modes, term.powers):
            powers[m - 1] += p
        deg = sum(powers)
        act = [k for k, p in enumerate(powers) if p]
        if deg == 2 and len(act) == 1 and powers[act[0]] == 2:
            continue
        ops = [ident if p == 0 else qp[p] for p in powers]
        H += float(term.coeff) * _kron_all(ops)
    return H


def _lowdin_orthonormalize(C):
    """Loewdin (symmetric) orthonormalization of the rows of C (M x n_ho):
    C_on = (C C^dagger)^{-1/2} C, so C_on C_on^dagger = I exactly.

    The printed VSCF modal coefficients, once truncated to M modals of n_ho HO
    levels, are only approximately orthonormal (C C^dagger = I to ~1e-3). That
    truncation artifact makes the per-mode factorized modal Hamiltonian disagree
    with the full U H U^dagger. Restoring exact orthonormality removes the
    artifact and makes the factorization exact (verified: subspace H == dense
    modal H to machine precision)."""
    C = np.asarray(C)
    S = C @ C.conj().T
    w, V = np.linalg.eigh(S)
    Sinv_half = V @ np.diag(w**-0.5) @ V.conj().T
    return Sinv_half @ C


def build_modal_rotation(mout_path, n_modes, n_ho, n_modals, orthonormalize=True):
    """Return per-mode truncated VSCF rotation matrices C_i[:M, :n_ho].

    orthonormalize=True (default) Loewdin-orthonormalizes each truncated C_i so
    C_i C_i^dagger = I exactly -- required for the per-mode factorized
    (subspace) modal Hamiltonian to equal the full U H U^dagger. Set False to
    keep the raw printed (slightly non-orthonormal) coefficients.
    """
    mats = parse_vscf_modals(mout_path, n_modes)
    rot = []
    for C in mats:
        Ci = C[:, :n_ho]
        if Ci.shape[0] < n_modals:
            raise ValueError(
                f"mode has only {Ci.shape[0]} modals printed, need {n_modals}"
            )
        Ci = Ci[:n_modals, :]
        if orthonormalize:
            Ci = _lowdin_orthonormalize(Ci)
        rot.append(Ci)
    return rot


def modal_hamiltonian_compact(H_ho, rot):
    """H_modal = U H_ho U^dagger, dim = prod(M_i). rot[i] has shape (M_i, n_ho)."""
    U = _kron_all(rot)  # (prod M, n_ho**N)
    return U @ H_ho @ U.conj().T


def _one_hot_index(occ, n_modals):
    """Big-endian one-hot index for occupation tuple occ (matches encoding
    convention in encoding/direct.py._physical_indices)."""
    bits = []
    for o in occ:
        block = [0] * n_modals
        block[o] = 1
        bits.extend(block)
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


def embed_onehot(H_compact, n_modes, n_modals):
    """Embed a compact (M^N) Hamiltonian into the 2^{N*M} one-hot qubit space.

    Returns (H_full, physical_indices) where H_full is 2^{NM} x 2^{NM} with the
    compact matrix placed on the physical one-hot indices (ordered the same way
    the compact product basis enumerates occupations: row-major over
    product(range(M), repeat=N)).
    """
    n_qubits = n_modes * n_modals
    if n_qubits > 16:
        raise ValueError(
            f"one-hot embedding needs 2^{n_qubits} which is intractable; "
            f"use a small modal truncation (n_modals) for the circuit "
            f"(N*M <= 16). Use the compact matrix for exact diagonalization."
        )
    dim_full = 2**n_qubits
    occs = list(product(range(n_modals), repeat=n_modes))
    phys = np.array([_one_hot_index(o, n_modals) for o in occs], dtype=int)
    H_full = np.zeros((dim_full, dim_full), dtype=complex)
    H_full[np.ix_(phys, phys)] = H_compact
    return H_full, phys


def ho_amplitudes_from_modal(t1_mod, t2_mod, rot):
    """Rotate modal-basis cluster amplitudes into the HO basis.

    Amplitudes index excited levels (a = modal/HO index - 1). The single- and
    double-excitation amplitudes transform with the (excited-block) rotation
    R_i = C_i[1:M, 1:M] between excited modals and excited HO functions:
        t1_HO[i,a] = sum_b R_i[b,a] t1_mod[i,b]         (per mode)
        t2_HO[i,j,a,c] = sum_{b,d} R_i[b,a] R_j[d,c] t2_mod[i,j,b,d]
    This is an approximation that keeps the excited-space block (the reference
    modal 0 maps predominantly to HO 0). Used only for the HO-basis comparison
    circuit; the MODAL amplitudes are the MidasCpp-native ones.
    """
    n_modes, n_exc = t1_mod.shape
    # Excited-excited blocks (drop the reference row/col), shape (n_exc, n_exc).
    R = [rot[i][1 : 1 + n_exc, 1 : 1 + n_exc] for i in range(n_modes)]
    t1_ho = np.zeros_like(t1_mod)
    for i in range(n_modes):
        t1_ho[i] = R[i].T @ t1_mod[i]
    t2_ho = np.zeros_like(t2_mod)
    for i in range(n_modes):
        for j in range(n_modes):
            if not np.any(t2_mod[i, j]):
                continue
            t2_ho[i, j] = R[i].T @ t2_mod[i, j] @ R[j]
    return t1_ho, t2_ho


def exact_ground_cm(H_compact):
    return float(np.linalg.eigvalsh(H_compact)[0].real) / CM2AU
