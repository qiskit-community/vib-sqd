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
Distribution of squared CI coefficients |c_{v1v2v3}|^2 in the vibrational
ground state, compared in the HO basis and the (VSCF) modal basis.

Background
----------
MidasCpp's VCC/VCI output is in the MODAL basis (verified: the .mout says
"modals space ordering with occupied as zero", and prints each modal as a
superposition of HO functions). The two representations are:

  HO basis:     Psi = sum_{v1v2v3} c^{HO}_{v1v2v3}   phi_{v1} phi_{v2} phi_{v3}
  Modal basis:  Psi = sum_{v1v2v3} c^{mod}_{v1v2v3}  psi_{v1} psi_{v2} psi_{v3}
                with  psi_v = sum_n C[v,n] phi_n   (VSCF modal-in-HO matrix)

The SAME physical ground state |Psi> has different coefficient distributions in
the two bases. In the modal basis the single-mode anharmonicity is absorbed into
the modals, so far fewer configurations carry weight (the distribution is much
sparser) -- e.g. HO needs 000,100,200,300,... to describe mode-1 anharmonicity,
while the modal basis needs only psi_0.

Method (self-consistent, uses MidasCpp's actual VSCF modals)
------------------------------------------------------------
1. Parse the per-mode modal-in-HO matrices C^{(i)}[v,n] from the .mout VSCF
   section (the first "Modals for mode" block set = ground-state VSCF).
2. Build the full vibrational Hamiltonian in the HO product basis (dim M^N,
   M = number of HO functions kept per mode) from the .mop force field
   (reusing the repo primitives), and diagonalize -> c^{HO} (the ground state
   in the HO product basis). This is exact within the retained HO space.
3. Rotate into the modal product basis:
   c^{mod}_{v1..} = sum_{n1..} conj(C1[v1,n1]) ... c^{HO}_{n1..}
   (each mode independently, via tensordot). Both vectors describe the SAME
   |Psi>; only the basis differs.
4. Report/plot the sorted |c|^2 spectra.

The HO-space ground energy is cross-checked against MidasCpp's VCI value.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from vib_sqd.basis.harmonic import number_operator, q_power_matrix

CM2AU = 4.556335252912000e-06


# --------------------------------------------------------------------------- #
# Parse VSCF modal-in-HO matrices from a .mout
# --------------------------------------------------------------------------- #
def parse_vscf_modals(mout_path, n_modes):
    """Return list of C^{(i)} arrays, shape (n_modals_printed, n_ho), for the
    FIRST VSCF modal set (ground-state VSCF). Each row v is psi_v in the HO
    basis: psi_v = sum_n C[v,n] phi_n.
    """
    text = Path(mout_path).read_text().splitlines()
    # Find the first "Modals for mode: Qk" for each of the first n_modes modes,
    # in order. We take the first occurrence block-set.
    mats = []
    i = 0
    header_re = re.compile(r"Modals for mode:\s*Q(\d+)")
    float_re = re.compile(r"^\s*-?\d+\.\d+E[-+]\d+\s*$")
    modes_seen = []
    while i < len(text) and len(mats) < n_modes:
        m = header_re.search(text[i])
        if not m:
            i += 1
            continue
        mode = int(m.group(1))
        if mode in modes_seen:
            break  # entered the 2nd VSCF set; stop
        modes_seen.append(mode)
        # Parse modals until the next "Modals for mode" or a blank/again header.
        rows = []
        j = i + 1
        cur = None
        while j < len(text):
            line = text[j]
            # Stop at the next mode header or at any line that ends the modal
            # listing for this mode (VSCF summary / finalize markers). Without
            # this, the LAST mode's block would run into the next VSCF set.
            if header_re.search(line):
                break
            stripped = line.strip()
            if (
                stripped.startswith("The modal energies")
                or stripped.startswith("The sum over modal")
                or stripped.startswith("Iter.")
                or "Vscf::Finalize" in stripped
                or stripped.startswith("Vscf ")
            ):
                break
            if stripped.startswith("modal:"):
                if cur is not None:
                    rows.append(cur)
                cur = []
            elif stripped.startswith("w:"):
                pass  # eigenvalue line, skip
            elif float_re.match(line) and cur is not None:
                cur.append(float(stripped))
            j += 1
        if cur:
            rows.append(cur)
        # rows: list of modal vectors (each length n_ho); ragged-safe
        n_ho = max(len(r) for r in rows)
        C = np.zeros((len(rows), n_ho))
        for v, r in enumerate(rows):
            C[v, : len(r)] = r
        mats.append(C)
        i = j
    if len(mats) != n_modes:
        raise ValueError(f"parsed {len(mats)} modal matrices, expected {n_modes}")
    return mats


# --------------------------------------------------------------------------- #
# Build HO-product-basis Hamiltonian and its ground state
# --------------------------------------------------------------------------- #
def _kron_all(ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def ho_ground_state(ff, n_ho):
    """Ground state of the .mop Hamiltonian in the HO product basis (dim
    n_ho**n_modes). Returns (c_ho reshaped to (n_ho,)*n_modes, E_cm)."""
    n_modes = len(ff.omega)
    ident = np.eye(n_ho)
    num = number_operator(n_ho - 1)
    qpow = {p: q_power_matrix(n_ho - 1, p) for p in range(1, 5)}
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
        ops = [ident if p == 0 else qpow[p] for p in powers]
        H += float(term.coeff) * _kron_all(ops)
    w, V = np.linalg.eigh(H)
    c = V[:, 0]
    c = c / np.linalg.norm(c)
    return c.reshape((n_ho,) * n_modes), float(w[0].real) / CM2AU


# --------------------------------------------------------------------------- #
# Rotate HO -> modal
# --------------------------------------------------------------------------- #
def ho_to_modal(c_ho, modal_mats, n_modals=None):
    """Rotate c^{HO} (shape (n_ho,)*N) into the modal product basis.

    c^{mod}_{v...} = sum_{n...} C1[v1,n1] ... c^{HO}_{n1...}
    Each modal matrix has shape (n_modal_rows, n_ho). We optionally truncate to
    the first n_modals modals per mode. Assumes real orthonormal modals.
    """
    c = c_ho
    N = c.ndim
    for i in range(N):
        C = modal_mats[i]
        n_ho = c.shape[i]
        Ci = C[:, :n_ho]
        if n_modals is not None:
            Ci = Ci[:n_modals, :]
        # contract mode i: c'_{...v...} = sum_n C[v,n] c_{...n...}
        c = np.tensordot(Ci, c, axes=([1], [i]))
        # tensordot puts the new axis first; move it back to position i
        c = np.moveaxis(c, 0, i)
    return c


def sorted_sq(c):
    """Flatten and return sorted-descending |c|^2 (normalized)."""
    p = np.abs(c.ravel()) ** 2
    p = p / p.sum()
    return np.sort(p)[::-1]


def leading_configs(c, n_modes, n_ho, top=6):
    """Return list of (label, |c|^2) for the largest configs."""
    p = np.abs(c.ravel()) ** 2
    p = p / p.sum()
    order = np.argsort(p)[::-1][:top]
    out = []
    shape = c.shape
    for idx in order:
        multi = np.unravel_index(idx, shape)
        out.append(("".join(str(v) for v in multi), float(p[idx])))
    return out
