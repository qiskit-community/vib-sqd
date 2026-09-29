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
What does VIm-uCJ actually do?  --  matrix-level analysis of the imaginary
Jastrow correlator, contrasted with the inert VLUCJ diagonal CP.

We build EVERY matrix directly from the circuit primitives used by
``vlucj_extensions.build_vim_ucj`` (via qiskit Operator), so the analysis
describes the real gates, not an idealization.  Four analyses, one per panel:

  (1) GENERATOR matrices.  The correlator is exp(theta * G).  We extract G for
      (a) the VIm-uCJ paired double-excitation and (b) the VLUCJ same-modal CP
      Jastrow, restricted to the physical one-hot subspace of a 2-mode/2-modal
      system, and show G_imucj is real ANTI-symmetric (off-diagonal, rotates
      amplitude) while G_cp is DIAGONAL (imaginary phase generator, inert on the
      reference).

  (2) REFERENCE -> CONFIG transfer map.  |<config | U(theta) | ref>|^2 as a
      function of theta for each ansatz, showing VIm-uCJ moves weight onto the
      doubly-excited config while VLUCJ leaves the reference untouched.

  (3) t2 -> theta INITIALIZATION mapper.  A concrete warm start
      theta_pair = clip(scale * t2[i,j,a,b]) for the VIm-uCJ correlators, tested
      on glycine-4: does it beat zero-init at the FIRST optimizer step (i.e. is
      the gradient direction it points along useful)?

  (4) EFFECTIVE 2-LEVEL rotation / commutator.  Projected onto the
      {|ref>, |double>} 2-level subspace each correlator is an SO(2) rotation
      exp(theta * [[0,-1],[1,0]]) -- a genuine Rabi rotation of amplitude --
      whereas the CP is exp(i phi * diag) -- a relative phase, no amplitude
      transfer.  We print both 2x2 blocks and the commutator norm [G,H_toy].

Outputs: report/figures/imucj_math.png and a console table.
"""

from __future__ import annotations
import sys
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from vib_sqd.circuits.ansatz_enhanced import (
    UVCCSDAnsatz,
    append_givens_rotation_native,
)

FIG = _HERE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=4, suppress=True, linewidth=140)


# --------------------------------------------------------------------------- #
# One-hot bookkeeping for a small n_modes x n_modals system
# --------------------------------------------------------------------------- #
def onehot_index(occ, n_modals):
    """Big-endian qubit index of the one-hot state for occupation tuple `occ`."""
    bits = []
    for v in occ:
        blk = [0] * n_modals
        blk[v] = 1
        bits.extend(blk)
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


def physical_subspace(n_modes, n_modals):
    """List of (occ tuple, big-endian statevector index) for all one-hot configs."""
    occs = list(product(range(n_modals), repeat=n_modes))
    return [(o, onehot_index(o, n_modals)) for o in occs]


def op_matrix_reversed(qc):
    """Operator matrix in Qiskit little-endian, then reverse to big-endian so it
    matches our onehot_index convention (Statevector(...).reverse_qargs())."""
    U = Operator(qc).data
    n = qc.num_qubits
    # reverse qubit order: permute basis indices by bit-reversal
    perm = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)])
    return U[np.ix_(perm, perm)]


def restrict(U_full, phys):
    """Restrict a full 2^nq x 2^nq operator to the physical one-hot subspace."""
    idx = [i for _, i in phys]
    return U_full[np.ix_(idx, idx)]


# --------------------------------------------------------------------------- #
# Build the two elementary correlators as circuits on a 2-mode/2-modal system
# --------------------------------------------------------------------------- #
# 2 modes x 2 modals = 4 qubits. mode0 -> qubits [0,1], mode1 -> qubits [2,3].
# Reference |ref> = |10 10> : modal 0 occupied in both modes.
# The one relevant double is |01 01> : modal 1 occupied in both modes.
N_MODES, N_MODALS = 2, 2
NQ = N_MODES * N_MODALS
MQ = [[m * N_MODALS + a for a in range(N_MODALS)] for m in range(N_MODES)]
PHYS = physical_subspace(N_MODES, N_MODALS)
PHYS_LABELS = ["".join(map(str, o)) for o, _ in PHYS]  # e.g. '00','01','10','11'
REF_OCC = (0, 0)  # -> label '00'
DBL_OCC = (1, 1)  # -> label '11'


def imucj_correlator_circuit(theta):
    """Single VIm-uCJ paired correlator: the UVCCSD double-excitation primitive
    coupling (mode0 modal0<->modal1, mode1 modal0<->modal1)."""
    qc = QuantumCircuit(NQ)
    uv = UVCCSDAnsatz("direct")
    uv._apply_double_excitation(
        qc,
        q_ref1=MQ[0][0],
        q_exc1=MQ[0][1],
        q_ref2=MQ[1][0],
        q_exc2=MQ[1][1],
        theta=theta,
    )
    return qc


def vlucj_cp_circuit(phi):
    """VLUCJ same-modal diagonal CP Jastrow between the two modes at modal 0
    (the reference-occupied qubits) -- the inert correlator."""
    qc = QuantumCircuit(NQ)
    qc.cp(phi, MQ[0][0], MQ[1][0])
    return qc


def givens_circuit(theta):
    """Intramode Givens on mode0 (modal0<->modal1) -- the separable single-mode
    rotation VLUCJ relies on for all its amplitude."""
    qc = QuantumCircuit(NQ)
    append_givens_rotation_native(qc, q_ref=MQ[0][0], q_exc=MQ[0][1], theta=theta)
    return qc


# --------------------------------------------------------------------------- #
# (1) Generator extraction:  G = logm(U(theta)) / theta   (small theta, stable)
# --------------------------------------------------------------------------- #
def extract_generator(circuit_fn, eps=1e-4):
    """Numerically extract the generator G with U(theta) ~ exp(theta G) via a
    finite difference: G ~ (U(eps) - U(-eps)) / (2 eps) evaluated at identity."""
    Up = restrict(op_matrix_reversed(circuit_fn(eps)), PHYS)
    Um = restrict(op_matrix_reversed(circuit_fn(-eps)), PHYS)
    return (Up - Um) / (2 * eps)


def analyze_generators():
    Gi = extract_generator(imucj_correlator_circuit)
    Gc = extract_generator(vlucj_cp_circuit)
    print(
        "\n=== (1) GENERATOR matrices on the physical one-hot subspace "
        f"(basis {PHYS_LABELS}) ==="
    )
    print(
        "VIm-uCJ paired correlator generator G_im  (real anti-symmetric => "
        "rotates amplitude):"
    )
    print(np.real_if_close(Gi, tol=1e6))
    print(
        f"  ||G_im - (-G_im^T)|| (anti-symmetry defect) = "
        f"{np.linalg.norm(Gi + Gi.T):.2e}"
    )
    print(
        f"  ||diag(G_im)|| = {np.linalg.norm(np.diag(Gi)):.2e}   "
        f"||offdiag(G_im)|| = "
        f"{np.linalg.norm(Gi - np.diag(np.diag(Gi))):.4f}"
    )
    print(
        "\nVLUCJ same-modal CP Jastrow generator G_cp  (diagonal, imaginary => "
        "pure phase):"
    )
    print(Gc)
    print(
        f"  ||offdiag(G_cp)|| = "
        f"{np.linalg.norm(Gc - np.diag(np.diag(Gc))):.2e}   "
        f"diag(G_cp) = {np.diag(Gc)}"
    )
    return Gi, Gc


# --------------------------------------------------------------------------- #
# (2) Reference -> config transfer map:  |<config|U(theta)|ref>|^2 vs theta
# --------------------------------------------------------------------------- #
def transfer_map(circuit_fn, thetas):
    ref_i = PHYS.index((REF_OCC, onehot_index(REF_OCC, N_MODALS)))
    weights = np.zeros((len(thetas), len(PHYS)))
    for t, th in enumerate(thetas):
        U = restrict(op_matrix_reversed(circuit_fn(th)), PHYS)
        psi = U[:, ref_i]
        weights[t] = np.abs(psi) ** 2
    return weights


# --------------------------------------------------------------------------- #
# (4) Effective 2-level rotation + commutator with a toy Hamiltonian
# --------------------------------------------------------------------------- #
def two_level_blocks(Gi, Gc):
    """Project both generators onto the {|ref>=00, |double>=11} 2-level space."""
    i_ref = PHYS_LABELS.index("00")
    i_dbl = PHYS_LABELS.index("11")
    sel = [i_ref, i_dbl]
    return Gi[np.ix_(sel, sel)], Gc[np.ix_(sel, sel)], (i_ref, i_dbl)


def commutator_norms(Gi, Gc):
    """[G, H_toy] for a toy diagonal H (site energies) -- nonzero commutator =>
    the correlator changes the energy expectation (does work); zero => inert."""
    # toy H: distinct energy on each physical config
    H = np.diag(np.arange(len(PHYS), dtype=float))
    ci = np.linalg.norm(Gi @ H - H @ Gi)
    cc = np.linalg.norm(Gc @ H - H @ Gc)
    return ci, cc


# --------------------------------------------------------------------------- #
# (3) t2 -> theta initialization mapper, tested on glycine-4
# --------------------------------------------------------------------------- #
def t2_to_theta_mapper(t2, chosen, scale=1.0, clip=np.pi / 2):
    """Map VCC t2 doubles amplitudes to VIm-uCJ correlator angles (warm start).
    theta_pair = clip(scale * t2[i,j,a,b]).  Returns an array aligned with
    `chosen` = the ranked (i,j,a,b) pairs used by build_vim_ucj."""
    th = np.array([t2[i, j, a, b] for (i, j, a, b) in chosen], dtype=float)
    return np.clip(scale * th, -clip, clip)


def glycine_warmstart_test(maxiter=200):
    """On glycine-4 modal basis: compare zero-init vs t2-mapped init for the
    VIm-uCJ correlator angles -- report initial energy and converged gap."""
    import warnings

    warnings.filterwarnings("ignore")
    import vib_sqd.basis.modal_transform as mb
    from vib_sqd.classical.mop_forcefield import parse_mop, mop_to_forcefield, CM2AU
    from vib_sqd.classical.vcc_extractor import extract_amplitudes
    from vib_sqd.classical.vcc_solver import VCCAmplitudes
    from vib_sqd.circuits.initial_states import build_vscf_reference_state
    from vib_sqd.run.dualbasis import ensure_clean_vcc, _ordered, _assign
    from vib_sqd.circuits.ucj_correlators import build_vim_ucj, rank_doubles_by_t2
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize

    n_modes, n_modals, n_ho = 4, 3, 9
    mout, ffmop = ensure_clean_vcc(
        "glycine4",
        _HERE.parents[1] / "data" / "mop_files" / "glycinehplus.mop",
        (24, 25, 26, 27),
        n_modals,
        str(_HERE / "runs/vlucj_levers"),
    )
    ff = mop_to_forcefield(parse_mop(ffmop))
    rot = mb.build_modal_rotation(mout, n_modes, n_ho, n_modals)
    t1m, t2m, _ = extract_amplitudes(Path(mout), n_modes, n_modals)
    amps = VCCAmplitudes(t1=t1m, t2=t2m, converged=True)
    Hc = mb.modal_hamiltonian_compact(mb.ho_hamiltonian_full(ff, n_ho), rot)
    w, V = np.linalg.eigh(Hc)
    exact = w[0].real / CM2AU
    c0 = V[:, 0] / np.linalg.norm(V[:, 0])

    prod, occs = [], list(product(range(n_modals), repeat=n_modes))
    prod = np.array([onehot_index(o, n_modals) for o in occs])
    ref = build_vscf_reference_state(n_modes, n_modals, 0)
    circ = build_vim_ucj(ref, n_modes, n_modals, layers=2, n_pairs=4, t2=amps.t2)
    params = _ordered(circ)
    n = len(params)

    def E(x):
        a = np.asarray(Statevector(_assign(circ, x)).reverse_qargs().data)[prod]
        a = a / np.linalg.norm(a)
        return float(np.real(np.vdot(a, Hc @ a))) / CM2AU

    # zero init
    x0_zero = np.zeros(n)
    # t2-mapped init: fill the correlator angles (the "im" pair params) from t2;
    # build_vim_ucj interleaves [intramode | pairs | rz] per layer -- locate the
    # pair slots by name-order. We know per-layer layout sizes:
    chosen = rank_doubles_by_t2(amps.t2)[:4]
    n_intra = n_modes * (n_modals - 1)
    n_pairs = len(chosen)
    n_rz = n_modes * n_modals
    per_layer = n_intra + n_pairs + n_rz
    theta_pairs = t2_to_theta_mapper(amps.t2, chosen, scale=1.0)
    x0_map = np.zeros(n)
    for L in range(2):
        base = L * per_layer + n_intra
        x0_map[base : base + n_pairs] = theta_pairs

    e_zero0, e_map0 = E(x0_zero), E(x0_map)
    r_zero = minimize(
        E, x0_zero, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.1}
    )
    r_map = minimize(
        E, x0_map, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.1}
    )
    return {
        "exact": exact,
        "chosen": chosen,
        "theta_pairs": theta_pairs,
        "E0_zero": e_zero0 - exact,
        "E0_map": e_map0 - exact,
        "gap_zero": r_zero.fun - exact,
        "gap_map": r_map.fun - exact,
    }


def main():
    Gi, Gc = analyze_generators()

    blk_i, blk_c, _ = two_level_blocks(Gi, Gc)
    print("\n=== (4) Effective 2-level {|ref>=00, |double>=11} blocks ===")
    print("VIm-uCJ 2-level generator (SO(2) rotation -> transfers amplitude):")
    print(np.real(blk_i))
    print("VLUCJ CP 2-level generator (diagonal phase -> no amplitude transfer):")
    print(blk_c)
    ci, cc = commutator_norms(Gi, Gc)
    print(
        f"\ncommutator with toy diagonal H:  ||[G_im, H]|| = {ci:.4f}   "
        f"||[G_cp, H]|| = {cc:.4f}"
    )
    print(
        "  (nonzero => the correlator can change <H>; the CP's small/zero "
        "value confirms it does no work on the reference)"
    )

    thetas = np.linspace(-np.pi, np.pi, 121)
    w_im = transfer_map(imucj_correlator_circuit, thetas)
    w_cp = transfer_map(vlucj_cp_circuit, thetas)
    w_gv = transfer_map(givens_circuit, thetas)
    print(
        "\n=== (2) transfer map summary (max weight moved onto the double "
        "'11' from the reference) ==="
    )
    dbl = PHYS_LABELS.index("11")
    print(f"  VIm-uCJ  max |<11|U|00>|^2 = {w_im[:, dbl].max():.4f}")
    print(f"  VLUCJ   max |<11|U|00>|^2 = {w_cp[:, dbl].max():.4e}  (inert)")
    print(
        f"  Givens  max |<11|U|00>|^2 = {w_gv[:, dbl].max():.4e}  "
        f"(single-mode: cannot reach a double)"
    )

    print("\n=== (3) t2 -> theta warm-start mapper on glycine-4 (modal) ===")
    try:
        ws = glycine_warmstart_test()
        print(f"  exact = {ws['exact']:.2f} cm-1")
        print(f"  chosen pairs (i,j,a,b) = {ws['chosen']}")
        print(f"  mapped angles theta   = {np.round(ws['theta_pairs'], 4)}")
        print(
            f"  initial gap  zero-init = {ws['E0_zero']:+8.2f} cm-1   "
            f"t2-mapped = {ws['E0_map']:+8.2f} cm-1"
        )
        print(
            f"  converged gap zero-init= {ws['gap_zero']:+8.2f} cm-1   "
            f"t2-mapped = {ws['gap_map']:+8.2f} cm-1"
        )
        better = "YES" if ws["gap_map"] <= ws["gap_zero"] + 1e-6 else "no"
        print(f"  t2 warm-start >= zero-init at convergence? {better}")
    except Exception as e:  # keep the matrix analysis usable even if MidasCpp absent
        print(f"  [skipped glycine warm-start: {e}]")

    from vib_sqd.analysis.plots import plot_vimucj_math

    plot_vimucj_math(
        Gi,
        Gc,
        thetas,
        w_im,
        w_cp,
        w_gv,
        blk_i,
        blk_c,
        PHYS_LABELS,
        FIG / "imucj_math.png",
    )
    print(f"\nwrote {FIG / 'imucj_math.png'}")


if __name__ == "__main__":
    main()
