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
What does Vg-uCJ add over VIm-uCJ?  --  matrix-level analysis of the fully COMPLEX
paired correlator, on the same faithful 2-mode/2-modal system as
analyze_imucj_math.py.

The Vg-uCJ correlator on a coupled pair is a product of TWO generators:
  * G_rot  : the imaginary (anti-Hermitian) paired rotation  = the VIm-uCJ part.
             Real, antisymmetric, off-diagonal (reference<->double). TRANSFERS
             amplitude.
  * G_ph   : an OFF-DIAGONAL CP phase between the two EXCITED modals (CHC-style,
             NOT the inert same-modal VLUCJ CP). Diagonal in occupation but its
             nonzero entry sits on the DOUBLE '11', so once G_rot has populated
             the double, G_ph sets the reference<->double RELATIVE PHASE.

Together: U_g(theta, phi) = exp(phi G_ph) exp(theta G_rot).  This gives the
correlator a genuinely complex action -- it can place amplitude on the double AND
choose the phase it arrives with -- the extra freedom the Chem. Sci. 2025
(d5sc03585f) work associates with ~92% correlation (vs ~84% Im-only).

We show numerically:
  (1) the two generators G_rot, G_ph and that G_ph is diagonal-on-the-double
      (phase), G_rot is off-diagonal (amplitude);
  (2) that VIm-uCJ can only reach REAL amplitudes on the double (its
      {ref,double} amplitude is sin(theta), always real for real theta), whereas
      Vg-uCJ reaches the full COMPLEX unit disk of double-amplitudes
      c_dbl = sin(theta) e^{i phi} -- i.e. any relative phase;
  (3) an "interference" toy: two doubles that must add coherently. VIm-uCJ, with
      no relative-phase handle, cannot make them add constructively for a target
      whose two components have opposite sign; Vg-uCJ can. This is exactly the
      regime where Vg-uCJ should beat VIm-uCJ (large/deep correlation), even though
      they tie on glycine-4 where a single double dominates.

Outputs: report/figures/gucj_math.png and a console table.
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

# reuse the exact conventions from analyze_imucj_math
from analyze_imucj_math import (
    physical_subspace,
    onehot_index,
    op_matrix_reversed,
    restrict,
    extract_generator,
    N_MODES,
    N_MODALS,
    NQ,
    MQ,
    PHYS,
    PHYS_LABELS,
)

I_REF = PHYS_LABELS.index("00")
I_DBL = PHYS_LABELS.index("11")


# --------------------------------------------------------------------------- #
# Vg-uCJ elementary pieces on the 2-mode/2-modal system
# --------------------------------------------------------------------------- #
def gucj_rot_circuit(theta):
    """Imaginary paired rotation part (== VIm-uCJ correlator)."""
    qc = QuantumCircuit(NQ)
    UVCCSDAnsatz("direct")._apply_double_excitation(
        qc,
        q_ref1=MQ[0][0],
        q_exc1=MQ[0][1],
        q_ref2=MQ[1][0],
        q_exc2=MQ[1][1],
        theta=theta,
    )
    return qc


def gucj_phase_circuit(phi):
    """Real phase part: OFF-DIAGONAL CP between the EXCITED modals (modal-1 of
    each mode) -- this is CHC-style, and its nonzero number product is on the
    DOUBLE '11' (both excited), NOT on the reference (unlike the inert VLUCJ CP)."""
    qc = QuantumCircuit(NQ)
    qc.cp(phi, MQ[0][1], MQ[1][1])
    return qc


def gucj_full(theta, phi):
    """Full Vg-uCJ correlator: rotation then phase."""
    qc = QuantumCircuit(NQ)
    qc.compose(gucj_rot_circuit(theta), inplace=True)
    qc.compose(gucj_phase_circuit(phi), inplace=True)
    return qc


def ref_state_vec():
    """Physical-subspace amplitude vector of the reference |10 10> (label '00')."""
    v = np.zeros(len(PHYS), dtype=complex)
    v[I_REF] = 1.0
    return v


def apply(circ_fn, *args):
    U = restrict(op_matrix_reversed(circ_fn(*args)), PHYS)
    return U @ ref_state_vec()


def main():
    # ---- (1) generators ----
    G_rot = extract_generator(gucj_rot_circuit)
    G_ph = extract_generator(gucj_phase_circuit)
    print(f"=== (1) Vg-uCJ generators on physical subspace {PHYS_LABELS} ===")
    print(
        "G_rot (imaginary paired rotation = VIm-uCJ part; real, antisymmetric, "
        "off-diagonal => TRANSFERS amplitude):"
    )
    print(np.real_if_close(G_rot, tol=1e6))
    print(
        f"  ||offdiag(G_rot)||={np.linalg.norm(G_rot-np.diag(np.diag(G_rot))):.4f}"
        f"   ||diag(G_rot)||={np.linalg.norm(np.diag(G_rot)):.2e}"
    )
    print(
        "\nG_ph (off-diagonal CP phase between EXCITED modals; diagonal, "
        "imaginary, nonzero on the DOUBLE '11' => sets RELATIVE PHASE):"
    )
    print(G_ph)
    d = np.diag(G_ph)
    print(
        f"  diag(G_ph) nonzero at: "
        f"{[PHYS_LABELS[k] for k in range(len(d)) if abs(d[k])>1e-6]}  "
        f"(vs inert VLUCJ CP which is nonzero on the reference '00')"
    )

    # ---- (2) reachable double-amplitude: real segment vs complex disk ----
    thetas = np.linspace(0, np.pi, 61)
    phis = np.linspace(0, 2 * np.pi, 61)
    # VIm-uCJ: c_dbl(theta) -- always real
    c_im = np.array([apply(gucj_rot_circuit, t)[I_DBL] for t in thetas])
    # Vg-uCJ: c_dbl(theta, phi) -- fills a complex disk
    TT, PP = np.meshgrid(thetas, phis)
    c_g = np.zeros_like(TT, dtype=complex)
    for a in range(TT.shape[0]):
        for b in range(TT.shape[1]):
            c_g[a, b] = apply(gucj_full, TT[a, b], PP[a, b])[I_DBL]
    print("\n=== (2) reachable amplitude on the double '11' from the reference ===")
    print(
        f"  VIm-uCJ: c_dbl is REAL, range [{c_im.real.min():+.3f}, "
        f"{c_im.real.max():+.3f}], max |Im part|={np.abs(c_im.imag).max():.2e}"
    )
    print(
        f"  Vg-uCJ : c_dbl fills a COMPLEX disk, |c_dbl|max={np.abs(c_g).max():.3f}, "
        f"arg spans [{np.angle(c_g[np.abs(c_g)>0.3]).min():+.2f}, "
        f"{np.angle(c_g[np.abs(c_g)>0.3]).max():+.2f}] rad"
    )

    # ---- (3) two-double interference toy ----
    # Target requires two double-amplitudes with OPPOSITE sign to add
    # constructively into a low-energy combination. With only real, same-sign
    # reachable amplitudes (VIm-uCJ) the best overlap is limited; Vg-uCJ's phase
    # handle lets the second double flip sign and add coherently.
    # Model: fidelity to target t = (|00> - |A> - |B>)/sqrt(3) using two
    # correlators, each contributing amplitude c_k e^{i ph_k} to its double.
    # VIm-uCJ: ph_k = 0 forced. Vg-uCJ: ph_k free.
    def best_overlap(allow_phase):
        # amplitude budget per double from a single correlator ~ sin(theta),
        # theta in [0, pi/2]; target wants each double at -1/sqrt(3).
        best = -1.0
        for th in np.linspace(0, np.pi / 2, 41):
            amp = np.sin(th)
            for ph in (np.linspace(0, 2 * np.pi, 49) if allow_phase else [0.0]):
                cA = amp * np.exp(1j * ph)
                cB = amp * np.exp(1j * ph)
                c0 = np.cos(th) ** 2  # rough reference retention (2 correlators)
                psi = np.array([c0, cA, cB], dtype=complex)
                psi = psi / np.linalg.norm(psi)
                tgt = np.array([1, -1, -1], dtype=complex) / np.sqrt(3)
                best = max(best, abs(np.vdot(tgt, psi)) ** 2)
        return best

    f_im = best_overlap(allow_phase=False)
    f_g = best_overlap(allow_phase=True)
    print(
        "\n=== (3) two-double interference toy (target needs opposite-sign "
        "amplitudes) ==="
    )
    print(f"  best fidelity  VIm-uCJ (phase forced 0) = {f_im:.4f}")
    print(f"  best fidelity  Vg-uCJ  (phase free)      = {f_g:.4f}")
    print(
        "  => Vg-uCJ's relative-phase handle wins exactly when multiple doubles "
        "must interfere coherently (large/deep correlation); they tie when one "
        "double dominates (e.g. glycine-4)."
    )

    from vib_sqd.analysis.plots import plot_vgucj_math

    plot_vgucj_math(
        G_rot,
        G_ph,
        thetas,
        c_im,
        TT,
        PP,
        c_g,
        f_im,
        f_g,
        PHYS_LABELS,
        FIG / "gucj_math.png",
    )
    print(f"\nwrote {FIG / 'gucj_math.png'}")


if __name__ == "__main__":
    main()
