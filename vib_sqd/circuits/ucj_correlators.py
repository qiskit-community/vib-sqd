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
VIBRATIONAL unitary cluster Jastrow correlators (VIm-uCJ / Vg-uCJ) and VLUCJ
extensions.

Naming / lineage: the ELECTRONIC-structure ansaetze "Im-uCJ" (imaginary orbital
rotations) and "g-uCJ" (fully complex orbital rotations) were introduced by
Motta and co-workers for molecular electronic structure --- "Beyond real:
Alternative unitary cluster Jastrow models" (arXiv:2505.10963), building on
Re-uCJ / LUCJ (arXiv:2408.01833) and the ffsim UCJ implementation
(UCJOpSpinBalanced.from_t_amplitudes with interaction_pairs / n_reps /
to_parameters).  The builders here are the VIBRATIONAL analogues on a DIRECT
ONE-HOT MODAL encoding: modes/modals rather than spin-orbitals, seeded from
VCCSD t1/t2 rather than electronic CCSD amplitudes.  They are NOT identical to
the electronic operators, so we name them with a leading "V":

  * build_vim_ucj  (VIm-uCJ)  -- imaginary (anti-Hermitian) paired correlator
  * build_vg_ucj   (Vg-uCJ)   -- fully complex correlator (imaginary rotation +
                                 off-diagonal phase)
  * build_vk_ucj   (Vk-uCJ)   -- k-replica scaffold (NotImplementedError)

Backward-compatible aliases build_im_ucj / build_g_ucj / build_k_ucj are kept.

Our finding: the diagonal controlled-phase (CP) Jastrow in VLUCJ is INERT on the
one-hot reference (it only imprints phases; it cannot create excited amplitude),
so VLUCJ reaches high fidelity on the dominant config but MISSES the small-weight
correlation tail -> its energy plateaus ~20 cm-1 above exact while UVCCSD reaches
~0.1 cm-1.  The fix is an amplitude-creating operator ALONGSIDE / INSTEAD of the
diagonal Jastrow.

Runnable:
  * build_vlucj_plus_doubles: VLUCJ layers + a few genuine paired
    double-excitation gates seeded from the largest |t2|.
    Ref: AGP / paired-CC + Jastrow, JPCA 2023 10.1021/acs.jpca.3c00525.
  * build_ucj_sandwich: paired-mode UCJ  U J U^dagger  with a CROSS-mode
    conjugating rotation (off-diagonal -> creates paired amplitude).
    Ref: Motta et al. JPCA 2024 arXiv:2408.01833.
  * build_vim_ucj (VIm-uCJ): IMAGINARY (anti-Hermitian) paired correlator ---
    rotates paired occupations, reaching doubly-excited configs directly (in the
    electronic study Re-uCJ ~60%, Im-uCJ ~84%). Ref: arXiv:2505.10963.
  * build_vg_ucj (Vg-uCJ): fully COMPLEX correlator = imaginary paired rotation
    + an OFF-DIAGONAL (CHC-style, non-inert) phase between the excited modals ---
    the extra phase freedom (~92% in the electronic study, near-UCCSD) sets the
    reference<->double relative phase. Ref: arXiv:2505.10963.
  * build_vk_ucj (Vk-uCJ): k Jastrow replicas with a Takagi / double-
    factorization warm start; scaffold (Ref JCTC 2025 5c01443).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from vib_sqd.circuits.ansatz_enhanced import (
    EnhancedVUCJAnsatz,
    UVCCSDAnsatz,
    append_givens_rotation_native,
)
from vib_sqd.circuits.chc import build_chc_ansatz


def t2_to_angle(t2_val: float, scale: float = 1.0) -> float:
    """Bounded/perturbative map from a VCC/CCSD cluster amplitude to a
    correlator ROTATION angle:  theta = arctan(scale * t2).

    Why not a raw copy:  t2 are cluster-operator coefficients calibrated for the
    CC exponential exp(T2); the Im-uCJ/g-uCJ correlator angle is the argument of
    a trigonometric SO(2) rotation whose amplitude saturates and wraps at
    theta ~ pi/2.  A raw copy overshoots (measured: it HURTS -- converges to
    +3.7 cm-1 vs +0.15 for zero-init on glycine-4).  arctan is monotonic, agrees
    with the raw copy to first order (arctan(x) ~ x for small x), and smoothly
    saturates to +-pi/2 so large amplitudes cannot overshoot or wrap.  This is
    the "CCSD warm start done right" for a trig-parameterized correlator.
    """
    return float(np.arctan(scale * t2_val))


def imucj_init_vector(
    t2: np.ndarray,
    n_modes: int,
    n_modals: int,
    layers: int,
    n_pairs: int,
    scale: float = 1.0,
    include_rz: bool = True,
) -> np.ndarray:
    """Warm-start x0 for build_im_ucj: fill the paired-correlator angles from t2
    via the bounded arctan map; leave intramode Givens and RZ at zero.  Layout
    per layer is [intramode | pairs | rz], repeated `layers` times."""
    chosen = rank_doubles_by_t2(t2)[:n_pairs]
    n_intra = n_modes * (n_modals - 1)
    n_rz = n_modes * n_modals if include_rz else 0
    per_layer = n_intra + len(chosen) + n_rz
    x0 = np.zeros(layers * per_layer)
    ang = np.array([t2_to_angle(t2[i, j, a, b], scale) for (i, j, a, b) in chosen])
    for L in range(layers):
        base = L * per_layer + n_intra
        x0[base : base + len(chosen)] = ang
    return x0


def gucj_init_vector(
    t2: np.ndarray,
    n_modes: int,
    n_modals: int,
    layers: int,
    n_pairs: int,
    scale: float = 1.0,
    include_rz: bool = True,
) -> np.ndarray:
    """Warm-start x0 for build_g_ucj: fill the ROTATION angle of each pair from
    t2 (arctan map); leave the PHASE angle, intramode Givens and RZ at zero.
    Layout per layer is [intramode | (rot, phase) x n_pairs | rz]."""
    chosen = rank_doubles_by_t2(t2)[:n_pairs]
    n_intra = n_modes * (n_modals - 1)
    n_rz = n_modes * n_modals if include_rz else 0
    per_layer = n_intra + 2 * len(chosen) + n_rz
    x0 = np.zeros(layers * per_layer)
    ang = [t2_to_angle(t2[i, j, a, b], scale) for (i, j, a, b) in chosen]
    for L in range(layers):
        base = L * per_layer + n_intra
        for k in range(len(chosen)):
            x0[base + 2 * k] = ang[k]  # rotation slot; phase slot stays 0
    return x0


def rank_doubles_by_t2(t2: np.ndarray) -> list[tuple[int, int, int, int]]:
    """All (mode1<mode2, exc1, exc2) double indices, sorted by descending |t2|."""
    n_modes, _, n_exc, _ = t2.shape
    idx = [
        (i, j, a, b)
        for i in range(n_modes)
        for j in range(i + 1, n_modes)
        for a in range(n_exc)
        for b in range(n_exc)
    ]
    return sorted(idx, key=lambda d: abs(t2[d]), reverse=True)


def build_vlucj_plus_doubles(
    initial_state: QuantumCircuit,
    n_modes: int,
    n_modals: int,
    layers: int = 2,
    n_doubles: int = 4,
    t2: np.ndarray | None = None,
    adjacent_jastrow: bool = True,
    include_rz: bool = True,
) -> QuantumCircuit:
    """VLUCJ (Givens + diagonal Jastrow + RZ) augmented with ``n_doubles`` genuine
    paired double-excitation gates on the largest-|t2| mode/modal pairs.
    """
    dummy = SparsePauliOp(["I" * (n_modes * n_modals)], np.array([0.0]))
    base = EnhancedVUCJAnsatz("direct").build(
        dummy,
        initial_state,
        {
            "n_modes": n_modes,
            "n_modals": n_modals,
            "layers": layers,
            "measure": False,
            "include_rz": include_rz,
            "adjacent_jastrow": adjacent_jastrow,
        },
    )[0]
    qc = base.copy()

    if t2 is not None:
        ranked = rank_doubles_by_t2(t2)
    else:
        n_exc = n_modals - 1
        ranked = [
            (i, j, a, b)
            for i in range(n_modes)
            for j in range(i + 1, n_modes)
            for a in range(n_exc)
            for b in range(n_exc)
        ]
    chosen = ranked[:n_doubles]

    uv = UVCCSDAnsatz("direct")
    mq = [[m * n_modals + a for a in range(n_modals)] for m in range(n_modes)]
    dbl = ParameterVector("d", len(chosen))
    for p, (i, j, a, b) in enumerate(chosen):
        uv._apply_double_excitation(
            qc,
            q_ref1=mq[i][0],
            q_exc1=mq[i][a + 1],
            q_ref2=mq[j][0],
            q_exc2=mq[j][b + 1],
            theta=dbl[p],
        )
    return qc


def _append_paired_doubles(qc, n_modes, n_modals, n_doubles, t2, pv_name="d"):
    """Append ``n_doubles`` genuine paired double-excitation gates (largest |t2|)
    to an existing parameterized circuit ``qc`` in place, returning qc."""
    if t2 is not None:
        ranked = rank_doubles_by_t2(t2)
    else:
        n_exc = n_modals - 1
        ranked = [
            (i, j, a, b)
            for i in range(n_modes)
            for j in range(i + 1, n_modes)
            for a in range(n_exc)
            for b in range(n_exc)
        ]
    chosen = ranked[:n_doubles]
    uv = UVCCSDAnsatz("direct")
    mq = [[m * n_modals + a for a in range(n_modals)] for m in range(n_modes)]
    dbl = ParameterVector(pv_name, len(chosen))
    for p, (i, j, a, b) in enumerate(chosen):
        uv._apply_double_excitation(
            qc,
            q_ref1=mq[i][0],
            q_exc1=mq[i][a + 1],
            q_ref2=mq[j][0],
            q_exc2=mq[j][b + 1],
            theta=dbl[p],
        )
    return qc


def build_chc_plus_doubles(
    initial_state: QuantumCircuit,
    n_modes: int,
    n_modals: int,
    layers: int = 2,
    n_doubles: int = 4,
    t2: np.ndarray | None = None,
) -> QuantumCircuit:
    """CHC ansatz augmented with ``n_doubles`` genuine paired double-excitation
    gates (largest |t2|) — the CHC analogue of build_vlucj_plus_doubles, to test
    whether the doubles lever is ansatz-specific or a general fix."""
    qc = build_chc_ansatz(initial_state, n_modes, n_modals, layers=layers).copy()
    return _append_paired_doubles(qc, n_modes, n_modals, n_doubles, t2)


# --------------------------------------------------------------------------- #
# Math fix (1): genuine UCJ correlator  U J U^dagger  (paired-mode)            #
# --------------------------------------------------------------------------- #
#
# WHY the plain VLUCJ Jastrow is inert (and CHC is not):
#   VLUCJ's CP-Jastrow couples the SAME modal index across two modes,
#   including modal 0:  cp(theta, mode[u][m], mode[l][m]).  On the one-hot
#   reference |...100...100...> the number product n_i n_j is 1 only for the
#   modal-0 pair (a global phase) and 0 for every excited modal (identity).
#   The conjugating Givens are INTRAMODE (block-diagonal per mode), so they do
#   not turn that diagonal phase into a cross-mode off-diagonal operator.  Net:
#   pure phase, zero paired amplitude.
#
#   Genuine UCJ is  exp(i U J U^dagger),  J = 1/2 sum J_ij n_i n_j, where U is a
#   NON-trivial orbital rotation SPANNING the coupled orbitals.  Then
#   U e^{iJ} U^dagger is off-diagonal and creates paired amplitude.  The fix
#   below supplies exactly that: for each coupled mode pair (i,j) and excited
#   modal a, we (a) rotate the reference occupation of mode i onto its excited
#   modal a AND mode j onto its excited modal b with cross-mode Givens (this is
#   the U spanning both modes), (b) apply the diagonal number-phase there, then
#   (c) rotate back with U^dagger.  The result is a real two-mode entangling
#   correlator seeded on the largest |t2| pairs -- the amplitude-creating pair
#   correlation the diagonal Jastrow lacks, expressed as a Jastrow (not as bolt-
#   on UVCCSD doubles).  Ref: LUCJ/UCJ, Motta et al. JPCA 2024 arXiv:2408.01833.


def _cross_mode_givens(qc, qa, qb, theta):
    """Real Givens rotation exp(theta (a_b^dag a_a - a_a^dag a_b)) between two
    (cross-mode) modal qubits -- the U that spans both modes in U J U^dagger."""
    append_givens_rotation_native(qc, q_ref=qa, q_exc=qb, theta=theta)


def build_ucj_sandwich(
    initial_state: QuantumCircuit,
    n_modes: int,
    n_modals: int,
    layers: int = 2,
    n_pairs: int = 4,
    t2: np.ndarray | None = None,
    include_rz: bool = True,
) -> QuantumCircuit:
    """Genuine paired-mode UCJ:  U J U^dagger  with a CROSS-mode conjugating
    rotation, replacing VLUCJ's inert same-modal diagonal CP.

    For each of the ``n_pairs`` largest-|t2| (mode_i, mode_j, exc_a, exc_b)
    couplings we emit, per layer:
        Givens(mode_i modal0 <-> exc_a)  [U on mode i]
        Givens(mode_j modal0 <-> exc_b)  [U on mode j]
        cp(phi, exc_a, exc_b)            [diagonal number-phase in rotated frame]
        Givens^dagger (both)             [U^dagger]
    which is off-diagonal on the one-hot reference and creates genuine paired
    amplitude on the (exc_a, exc_b) doubly-excited configuration.  Optional
    per-qubit RZ tail (same as VLUCJ).  All parameters fresh ("ucj").
    """
    n_qubits = n_modes * n_modals
    qc = QuantumCircuit(n_qubits)
    qc.compose(initial_state, inplace=True)
    mq = [[m * n_modals + a for a in range(n_modals)] for m in range(n_modes)]

    if t2 is not None:
        ranked = rank_doubles_by_t2(t2)
    else:
        n_exc = n_modals - 1
        ranked = [
            (i, j, a, b)
            for i in range(n_modes)
            for j in range(i + 1, n_modes)
            for a in range(n_exc)
            for b in range(n_exc)
        ]
    chosen = ranked[:n_pairs]

    # 3 params per pair per layer (two rotation angles + one phase) + RZ tail.
    n_rz = n_qubits if include_rz else 0
    pv = ParameterVector("ucj", layers * (3 * len(chosen) + n_rz))
    p = 0
    for _ in range(layers):
        for i, j, a, b in chosen:
            qi0, qia = mq[i][0], mq[i][a + 1]
            qj0, qjb = mq[j][0], mq[j][b + 1]
            _cross_mode_givens(qc, qi0, qia, pv[p])
            p += 1
            _cross_mode_givens(qc, qj0, qjb, pv[p])
            p += 1
            qc.cp(pv[p], qia, qjb)
            p += 1
            # U^dagger: undo the two rotations (Givens is antisymmetric ->
            # inverse is the negative angle; reuse the same params by negating).
            _cross_mode_givens(qc, qj0, qjb, -pv[p - 3])
            _cross_mode_givens(qc, qi0, qia, -pv[p - 3])
        if include_rz:
            for q in range(n_qubits):
                qc.rz(pv[p], q)
                p += 1
    return qc


# --------------------------------------------------------------------------- #
# Math fix (2): VIm-uCJ -- IMAGINARY correlator (reaches doubly-excited configs) #
# --------------------------------------------------------------------------- #
#
# A real diagonal Jastrow e^{iJ} only adds phases.  Im-uCJ replaces the phase
# correlator with an ANTI-Hermitian (imaginary) generator that ROTATES paired
# occupations rather than phasing them -- reaching doubly-excited configs
# directly and recovering far more correlation per 2-qubit gate than Re-uCJ
# (Re ~60%, Im ~84%, complex g-uCJ ~92%; Chem. Sci. 2025 d5sc03585f).
#
# Concretely, for a coupled excited-modal pair (exc_a on mode i, exc_b on
# mode j) we apply the paired double-excitation generator
#     exp( phi * (a_ib_j^dag a_0a_0 - h.c.) )
# built from the UVCCSD 8-term double-excitation primitive (which IS the
# imaginary/off-diagonal paired rotation), interleaved with intramode Givens.
# This is the "make the Jastrow imaginary" fix, distinct from bolt-on doubles:
# here the paired rotation is the correlator layer itself, seeded on |t2|.


def build_vim_ucj(
    initial_state: QuantumCircuit,
    n_modes: int,
    n_modals: int,
    layers: int = 2,
    n_pairs: int = 4,
    t2: np.ndarray | None = None,
    include_rz: bool = True,
) -> QuantumCircuit:
    """VIm-uCJ: vibrational imaginary (anti-Hermitian) paired correlator layers.

    Per layer: an intramode Givens block (the single-particle rotation), then
    ``n_pairs`` imaginary paired-rotation correlators on the largest-|t2|
    (mode_i, mode_j, exc_a, exc_b) couplings (UVCCSD double-excitation
    primitive = the off-diagonal paired rotation), then optional RZ.
    Vibrational analogue of the electronic Im-uCJ (arXiv:2505.10963)."""
    n_qubits = n_modes * n_modals
    qc = QuantumCircuit(n_qubits)
    qc.compose(initial_state, inplace=True)
    mq = [[m * n_modals + a for a in range(n_modals)] for m in range(n_modes)]

    if t2 is not None:
        ranked = rank_doubles_by_t2(t2)
    else:
        n_exc = n_modals - 1
        ranked = [
            (i, j, a, b)
            for i in range(n_modes)
            for j in range(i + 1, n_modes)
            for a in range(n_exc)
            for b in range(n_exc)
        ]
    chosen = ranked[:n_pairs]

    uv = UVCCSDAnsatz("direct")
    n_intra = n_modes * (n_modals - 1)
    n_rz = n_qubits if include_rz else 0
    pv = ParameterVector("im", layers * (n_intra + len(chosen) + n_rz))
    p = 0
    for _ in range(layers):
        # intramode single-particle rotations
        for m in range(n_modes):
            for a in range(n_modals - 1):
                append_givens_rotation_native(
                    qc, q_ref=mq[m][0], q_exc=mq[m][a + 1], theta=pv[p]
                )
                p += 1
        # imaginary paired-rotation correlators (off-diagonal, creates amplitude)
        for i, j, a, b in chosen:
            uv._apply_double_excitation(
                qc,
                q_ref1=mq[i][0],
                q_exc1=mq[i][a + 1],
                q_ref2=mq[j][0],
                q_exc2=mq[j][b + 1],
                theta=pv[p],
            )
            p += 1
        if include_rz:
            for q in range(n_qubits):
                qc.rz(pv[p], q)
                p += 1
    return qc


# --------------------------------------------------------------------------- #
# Math fix (3): Vg-uCJ -- fully COMPLEX correlator (imaginary + real parts)      #
# --------------------------------------------------------------------------- #
#
# Im-uCJ uses a purely IMAGINARY (anti-Hermitian) correlator: it rotates paired
# amplitude but adds no independent relative phase.  g-uCJ makes the correlator
# fully COMPLEX -- an amplitude-transferring (imaginary) part PLUS an
# independent phase (real) part -- which the Chem. Sci. 2025 (d5sc03585f) study
# finds recovers ~92% of the correlation energy (vs ~84% for Im-uCJ, ~60% for
# Re-uCJ), approaching UCCSD.  The extra real/phase freedom lets the correlator
# fix the RELATIVE PHASE between the reference and the double it just populated
# (which pure Im-uCJ leaves fixed at the SO(2) value), so a subsequent layer's
# rotation can add coherently instead of partially cancelling.
#
# Realization here: each paired correlator on (mode_i exc_a, mode_j exc_b) is
#     [imaginary part]  UVCCSD double-excitation primitive   (rotates amplitude)
#     [real part]       CHC-style CP phase between the EXCITED modals exc_a,exc_b
#                       (an OFF-DIAGONAL phase on the one-hot ref -- NOT the inert
#                        same-modal VLUCJ CP; it phases the double relative to
#                        the reference once amplitude is present).
# Two independent parameters per pair per layer -> genuinely complex K-matrix.


def build_vg_ucj(
    initial_state: QuantumCircuit,
    n_modes: int,
    n_modals: int,
    layers: int = 2,
    n_pairs: int = 4,
    t2: np.ndarray | None = None,
    include_rz: bool = True,
) -> QuantumCircuit:
    """Vg-uCJ: vibrational fully complex paired correlator (imaginary rotation +
    real phase). Vibrational analogue of the electronic g-uCJ (arXiv:2505.10963,
    ~92% correlation / near-UCCSD in the electronic study).

    Per layer: intramode Givens block, then for each of the ``n_pairs``
    largest-|t2| couplings a COMPLEX correlator = imaginary paired rotation
    (UVCCSD double-excitation primitive) followed by a CHC-style CP phase between
    the two excited modals (off-diagonal on the one-hot reference), then optional
    RZ. Two params per pair per layer."""
    n_qubits = n_modes * n_modals
    qc = QuantumCircuit(n_qubits)
    qc.compose(initial_state, inplace=True)
    mq = [[m * n_modals + a for a in range(n_modals)] for m in range(n_modes)]

    if t2 is not None:
        ranked = rank_doubles_by_t2(t2)
    else:
        n_exc = n_modals - 1
        ranked = [
            (i, j, a, b)
            for i in range(n_modes)
            for j in range(i + 1, n_modes)
            for a in range(n_exc)
            for b in range(n_exc)
        ]
    chosen = ranked[:n_pairs]

    uv = UVCCSDAnsatz("direct")
    n_intra = n_modes * (n_modals - 1)
    n_rz = n_qubits if include_rz else 0
    # 2 params per pair (imaginary rotation + real phase)
    pv = ParameterVector("g", layers * (n_intra + 2 * len(chosen) + n_rz))
    p = 0
    for _ in range(layers):
        for m in range(n_modes):
            for a in range(n_modals - 1):
                append_givens_rotation_native(
                    qc, q_ref=mq[m][0], q_exc=mq[m][a + 1], theta=pv[p]
                )
                p += 1
        for i, j, a, b in chosen:
            # imaginary part: rotate amplitude onto the double
            uv._apply_double_excitation(
                qc,
                q_ref1=mq[i][0],
                q_exc1=mq[i][a + 1],
                q_ref2=mq[j][0],
                q_exc2=mq[j][b + 1],
                theta=pv[p],
            )
            p += 1
            # real part: OFF-DIAGONAL phase between the EXCITED modals (CHC-style,
            # not the inert same-modal VLUCJ CP)
            qc.cp(pv[p], mq[i][a + 1], mq[j][b + 1])
            p += 1
        if include_rz:
            for q in range(n_qubits):
                qc.rz(pv[p], q)
                p += 1
    return qc


def build_vk_ucj(*args, **kwargs) -> QuantumCircuit:
    """k-uCJ: k replicas of the (non-diagonal/complex) Jastrow with a Takagi /
    double-factorization warm start. Ref: JCTC 2025 10.1021/acs.jctc.5c01443."""
    raise NotImplementedError(
        "k-uCJ is scaffolded (Tier 2 roadmap). Seed k Jastrow factors via Takagi "
        "decomposition per JCTC 2025 10.1021/acs.jctc.5c01443; increase k for "
        "higher (quadruple) excitations."
    )


# Backward-compatible aliases (electronic-style names -> vibrational builders).
build_im_ucj = build_vim_ucj
build_g_ucj = build_vg_ucj
build_k_ucj = build_vk_ucj
