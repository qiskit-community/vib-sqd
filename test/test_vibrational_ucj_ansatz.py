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

"""Ansatz correctness for the vibrational UCJ correlators (VIm-uCJ / Vg-uCJ) and
the VLUCJ / CHC baselines.

Checks, on tiny systems (statevector, no MidasCpp):
  * builders produce circuits with the documented parameter counts;
  * every ansatz is NUMBER-CONSERVING: it maps the one-hot physical subspace to
    itself (negligible leakage);
  * VIm-uCJ MOVES amplitude onto a doubly-excited config, while the VLUCJ
    same-modal diagonal CP is INERT on the one-hot reference (only phases);
  * the vibrational names and their backward-compat aliases are the same object.
"""

from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from vib_sqd.circuits import ucj_correlators as uc
from vib_sqd.circuits.initial_states import build_vscf_reference_state


def _onehot_index_big(occ, n_modals):
    bits = []
    for v in occ:
        blk = [0] * n_modals
        blk[v] = 1
        bits.extend(blk)
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


def _physical_indices(n_modes, n_modals):
    occs = list(product(range(n_modals), repeat=n_modes))
    return occs, np.array([_onehot_index_big(o, n_modals) for o in occs])


def _bind_zeros_then(circ, overrides=None):
    """Bind all parameters to 0 unless overridden (name->value)."""
    params = sorted(circ.parameters, key=lambda p: (p.vector.name, p.index))
    vals = {p: 0.0 for p in params}
    if overrides:
        for p in params:
            key = (p.vector.name, p.index)
            if key in overrides:
                vals[p] = overrides[key]
    return circ.assign_parameters(vals)


def test_aliases_are_same_object():
    assert uc.build_im_ucj is uc.build_vim_ucj
    assert uc.build_g_ucj is uc.build_vg_ucj
    assert uc.build_k_ucj is uc.build_vk_ucj


@pytest.mark.parametrize("n_modes,n_modals", [(2, 2), (3, 3), (2, 3)])
def test_vim_vg_parameter_counts(n_modes, n_modals):
    ref = build_vscf_reference_state(n_modes, n_modals, 0)
    t2 = np.zeros((n_modes, n_modes, n_modals - 1, n_modals - 1))
    layers, n_pairs = 2, min(4, n_modes * (n_modes - 1) // 2 * (n_modals - 1) ** 2)
    n_pairs = max(n_pairs, 1)
    vim = uc.build_vim_ucj(
        ref, n_modes, n_modals, layers=layers, n_pairs=n_pairs, t2=t2
    )
    vg = uc.build_vg_ucj(ref, n_modes, n_modals, layers=layers, n_pairs=n_pairs, t2=t2)
    n_intra = n_modes * (n_modals - 1)
    n_rz = n_modes * n_modals
    # VIm-uCJ per layer: intramode + n_pairs + rz ; Vg-uCJ adds a phase per pair
    n_chosen = len(uc.rank_doubles_by_t2(t2)[:n_pairs])
    assert len(vim.parameters) == layers * (n_intra + n_chosen + n_rz)
    assert len(vg.parameters) == layers * (n_intra + 2 * n_chosen + n_rz)
    # the init-vector length must match the circuit param count exactly
    assert len(uc.imucj_init_vector(t2, n_modes, n_modals, layers, n_pairs)) == len(
        vim.parameters
    )
    assert len(uc.gucj_init_vector(t2, n_modes, n_modals, layers, n_pairs)) == len(
        vg.parameters
    )


@pytest.mark.parametrize("builder", ["vim", "vg", "vlucj_plus", "chc_plus"])
def test_number_conserving(builder):
    """Optimized/nonzero-parameter state stays inside the one-hot subspace."""
    n_modes, n_modals = 2, 3
    ref = build_vscf_reference_state(n_modes, n_modals, 0)
    t2 = np.zeros((n_modes, n_modes, n_modals - 1, n_modals - 1))
    t2[0, 1, 0, 0] = 0.3
    if builder == "vim":
        circ = uc.build_vim_ucj(ref, n_modes, n_modals, layers=2, n_pairs=2, t2=t2)
    elif builder == "vg":
        circ = uc.build_vg_ucj(ref, n_modes, n_modals, layers=2, n_pairs=2, t2=t2)
    elif builder == "vlucj_plus":
        circ = uc.build_vlucj_plus_doubles(
            ref, n_modes, n_modals, layers=2, n_doubles=2, t2=t2
        )
    else:
        circ = uc.build_chc_plus_doubles(
            ref, n_modes, n_modals, layers=2, n_doubles=2, t2=t2
        )
    params = sorted(circ.parameters, key=lambda p: (p.vector.name, p.index))
    bound = circ.assign_parameters({p: 0.37 for p in params})
    sv = np.asarray(Statevector(bound).reverse_qargs().data)
    _, prod = _physical_indices(n_modes, n_modals)
    phys_weight = float(np.sum(np.abs(sv[prod]) ** 2))
    assert phys_weight == pytest.approx(1.0, abs=1e-9)


def test_vim_moves_amplitude_but_vlucj_cp_is_inert():
    """VIm-uCJ correlator populates the double; the VLUCJ same-modal CP does not.

    Both act on the physical one-hot REFERENCE state (modal 0 occupied in every
    mode), which is what the ansatz actually starts from.
    """
    from vib_sqd.circuits.ansatz_enhanced import UVCCSDAnsatz

    n_modes, n_modals = 2, 2
    occs, prod = _physical_indices(n_modes, n_modals)
    ref_i = occs.index((0, 0))
    dbl_i = occs.index((1, 1))
    mq = [[m * n_modals + a for a in range(n_modals)] for m in range(n_modes)]

    # VIm-uCJ elementary correlator = UVCCSD paired double-excitation on the reference
    qc = build_vscf_reference_state(n_modes, n_modals, 0)
    UVCCSDAnsatz("direct")._apply_double_excitation(
        qc,
        q_ref1=mq[0][0],
        q_exc1=mq[0][1],
        q_ref2=mq[1][0],
        q_exc2=mq[1][1],
        theta=0.6,
    )
    sv = np.asarray(Statevector(qc).reverse_qargs().data)[prod]
    assert abs(sv[dbl_i]) ** 2 > 0.05  # amplitude created on the double
    assert abs(sv[ref_i]) ** 2 < 0.999  # reference depleted

    # VLUCJ same-modal diagonal CP on the reference qubits -> only a phase: the
    # population stays entirely on the reference config (inert).
    r = build_vscf_reference_state(n_modes, n_modals, 0)
    r.cp(0.6, mq[0][0], mq[1][0])
    p = np.abs(np.asarray(Statevector(r).reverse_qargs().data)[prod]) ** 2
    assert p[ref_i] == pytest.approx(1.0, abs=1e-12)
    assert p[dbl_i] == pytest.approx(0.0, abs=1e-12)


def test_vg_reaches_complex_double_amplitude():
    """Vg-uCJ's phase part gives the double a nonzero imaginary amplitude that
    VIm-uCJ (real correlator) cannot reach. Acts on the reference state."""
    from vib_sqd.circuits.ansatz_enhanced import UVCCSDAnsatz

    n_modes, n_modals = 2, 2
    occs, prod = _physical_indices(n_modes, n_modals)
    dbl_i = occs.index((1, 1))
    mq = [[m * n_modals + a for a in range(n_modals)] for m in range(n_modes)]

    # VIm-uCJ (real correlator): double amplitude is real
    qc_im = build_vscf_reference_state(n_modes, n_modals, 0)
    UVCCSDAnsatz("direct")._apply_double_excitation(
        qc_im,
        q_ref1=mq[0][0],
        q_exc1=mq[0][1],
        q_ref2=mq[1][0],
        q_exc2=mq[1][1],
        theta=0.8,
    )
    c_im = np.asarray(Statevector(qc_im).reverse_qargs().data)[prod][dbl_i]
    assert abs(c_im.imag) < 1e-9

    # Vg-uCJ: same rotation THEN an off-diagonal phase on the excited modals ->
    # the (populated) double picks up a genuine phase -> nonzero imaginary part
    qc_g = build_vscf_reference_state(n_modes, n_modals, 0)
    UVCCSDAnsatz("direct")._apply_double_excitation(
        qc_g,
        q_ref1=mq[0][0],
        q_exc1=mq[0][1],
        q_ref2=mq[1][0],
        q_exc2=mq[1][1],
        theta=0.8,
    )
    qc_g.cp(1.3, mq[0][1], mq[1][1])
    c_g = np.asarray(Statevector(qc_g).reverse_qargs().data)[prod][dbl_i]
    assert abs(c_g.imag) > 1e-3  # genuinely complex, unlike the real-only VIm-uCJ
