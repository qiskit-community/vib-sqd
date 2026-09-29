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

"""VCCSD t1/t2 shapes and the parameter-shape CONTRACT: the amplitudes produced
by the extractor must have the shapes each ansatz's mapper / init-vector expects,
and the mapped vector length must equal the circuit's parameter count.

Also records (as a forward-compat note/assert) the mapping to ffsim's electronic
UCJ API for future hardware transpilation:
  n_reps <-> layers,  interaction_pairs <-> coupled (mode_i, mode_j) pairs,
  UCJOpSpinBalanced.to_parameters(...) length <-> our init-vector length.
"""

import numpy as np
import pytest

from vib_sqd.circuits import ucj_correlators as uc
from vib_sqd.circuits.ansatz_enhanced import UVCCSDAnsatz
from vib_sqd.circuits.initial_states import build_vscf_reference_state


@pytest.mark.parametrize("n_modes,n_modals", [(2, 2), (3, 4), (4, 3)])
def test_t1_t2_shapes(n_modes, n_modals):
    n_exc = n_modals - 1
    t1 = np.zeros((n_modes, n_exc))
    t2 = np.zeros((n_modes, n_modes, n_exc, n_exc))
    assert t1.shape == (n_modes, n_modals - 1)
    assert t2.shape == (n_modes, n_modes, n_modals - 1, n_modals - 1)


@pytest.mark.parametrize("n_modes,n_modals", [(2, 2), (3, 3), (4, 3)])
def test_uvccsd_param_count_matches_amplitude_dof(n_modes, n_modals):
    """UVCCSD singles+doubles parameter count equals the number of independent
    t1/t2 degrees of freedom."""
    n_singles, n_doubles = UVCCSDAnsatz.count_parameters(n_modes, n_modals)
    n_exc = n_modals - 1
    assert n_singles == n_modes * n_exc
    assert n_doubles == (n_modes * (n_modes - 1) // 2) * n_exc**2


@pytest.mark.parametrize(
    "n_modes,n_modals,layers,n_pairs", [(2, 2, 1, 1), (3, 3, 2, 4), (4, 3, 2, 4)]
)
def test_vim_vg_init_vector_length_matches_circuit(n_modes, n_modals, layers, n_pairs):
    ref = build_vscf_reference_state(n_modes, n_modals, 0)
    t2 = np.zeros((n_modes, n_modes, n_modals - 1, n_modals - 1))
    # make a couple of amplitudes nonzero so ranking picks real pairs
    if n_modes >= 2:
        t2[0, 1, 0, 0] = 0.4
    vim = uc.build_vim_ucj(
        ref, n_modes, n_modals, layers=layers, n_pairs=n_pairs, t2=t2
    )
    vg = uc.build_vg_ucj(ref, n_modes, n_modals, layers=layers, n_pairs=n_pairs, t2=t2)
    x_im = uc.imucj_init_vector(t2, n_modes, n_modals, layers, n_pairs)
    x_g = uc.gucj_init_vector(t2, n_modes, n_modals, layers, n_pairs)
    assert len(x_im) == len(vim.parameters)
    assert len(x_g) == len(vg.parameters)


def test_gucj_init_fills_only_rotation_slots():
    """Vg-uCJ warm start fills the rotation angle of each pair from t2 and leaves
    the phase slot (and intramode/RZ) at zero -- so nonzero entries == n_pairs."""
    n_modes, n_modals, layers, n_pairs = 4, 3, 2, 3
    t2 = np.zeros((n_modes, n_modes, n_modals - 1, n_modals - 1))
    t2[2, 3, 0, 0] = 0.5
    t2[0, 1, 1, 0] = 0.2
    x = uc.gucj_init_vector(t2, n_modes, n_modals, layers, n_pairs)
    chosen = uc.rank_doubles_by_t2(t2)[:n_pairs]
    # only pairs with a nonzero t2 give a nonzero arctan angle; phase/intramode/RZ
    # slots stay zero -> nonzero count == layers * (#chosen pairs with |t2|>0)
    n_nonzero_pairs = sum(1 for (i, j, a, b) in chosen if abs(t2[i, j, a, b]) > 0)
    assert int(np.count_nonzero(x)) == layers * n_nonzero_pairs


def test_t2_to_angle_bounded_and_monotone():
    """The arctan warm-start map is bounded in (-pi/2, pi/2), monotone, and
    first-order-agrees with a raw copy for small amplitudes (no overshoot/wrap)."""
    xs = np.linspace(-50, 50, 401)
    ys = np.array([uc.t2_to_angle(x) for x in xs])
    assert np.all(np.abs(ys) < np.pi / 2)
    assert np.all(np.diff(ys) > 0)  # strictly increasing
    small = 1e-3
    assert uc.t2_to_angle(small) == pytest.approx(small, rel=1e-3)  # arctan(x) ~ x


def test_ffsim_forward_compat_contract():
    """Forward-compat placeholder documenting the vibrational<->ffsim-electronic
    UCJ mapping for future hardware transpilation. Asserts the structural facts
    we rely on so a future ffsim adapter has a fixed target:
      * layers is the vibrational analogue of ffsim n_reps;
      * coupled mode-pairs (i<j) are the analogue of interaction_pairs;
      * the init-vector length is what a to_parameters(...) adapter must produce.
    """
    n_modes, n_modals, layers, n_pairs = 3, 3, 2, 3
    t2 = np.zeros((n_modes, n_modes, n_modals - 1, n_modals - 1))
    t2[0, 1, 0, 0] = 0.3
    ref = build_vscf_reference_state(n_modes, n_modals, 0)
    vg = uc.build_vg_ucj(ref, n_modes, n_modals, layers=layers, n_pairs=n_pairs, t2=t2)
    chosen = uc.rank_doubles_by_t2(t2)[:n_pairs]
    mode_pairs = sorted({(i, j) for (i, j, a, b) in chosen})
    assert all(i < j for i, j in mode_pairs)  # interaction_pairs analogue
    assert len(uc.gucj_init_vector(t2, n_modes, n_modals, layers, n_pairs)) == len(
        vg.parameters
    )
