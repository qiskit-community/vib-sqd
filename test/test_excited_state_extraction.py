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

"""extract_amplitudes_for_occupation against a committed excited-state .mout
fixture (H2O, mode 1 excited to modal 1 via #3 Occup) -- the ground-state
extractor (extract_amplitudes) explicitly refuses this file, since its
Psi_Ref occupation is not all-zero.
"""

from pathlib import Path

import numpy as np
import pytest

from vib_sqd.classical.vcc_extractor import (
    extract_amplitudes,
    extract_amplitudes_for_occupation,
)

FIX = Path(__file__).parent / "fixtures" / "midascpp"
MOUT = FIX / "h2o_excited_vcc.mout"
N_MODES, N_MODALS = 3, 2
OCCUPATION = [0, 1, 0]


def test_ground_state_extractor_refuses_excited_mout():
    with pytest.raises(ValueError):
        extract_amplitudes(MOUT, N_MODES, N_MODALS)


def test_extract_amplitudes_for_occupation_shapes():
    t1, t2, meta = extract_amplitudes_for_occupation(
        MOUT, N_MODES, N_MODALS, OCCUPATION
    )
    n_exc = N_MODALS - 1
    assert t1.shape == (N_MODES, n_exc)
    assert t2.shape == (N_MODES, N_MODES, n_exc, n_exc)
    assert np.isfinite(t1).all() and np.isfinite(t2).all()
    assert meta["occupation"] == OCCUPATION
    assert meta["converged"]


def test_excited_mode_deexcitation_channel_is_exactly_zero():
    """At n_modals=2, mode 1's reference occupation (absolute modal 1) is
    already the top of the truncation; VCC has no de-excitation operator, so
    that mode's only "other" modal (absolute 0, below the reference) must be
    unreachable and its amplitude exactly zero -- not a parsing gap."""
    t1, _t2, _meta = extract_amplitudes_for_occupation(
        MOUT, N_MODES, N_MODALS, OCCUPATION
    )
    assert t1[1, 0] == 0.0


def test_non_excited_modes_have_nonzero_amplitude():
    t1, _t2, _meta = extract_amplitudes_for_occupation(
        MOUT, N_MODES, N_MODALS, OCCUPATION
    )
    assert t1[0, 0] != 0.0
    assert t1[2, 0] != 0.0
