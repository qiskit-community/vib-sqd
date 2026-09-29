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

"""excited_recovery_loop must select the eigenvector with maximum overlap
on the guaranteed-included excited reference, not simply the lowest
eigenvalue of the sampled subspace.

Taking the lowest eigenvalue is the right choice for the GROUND state (the
variational principle guarantees it), but is silently wrong for an excited
state: once the sampled subspace contains any lower-energy configuration --
which repair starting from a uniform occupancy prior makes near-certain as
soon as there is any sampling noise -- the lowest eigenvalue collapses
toward the GLOBAL ground state's energy regardless of whether the excited
reference was correctly included. This was caught by exactly the
regression scenario below: 99% of raw counts on the exact correct
excited-reference bitstring, 1% uniform noise -- the buggy w[0] selection
recovered the ground-state energy (off by >1000 cm^-1); the fix recovers
the excited reference almost exactly.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
CH2O_MOP = _REPO_ROOT / "data" / "mop_files" / "CH2O_Gaussian.mop"
N_MODES_FULL = 6
N_HO = 11

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import vib_sqd.basis.modal_transform as mb
from vib_sqd.classical.mop_forcefield import mop_to_forcefield, parse_mop
from vib_sqd.run.dualbasis import ensure_clean_vcc
from vib_sqd.run.excited_state import (
    build_excited_amps,
    excited_recovery_loop,
    excited_reference_energy,
)
from vib_sqd.sqd.vibrational_recovery import _onehot_from_modal_indices

midascpp = pytest.importorskip(
    "vib_sqd.run.pipeline", reason="requires a local MidasCpp installation"
)


@pytest.fixture(scope="module")
def ch2o_excited_fixture():
    n_modes, n_modals = N_MODES_FULL, 6
    mout, ffmop = ensure_clean_vcc(
        f"ch2o_6x{n_modals}", CH2O_MOP, None, n_modals, str(_HERE / "runs" / "ch2o")
    )
    ff = mop_to_forcefield(parse_mop(ffmop))
    rot = mb.build_modal_rotation(mout, n_modes, N_HO, n_modals)
    occupation, _amps, _mout_exc = build_excited_amps(
        ffmop,
        n_modes,
        n_modals,
        5,
        rot,
        str(_HERE / "runs" / "ch2o"),
        f"ch2o_6x{n_modals}",
    )
    exc_ref = excited_reference_energy(
        "modal", occupation, ff, rot, None, n_modals, N_HO
    )
    ground_ref = excited_reference_energy(
        "modal", [0] * n_modes, ff, rot, None, n_modals, N_HO
    )
    return dict(
        n_modes=n_modes,
        n_modals=n_modals,
        ff=ff,
        rot=rot,
        N_HO=N_HO,
        occupation=occupation,
        exc_ref=exc_ref,
        ground_ref=ground_ref,
    )


def test_recovers_excited_state_not_ground_state_under_noise(ch2o_excited_fixture):
    f = ch2o_excited_fixture
    n_modes, n_modals = f["n_modes"], f["n_modals"]
    main_config = _onehot_from_modal_indices([0] * n_modes, n_modals)

    raw = {main_config: 99000}
    rng = np.random.default_rng(0)
    for _ in range(1000):
        b = "".join(rng.choice(["0", "1"]) for _ in range(n_modes * n_modals))
        raw[b] = raw.get(b, 0) + 1

    e, _iters, _rec, _occ = excited_recovery_loop(
        "modal",
        raw,
        f["occupation"],
        f["ff"],
        f["rot"],
        None,
        n_modes,
        n_modals,
        f["N_HO"],
    )

    dist_to_excited = abs(e - f["exc_ref"])
    dist_to_ground = abs(e - f["ground_ref"])
    assert dist_to_excited < 20.0, (
        f"recovered {e:.2f} cm-1 is not close to the excited reference "
        f"{f['exc_ref']:.2f} cm-1 (95%-clean-signal regression case)"
    )
    assert dist_to_excited < dist_to_ground, (
        "recovered energy is closer to the GROUND reference than the "
        "excited one -- this is exactly the w[0]-selection bug"
    )
