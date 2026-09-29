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

"""Modal vs harmonic-oscillator basis Hamiltonians, and the exact ground-state
probability distribution. Uses a small built-in force field (no MidasCpp).
"""

import numpy as np
import pytest

import vib_sqd.basis.modal_transform as mb
from vib_sqd.data.forcefields.co2 import load_co2_bending_only


@pytest.fixture(scope="module")
def ff():
    return load_co2_bending_only()


def test_ho_hamiltonian_hermitian_and_dim(ff):
    n_modes = len(ff.omega)
    for n_ho in (3, 4, 5):
        H = mb.ho_hamiltonian_full(ff, n_ho)
        assert H.shape == (n_ho**n_modes, n_ho**n_modes)
        assert np.allclose(H, H.conj().T)  # Hermitian


def test_exact_ground_distribution_normalized_and_peaks_on_reference(ff):
    """|c|^2 of the exact HO ground state sums to 1 and its largest weight is on
    the all-zeros (reference) configuration for a near-harmonic potential."""
    n_ho = 5
    H = mb.ho_hamiltonian_full(ff, n_ho)
    w, V = np.linalg.eigh(H)
    p = np.abs(V[:, 0]) ** 2
    assert p.sum() == pytest.approx(1.0)
    # reference config = index 0 (all modes in modal 0) in product order
    assert int(np.argmax(p)) == 0
    assert p[0] > 0.5  # dominant reference weight


def test_modal_compact_preserves_low_spectrum_when_M_equals_nho(ff):
    """When the modal count M equals n_ho, the modal transform U H U-dagger is a
    similarity transform and must reproduce the HO eigenvalues exactly.

    We build the rotation as identity blocks (M == n_ho) to exercise
    modal_hamiltonian_compact without needing a MidasCpp VSCF file.
    """
    n_modes = len(ff.omega)
    n = 5
    H = mb.ho_hamiltonian_full(ff, n)
    # identity rotation per mode: C_i = I_{n x n}; modal basis == HO basis
    rot = [np.eye(n) for _ in range(n_modes)]
    Hc = mb.modal_hamiltonian_compact(H, rot)
    assert Hc.shape == H.shape
    assert np.allclose(np.linalg.eigvalsh(Hc), np.linalg.eigvalsh(H), atol=1e-8)


def test_modal_truncation_is_variational_upper_bound(ff):
    """Truncating to M < n_ho modals (identity embedding) gives a ground energy
    >= the full HO ground energy (Rayleigh-Ritz)."""
    n_modes = len(ff.omega)
    n_ho = 6
    H = mb.ho_hamiltonian_full(ff, n_ho)
    e_full = np.linalg.eigvalsh(H)[0]
    for M in (3, 4, 5):
        rot = [np.eye(n_ho)[:M, :] for _ in range(n_modes)]  # M x n_ho isometry
        Hc = mb.modal_hamiltonian_compact(H, rot)
        assert Hc.shape == (M**n_modes, M**n_modes)
        e_trunc = np.linalg.eigvalsh(Hc)[0]
        assert e_trunc >= e_full - 1e-9  # variational: no lower than full


def test_embed_onehot_places_compact_H_on_physical_indices(ff):
    """embed_onehot lifts the compact M^N Hamiltonian to the 2^(N*M) one-hot space
    and its eigenvalues on the physical subspace match the compact ones."""
    n_modes = len(ff.omega)
    n_modals = 3
    Hc = mb.modal_hamiltonian_compact(
        mb.ho_hamiltonian_full(ff, 6), [np.eye(6)[:n_modals, :] for _ in range(n_modes)]
    )
    Hfull, phys = mb.embed_onehot(Hc, n_modes, n_modals)
    assert Hfull.shape[0] == 2 ** (n_modes * n_modals)
    sub = Hfull[np.ix_(phys, phys)]
    assert np.allclose(np.linalg.eigvalsh(sub), np.linalg.eigvalsh(Hc), atol=1e-8)
