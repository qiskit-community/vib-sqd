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

"""QSCI+X augmentation and the iterative subspace-growth recovery, including a
regression test for the fixed bug (index-based truncation used to drop the
high-weight reference config).
"""

import numpy as np
import pytest

from vib_sqd.sqd import qsci_augment as qa


def _spd_toy_h(n_modes, n_modals, seed_scale=0.01):
    """A small Hermitian, diagonally-dominant toy Hamiltonian in product order,
    with the reference config (index 0) as the lowest diagonal."""
    dim = n_modals**n_modes
    # deterministic (no RNG): symmetric small off-diagonals + increasing diagonal
    A = np.fromfunction(lambda i, j: seed_scale / (1 + np.abs(i - j)), (dim, dim))
    H = (A + A.T) / 2 + np.diag(np.arange(dim, dtype=float))
    return H


def test_singles_doubles_counts():
    parent = (0, 0, 0)
    vmax = 2
    s = qa.singles(parent, vmax)
    # each of 3 modes -> vmax other levels
    assert len(s) == 3 * vmax
    d = qa.doubles(parent, vmax)
    assert all(p != parent for p in d)
    assert len(d) == (3 * 2 // 2) * vmax * vmax  # C(3,2) * vmax^2


def test_augment_includes_parents():
    parents = [(0, 0, 0)]
    aug = qa.augment_configs(parents, vmax=2, level="SD")
    assert (0, 0, 0) in set(map(tuple, aug))


def test_growth_from_reference_recovers_and_saturates():
    """From the single reference parent, growth reaches the exact ground energy
    once the subspace saturates the full space."""
    n_modes, n_modals = 3, 3
    H = _spd_toy_h(n_modes, n_modals)
    exact = float(np.linalg.eigvalsh(H)[0])
    trace = qa.iterative_subspace_growth(
        H, [0], n_modes, n_modals, max_rounds=6, tol_cm=1e-9, cm2au=1.0
    )
    n_last, e_last = trace[-1]
    assert e_last == pytest.approx(exact, abs=1e-6)
    # monotone non-increasing energy as the subspace grows
    energies = [e for _, e in trace]
    assert all(energies[i + 1] <= energies[i] + 1e-9 for i in range(len(energies) - 1))


def test_growth_keeps_high_weight_reference_under_tight_cap():
    """REGRESSION: with a bounded subspace and weight ranking, the high-weight
    reference config must be kept (the old index-based truncation dropped it,
    giving spuriously large recovered energies)."""
    n_modes, n_modals = 3, 3
    H = _spd_toy_h(n_modes, n_modals)
    exact = float(np.linalg.eigvalsh(H)[0])
    # parents: the reference (idx 0, high weight) + several high-index "noise"
    # configs with near-flat low weights, mimicking a noisy sample.
    noise = list(range(1, 20))
    parents = [0] + noise
    weights = {0: 0.30}
    weights.update({k: 0.02 for k in noise})
    trace = qa.iterative_subspace_growth(
        H,
        parents,
        n_modes,
        n_modals,
        max_rounds=4,
        tol_cm=1e-6,
        cm2au=1.0,
        max_size=12,
        weights=weights,
    )
    # if the reference (idx 0) survived truncation, the recovered energy is close
    # to exact; the old bug produced a large gap because idx 0 was dropped.
    assert trace[-1][1] - exact < 1.5  # NOT the ~hundreds-cm-1 bug regime


def test_growth_ignores_weights_falls_back_to_index_order():
    """weights=None path still works (used where the full subspace fits)."""
    n_modes, n_modals = 2, 3
    H = _spd_toy_h(n_modes, n_modals)
    exact = float(np.linalg.eigvalsh(H)[0])
    trace = qa.iterative_subspace_growth(
        H, [0], n_modes, n_modals, max_rounds=5, tol_cm=1e-9, cm2au=1.0
    )
    assert trace[-1][1] == pytest.approx(exact, abs=1e-6)


def test_augmented_ground_energy_le_reference_only():
    """Augmenting the parent set can only lower (or equal) the projected energy."""
    n_modes, n_modals = 3, 3
    H = _spd_toy_h(n_modes, n_modals)
    e_ref_only, nb1, _ = qa.augmented_ground_energy(H, [0], n_modes, n_modals, level="")
    e_aug, nb2, _ = qa.augmented_ground_energy(H, [0], n_modes, n_modals, level="SD")
    assert nb2 >= nb1
    assert e_aug <= e_ref_only + 1e-9
