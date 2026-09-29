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

"""build_subspace_hamiltonian: cross-check the sparse per-mode-factorized
Hamiltonian (restricted to a config subset) against the dense modal
Hamiltonian built the "obvious" way (ho_hamiltonian_full + modal_hamiltonian_
compact), for both the harmonic term and multi-mode coupling terms. Uses an
identity rotation (M == n_ho) so no MidasCpp VSCF file is needed, matching the
convention in test_modal_vs_ho_basis.py.
"""

from itertools import product

import numpy as np
import pytest

import vib_sqd.basis.modal_transform as mb
from vib_sqd.data.forcefields.co2 import (
    ForceTerm,
    VibrationalForceField,
    load_co2_approximate_forcefield,
)
from vib_sqd.sqd.subspace_hamiltonian import build_subspace_hamiltonian


def _dense_and_configs(ff, n_ho, n_modes, rng, n_subset):
    """Ground truth via the dense route, plus a random config subset (as tuples)
    and their indices into the product-order enumeration."""
    H_ho = mb.ho_hamiltonian_full(ff, n_ho)
    rot = [np.eye(n_ho) for _ in range(n_modes)]  # identity: modal == HO
    Hc = mb.modal_hamiltonian_compact(H_ho, rot)
    all_cfgs = list(product(range(n_ho), repeat=n_modes))
    chosen = rng.choice(len(all_cfgs), size=min(n_subset, len(all_cfgs)), replace=False)
    configs = [all_cfgs[i] for i in chosen]
    return Hc, rot, configs, chosen


@pytest.fixture(scope="module")
def ff():
    return load_co2_approximate_forcefield()


def test_subspace_matches_dense_modal_hamiltonian(ff):
    n_ho = 5
    n_modes = len(ff.omega)
    rng = np.random.default_rng(0)
    Hc, rot, configs, chosen = _dense_and_configs(ff, n_ho, n_modes, rng, n_subset=40)
    H_sub = build_subspace_hamiltonian(configs, ff, rot, n_ho)
    expected = Hc[np.ix_(chosen, chosen)]
    assert H_sub.shape == expected.shape
    assert np.allclose(H_sub, expected, atol=1e-8)


def test_subspace_hamiltonian_is_hermitian(ff):
    n_ho = 4
    n_modes = len(ff.omega)
    rng = np.random.default_rng(1)
    _, rot, configs, _ = _dense_and_configs(ff, n_ho, n_modes, rng, n_subset=25)
    H_sub = build_subspace_hamiltonian(configs, ff, rot, n_ho)
    assert np.allclose(H_sub, H_sub.conj().T, atol=1e-10)


def test_subspace_matches_dense_with_full_product_space(ff):
    """Using ALL configs (not a subset) must reproduce the dense modal H exactly
    -- this exercises every branch of the untouched-mode grouping without any
    subsampling-induced coincidence."""
    n_ho = 3
    n_modes = len(ff.omega)
    H_ho = mb.ho_hamiltonian_full(ff, n_ho)
    rot = [np.eye(n_ho) for _ in range(n_modes)]
    Hc = mb.modal_hamiltonian_compact(H_ho, rot)
    all_cfgs = list(product(range(n_ho), repeat=n_modes))
    H_sub = build_subspace_hamiltonian(all_cfgs, ff, rot, n_ho)
    assert np.allclose(H_sub, Hc, atol=1e-8)


def test_harmonic_omega_override(ff):
    """Passing an explicit harmonic_omega must be used instead of ff.omega."""
    n_ho = 4
    n_modes = len(ff.omega)
    custom_omega = [w * 1.5 for w in ff.omega]
    rng = np.random.default_rng(2)
    _, rot, configs, chosen = _dense_and_configs(ff, n_ho, n_modes, rng, n_subset=20)

    class _FFCustomOmega:
        omega = custom_omega
        terms = ff.terms

    H_ho_custom = mb.ho_hamiltonian_full(_FFCustomOmega(), n_ho)
    Hc_custom = mb.modal_hamiltonian_compact(H_ho_custom, rot)
    expected = Hc_custom[np.ix_(chosen, chosen)]

    H_sub = build_subspace_hamiltonian(
        configs, ff, rot, n_ho, harmonic_omega=custom_omega
    )
    assert np.allclose(H_sub, expected, atol=1e-8)


def test_single_config_gives_diagonal_element_only():
    """A one-config subspace is a 1x1 matrix equal to <c|H|c>."""
    ff2 = VibrationalForceField(
        omega=(500.0, 700.0),
        terms=(ForceTerm(2.0, (1, 2), (1, 1)),),
    )
    n_ho = 4
    rot = [np.eye(n_ho), np.eye(n_ho)]
    H_ho = mb.ho_hamiltonian_full(ff2, n_ho)
    Hc = mb.modal_hamiltonian_compact(H_ho, rot)
    cfg = (2, 1)
    idx = list(product(range(n_ho), repeat=2)).index(cfg)
    H_sub = build_subspace_hamiltonian([cfg], ff2, rot, n_ho)
    assert H_sub.shape == (1, 1)
    assert H_sub[0, 0] == pytest.approx(Hc[idx, idx], abs=1e-8)


def test_no_python_quadratic_blowup_on_larger_synthetic_system():
    """Non-regression guard: a ~20-mode, ~500-config subspace with many random
    2-3-mode coupling terms must build in well under a generous wall-clock
    bound. This does not assert a specific speed, just that the vectorized
    grouping path is actually being used (a reintroduced O(n_cfg^2) Python
    loop would make this test take tens of seconds to minutes instead)."""
    import time

    rng = np.random.default_rng(3)
    n_modes, n_modals, n_ho = 20, 4, 6
    omega = tuple(rng.uniform(400, 2000, size=n_modes))
    terms = []
    for _ in range(60):
        arity = rng.choice([1, 2, 3])
        modes = tuple(
            sorted(
                rng.choice(
                    np.arange(1, n_modes + 1), size=arity, replace=False
                ).tolist()
            )
        )
        powers = tuple(int(x) for x in rng.integers(1, 3, size=arity))
        terms.append(ForceTerm(float(rng.uniform(-5, 5)), modes, powers))
    ff_big = VibrationalForceField(omega=omega, terms=tuple(terms))
    rot = [np.eye(n_ho)[:n_modals, :] for _ in range(n_modes)]
    configs = [
        tuple(int(x) for x in rng.integers(0, n_modals, size=n_modes))
        for _ in range(500)
    ]

    t0 = time.perf_counter()
    H = build_subspace_hamiltonian(configs, ff_big, rot, n_ho)
    elapsed = time.perf_counter() - t0

    assert H.shape == (500, 500)
    assert np.allclose(H, H.conj().T, atol=1e-8)
    assert (
        elapsed < 10.0
    ), f"took {elapsed:.1f}s -- possible reintroduced O(n_cfg^2) Python loop"
