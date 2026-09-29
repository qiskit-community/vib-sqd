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
Selected / sparse UVCCSD: keep only the largest-|t2| double excitations.

UVCCSD's double-excitation operators are the only gates that actually use the
VCC t2 amplitudes (LUCJ/CHC map t2 onto diagonal phases that are inert on the
reference). Doubles also dominate the circuit depth. Since t2 is sparse (for
water, 15 nonzero of 81; the largest is ~4x the next), keeping only the largest
doubles retains most of the warm-start benefit at a fraction of the depth.

This module provides:
  * selection of the double set (by threshold or top-k), and
  * the matching sliced UVCCSD initial-parameter vector,
kept in EXACTLY the order the builder emits doubles
(nested `for m1: for m2>m1: for e1: for e2`) so the vector binds correctly to
`UVCCSDAnsatz.build(..., selected_doubles=...)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2] / "src"))

from vib_sqd.classical.vcc_solver import VCCAmplitudes
from vib_sqd.initialization.vcc_initialization import (
    map_vccsd_to_uvccsd_initial_point,
)


def all_double_indices(n_modes: int, n_modals: int):
    """Every (m1,m2,e1,e2) in the builder's emission order."""
    n_exc = n_modals - 1
    out = []
    for m1 in range(n_modes):
        for m2 in range(m1 + 1, n_modes):
            for e1 in range(n_exc):
                for e2 in range(n_exc):
                    out.append((m1, m2, e1, e2))
    return out


def select_doubles_by_threshold(t2: np.ndarray, thr: float) -> list[tuple]:
    """Doubles with |t2| > thr, in builder-emission order."""
    n_modes = t2.shape[0]
    n_modals = t2.shape[2] + 1
    return [d for d in all_double_indices(n_modes, n_modals) if abs(t2[d]) > thr]


def select_top_k_doubles(t2: np.ndarray, k: int) -> list[tuple]:
    """The k largest-|t2| doubles, returned in builder-emission order."""
    n_modes = t2.shape[0]
    n_modals = t2.shape[2] + 1
    alld = all_double_indices(n_modes, n_modals)
    ranked = sorted(alld, key=lambda d: abs(t2[d]), reverse=True)
    keep = set(ranked[:k])
    return [d for d in alld if d in keep]  # preserve emission order


def sliced_uvccsd_init(
    amps: VCCAmplitudes,
    n_modes: int,
    n_modals: int,
    selected_doubles: list[tuple],
    scaling_strategy: str = "direct",
    clip: float | None = None,
) -> np.ndarray:
    """UVCCSD init vector restricted to `selected_doubles`.

    The full mapper emits [singles (N*(M-1))] + [doubles in nested order]. We
    keep all singles and only the doubles whose (m1,m2,e1,e2) is selected, in
    the same order the sparse builder emits them.
    """
    full = map_vccsd_to_uvccsd_initial_point(
        amps,
        n_modes=n_modes,
        n_modals=n_modals,
        scaling_strategy=scaling_strategy,
        clip=clip,
    )
    n_exc = n_modals - 1
    n_singles = n_modes * n_exc
    singles = full[:n_singles]

    alld = all_double_indices(n_modes, n_modals)
    doubles_full = full[n_singles:]
    pos = {d: i for i, d in enumerate(alld)}
    sel_doubles = np.array(
        [doubles_full[pos[d]] for d in selected_doubles], dtype=float
    )
    return np.concatenate([singles, sel_doubles])
