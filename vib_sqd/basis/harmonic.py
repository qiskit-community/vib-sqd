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

from __future__ import annotations

import numpy as np


def ladder_operators(vmax: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build annihilation and creation operators in the harmonic oscillator basis.

    Basis:
        |0>, |1>, ..., |vmax>

    Returns:
        a, adag
    """
    if vmax < 0:
        raise ValueError("vmax must be non-negative.")

    dim = vmax + 1
    a = np.zeros((dim, dim), dtype=complex)
    adag = np.zeros((dim, dim), dtype=complex)

    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)
        adag[n, n - 1] = np.sqrt(n)

    return a, adag


def q_power_matrix(vmax: int, power: int) -> np.ndarray:
    """
    Compute <n'| q^power |n> in the harmonic oscillator basis n=0..vmax.

    Coordinate convention:
        q = (a + a†) / sqrt(2)

    This is a dimensionless harmonic oscillator coordinate.
    """
    if vmax < 0:
        raise ValueError("vmax must be non-negative.")
    if power < 0:
        raise ValueError("power must be non-negative.")

    # Use a larger workspace before truncation so q^k can virtually leave
    # the final retained subspace during multiplication.
    dim_work = vmax + 1 + power
    a, adag = ladder_operators(dim_work - 1)
    q = (a + adag) / np.sqrt(2.0)

    qk = np.linalg.matrix_power(q, power)
    return qk[: vmax + 1, : vmax + 1]


def number_operator(vmax: int) -> np.ndarray:
    """
    Harmonic oscillator number operator n in basis |0>, ..., |vmax>.
    """
    if vmax < 0:
        raise ValueError("vmax must be non-negative.")
    return np.diag(np.arange(vmax + 1, dtype=float))
