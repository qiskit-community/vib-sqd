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
Modal basis solver for vibrational quantum chemistry.

Implements the modal basis approach from:
"Quantum simulation of molecular vibrations with quantum computers"
McArdle et al., Chem. Sci., 2020

The modal basis is obtained by solving the one-body Hamiltonian:
    (T(Q) + V(Q)) φ_k(Q) = ε_k φ_k(Q)

where T is kinetic energy, V is the anharmonic potential for a single mode.

This uses the Discrete Variable Representation (DVR) method with
distributed Gaussian basis functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.linalg import eigh


@dataclass
class ModalBasis:
    """
    Modal basis for a single vibrational mode.

    Attributes
    ----------
    energies : np.ndarray
        Modal energies (eigenvalues) in cm⁻¹
    wavefunctions : np.ndarray
        Modal wavefunctions on the DVR grid, shape (n_dvr, n_modals)
    grid : np.ndarray
        DVR grid points (coordinate values)
    weights : np.ndarray
        DVR quadrature weights
    n_modals : int
        Number of modals retained
    """

    energies: np.ndarray
    wavefunctions: np.ndarray
    grid: np.ndarray
    weights: np.ndarray
    n_modals: int


def build_dvr_grid(
    q_min: float, q_max: float, n_points: int, basis_type: str = "gaussian"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build DVR grid and quadrature weights.

    Parameters
    ----------
    q_min : float
        Minimum coordinate value
    q_max : float
        Maximum coordinate value
    n_points : int
        Number of DVR points
    basis_type : str
        Type of basis functions ("gaussian" or "sinc")

    Returns
    -------
    grid : np.ndarray
        DVR grid points
    weights : np.ndarray
        Quadrature weights
    """
    if basis_type == "sinc":
        # Sinc-DVR (Colbert-Miller) on uniform grid
        # Note: This is NOT distributed Gaussian basis
        grid = np.linspace(q_min, q_max, n_points)
        dq = grid[1] - grid[0]
        weights = np.full(n_points, dq)

    elif basis_type == "gaussian":
        raise NotImplementedError(
            "Distributed Gaussian basis is not yet implemented. "
            "Use basis_type='sinc' for uniform-grid DVR."
        )

    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

    return grid, weights


def build_kinetic_matrix_dvr(
    grid: np.ndarray, omega: float = 1000.0, mass: float = 1.0, hbar: float = 1.0
) -> np.ndarray:
    """
    Build kinetic energy matrix in DVR.

    Uses the Colbert-Miller formula for kinetic energy in DVR:
    T_ij = (ℏ²/2m) * (π²/3Δx²) δ_ij  for i=j
    T_ij = (ℏ²/2m) * (2(-1)^(i-j)/(i-j)²Δx²)  for i≠j

    For vibrational problems in dimensionless coordinates, we scale by ω.

    Parameters
    ----------
    grid : np.ndarray
        DVR grid points
    omega : float
        Harmonic frequency in cm⁻¹ (for scaling)
    mass : float
        Reduced mass (atomic units)
    hbar : float
        Reduced Planck constant (atomic units)

    Returns
    -------
    T : np.ndarray
        Kinetic energy matrix in cm⁻¹
    """
    n = len(grid)
    dx = grid[1] - grid[0]

    T = np.zeros((n, n))

    # Diagonal elements
    # Scale kinetic energy to match potential energy units (cm⁻¹)
    prefactor = omega / (2 * dx**2)
    for i in range(n):
        T[i, i] = prefactor * (np.pi**2 / 3.0)

    # Off-diagonal elements
    for i in range(n):
        for j in range(i + 1, n):
            diff = i - j
            T[i, j] = prefactor * 2 * ((-1) ** diff) / (diff**2)
            T[j, i] = T[i, j]

    return T


def build_potential_matrix_dvr(
    grid: np.ndarray, potential_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Build potential energy matrix in DVR.

    In DVR, the potential is diagonal:
    V_ij = V(q_i) δ_ij

    Parameters
    ----------
    grid : np.ndarray
        DVR grid points
    potential_func : callable
        Function that takes grid points and returns potential values

    Returns
    -------
    V : np.ndarray
        Potential energy matrix (diagonal)
    """
    V_values = potential_func(grid)
    return np.diag(V_values)


def solve_modal_basis(
    omega: float,
    anharmonic_terms: list,
    n_modals: int,
    n_dvr: int = 50,
    q_range: float = 10.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ModalBasis:
    """
    Solve for modal basis of a single vibrational mode.

    Solves the 1D Schrödinger equation:
        (T + V) φ_k = ε_k φ_k

    where V includes harmonic + anharmonic terms.

    Parameters
    ----------
    omega : float
        Harmonic frequency in cm⁻¹
    anharmonic_terms : list of tuple
        List of (coefficient, power) for anharmonic terms
        e.g., [(k3, 3), (k4, 4)] for k3*Q³ + k4*Q⁴
    n_modals : int
        Number of modals to retain
    n_dvr : int
        Number of DVR grid points
    q_range : float
        Range of coordinate grid (±q_range)
    mass : float
        Reduced mass in atomic units
    hbar : float
        Reduced Planck constant in atomic units

    Returns
    -------
    modal_basis : ModalBasis
        Modal basis with energies and wavefunctions
    """
    # Build DVR grid (using sinc-DVR, not distributed Gaussians)
    grid, weights = build_dvr_grid(-q_range, q_range, n_dvr, "sinc")

    # Build kinetic energy matrix
    T = build_kinetic_matrix_dvr(grid, omega, mass, hbar)

    # Build potential energy matrix
    def potential(q):
        # Work in cm⁻¹ units directly (no conversion needed)
        # Harmonic term: (1/2) * ω² * Q² (in dimensionless units)
        # For harmonic oscillator: V = (1/2) * ω * Q²
        V = 0.5 * omega * q**2

        # Add anharmonic terms
        for coeff, power in anharmonic_terms:
            V += coeff * q**power

        return V

    V = build_potential_matrix_dvr(grid, potential)

    # Solve eigenvalue problem
    H = T + V
    eigenvalues, eigenvectors = eigh(H)

    # Keep only the lowest n_modals states
    energies = eigenvalues[:n_modals]
    wavefunctions = eigenvectors[:, :n_modals]

    # Normalize wavefunctions (do NOT multiply by weights when using simpson with x=grid)
    for k in range(n_modals):
        norm = np.sqrt(simpson(wavefunctions[:, k] ** 2, x=grid))
        wavefunctions[:, k] /= norm

    # Energies are already in cm⁻¹
    return ModalBasis(
        energies=energies,
        wavefunctions=wavefunctions,
        grid=grid,
        weights=weights,
        n_modals=n_modals,
    )


def compute_modal_matrix_element_1body(
    modal_basis: ModalBasis,
    operator_func: Callable[[np.ndarray], np.ndarray],
    k: int,
    l: int,
) -> float:
    """
    Compute one-body matrix element in modal basis.

    <φ_k | Ô | φ_l> = ∫ φ_k(q) Ô(q) φ_l(q) dq

    Parameters
    ----------
    modal_basis : ModalBasis
        Modal basis
    operator_func : callable
        Operator as function of coordinate
    k, l : int
        Modal indices

    Returns
    -------
    matrix_element : float
        <φ_k | Ô | φ_l>
    """
    grid = modal_basis.grid
    weights = modal_basis.weights

    psi_k = modal_basis.wavefunctions[:, k]
    psi_l = modal_basis.wavefunctions[:, l]

    O_values = operator_func(grid)

    # Do NOT multiply by weights when using simpson with x=grid
    integrand = psi_k * O_values * psi_l
    result = simpson(integrand, x=grid)
    return float(result)


def compute_modal_matrix_element_2body(
    modal_basis1: ModalBasis,
    modal_basis2: ModalBasis,
    operator_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    k1: int,
    l1: int,
    k2: int,
    l2: int,
) -> float:
    """
    Compute two-body matrix element in modal basis.

    <φ_k1 φ_k2 | Ô | φ_l1 φ_l2> = ∫∫ φ_k1(q1) φ_k2(q2) Ô(q1,q2) φ_l1(q1) φ_l2(q2) dq1 dq2

    Parameters
    ----------
    modal_basis1, modal_basis2 : ModalBasis
        Modal bases for modes 1 and 2
    operator_func : callable
        Operator as function of two coordinates
    k1, l1 : int
        Modal indices for mode 1
    k2, l2 : int
        Modal indices for mode 2

    Returns
    -------
    matrix_element : float
        <φ_k1 φ_k2 | Ô | φ_l1 φ_l2>
    """
    grid1 = modal_basis1.grid
    grid2 = modal_basis2.grid
    weights1 = modal_basis1.weights
    weights2 = modal_basis2.weights

    psi_k1 = modal_basis1.wavefunctions[:, k1]
    psi_l1 = modal_basis1.wavefunctions[:, l1]
    psi_k2 = modal_basis2.wavefunctions[:, k2]
    psi_l2 = modal_basis2.wavefunctions[:, l2]

    # Create 2D grids
    Q1, Q2 = np.meshgrid(grid1, grid2, indexing="ij")
    O_values = operator_func(Q1, Q2)

    # Compute 2D integral (do NOT multiply by weights when using simpson with x=grid)
    integrand = np.outer(psi_k1 * psi_l1, psi_k2 * psi_l2) * O_values

    result = simpson(simpson(integrand, x=grid2, axis=1), x=grid1)
    return float(result)


def build_modal_hamiltonian_terms(
    modal_bases: list[ModalBasis], forcefield_terms: list, omega: np.ndarray
) -> list:
    """
    Build Hamiltonian terms in modal basis.

    Converts forcefield terms from coordinate representation to
    modal basis matrix elements.

    Parameters
    ----------
    modal_bases : list of ModalBasis
        Modal basis for each mode
    forcefield_terms : list
        Forcefield terms with (coefficient, mode_indices)
    omega : np.ndarray
        Harmonic frequencies for each mode

    Returns
    -------
    modal_terms : list
        Hamiltonian terms in modal basis
    """
    n_modes = len(modal_bases)
    modal_terms = []

    # One-body terms (diagonal energies)
    for mode_idx in range(n_modes):
        basis = modal_bases[mode_idx]
        for k in range(basis.n_modals):
            # Modal energy term
            coeff = basis.energies[k]
            modal_terms.append(
                {
                    "coeff": coeff,
                    "modes": [mode_idx],
                    "modals": [k, k],
                    "type": "diagonal",
                }
            )

    # Process forcefield terms
    for term in forcefield_terms:
        coeff = term.coeff
        modes = term.modes

        if len(modes) == 1:
            # Single-mode anharmonic term (already included in modal energies)
            continue

        elif len(modes) == 2:
            # Two-mode coupling with actual powers
            mode1, mode2 = modes[0] - 1, modes[1] - 1  # Convert to 0-indexed

            # Get powers for this term
            if hasattr(term, "powers") and term.powers is not None:
                p1, p2 = term.powers
            else:
                # Default to linear coupling if powers not specified
                p1, p2 = 1, 1

            basis1 = modal_bases[mode1]
            basis2 = modal_bases[mode2]

            # Create operator with correct powers
            def make_operator(pow1, pow2):
                def operator(q1, q2):
                    return q1**pow1 * q2**pow2

                return operator

            operator_func = make_operator(p1, p2)

            # Compute all matrix elements
            for k1 in range(basis1.n_modals):
                for l1 in range(basis1.n_modals):
                    for k2 in range(basis2.n_modals):
                        for l2 in range(basis2.n_modals):
                            element = compute_modal_matrix_element_2body(
                                basis1, basis2, operator_func, k1, l1, k2, l2
                            )

                            if abs(element) > 1e-10:
                                modal_terms.append(
                                    {
                                        "coeff": coeff * element,
                                        "modes": [mode1, mode2],
                                        "modals": [k1, l1, k2, l2],
                                        "type": "coupling",
                                        "powers": (p1, p2),
                                    }
                                )

    return modal_terms
