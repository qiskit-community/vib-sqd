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

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class NaturalModalInitialization:
    """
    Natural-modal initialization data.

    rdms:
        One-mode reduced density matrices, one per mode.
    occupations:
        Eigenvalues of each one-mode RDM, sorted descending.
    natural_orbitals:
        Eigenvectors of each RDM, columns sorted by descending occupation.
    chain_angles:
        Adjacent Givens angles per mode that prepare the dominant natural modal
        from modal-0 reference.
    vlucj_x0:
        Flat VLUCJ parameter vector matching build_parameterized_vlucj order.
    """

    rdms: tuple[np.ndarray, ...]
    occupations: tuple[np.ndarray, ...]
    natural_orbitals: tuple[np.ndarray, ...]
    chain_angles: tuple[np.ndarray, ...]
    vlucj_x0: np.ndarray


def qiskit_one_hot_index_from_occupation(
    occupation: Sequence[int],
    n_modals: int,
) -> int:
    """
    Map modal occupation tuple to Qiskit statevector integer index.

    Convention used by your circuits:
        qubit = mode * n_modals + modal

    Qiskit statevector integer indexing is little-endian:
        qubit q contributes 2**q.
    """
    idx = 0
    for mode, modal in enumerate(occupation):
        qubit = mode * n_modals + int(modal)
        idx |= 1 << qubit
    return idx


def physical_amplitude_tensor_from_statevector(
    statevector: Statevector | np.ndarray,
    n_modes: int,
    n_modals: int,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract amplitudes on the physical one-hot subspace and reshape them as:

        coeffs[k1, k2, ..., kL]

    where kl is the modal index occupied by mode l.

    This uses the circuit/Qiskit qubit convention:
        qubit = mode * n_modals + modal
    """
    data = np.asarray(
        statevector.data if isinstance(statevector, Statevector) else statevector,
        dtype=complex,
    )

    coeffs = np.zeros((n_modals,) * n_modes, dtype=complex)

    for occ in product(range(n_modals), repeat=n_modes):
        idx = qiskit_one_hot_index_from_occupation(occ, n_modals)
        coeffs[occ] = data[idx]

    if normalize:
        norm = np.linalg.norm(coeffs.ravel())
        if norm > 1e-14:
            coeffs = coeffs / norm

    return coeffs


def one_mode_rdms_from_physical_coefficients(
    coeffs: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Compute one-mode reduced density matrices from physical coefficients.

    coeffs shape:
        (n_modals, n_modals, ..., n_modals)

    For mode m:
        rho_m[a,b] = sum_{all other modes} C[...,a,...] C*[...,b,...]
    """
    n_modes = coeffs.ndim
    n_modals = coeffs.shape[0]

    rdms = []

    for mode in range(n_modes):
        rho = np.zeros((n_modals, n_modals), dtype=complex)

        # Move target mode to first axis and flatten all other modes.
        moved = np.moveaxis(coeffs, mode, 0)
        mat = moved.reshape(n_modals, -1)

        rho = mat @ mat.conj().T

        # Symmetrize for numerical stability.
        rho = 0.5 * (rho + rho.conj().T)
        rdms.append(rho)

    return tuple(rdms)


def one_mode_rdms_from_statevector(
    statevector: Statevector | np.ndarray,
    n_modes: int,
    n_modals: int,
) -> tuple[np.ndarray, ...]:
    coeffs = physical_amplitude_tensor_from_statevector(
        statevector,
        n_modes=n_modes,
        n_modals=n_modals,
        normalize=True,
    )
    return one_mode_rdms_from_physical_coefficients(coeffs)


def two_mode_rdms_from_physical_coefficients(
    coeffs: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Compute two-mode reduced density matrices from physical coefficients.

    coeffs shape:
        (n_modals, n_modals, ..., n_modals)

    For mode pair (i, j), returns rho_ij with shape:

        (n_modals, n_modals, n_modals, n_modals)

    where:

        rho_ij[a, b, c, d]
          = sum_rest C[..., a_i=a, a_j=b, ...]
                     C*[..., a_i=c, a_j=d, ...]

    The diagonal rho_ij[a,b,a,b] is the joint probability that:
        mode i occupies modal a and mode j occupies modal b.
    """
    n_modes = coeffs.ndim
    n_modals = coeffs.shape[0]

    out: dict[tuple[int, int], np.ndarray] = {}

    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            moved = np.moveaxis(coeffs, (i, j), (0, 1))
            mat = moved.reshape(n_modals * n_modals, -1)
            rho_flat = mat @ mat.conj().T
            rho = rho_flat.reshape(n_modals, n_modals, n_modals, n_modals)

            # Hermitian symmetrization in pair-index space.
            rho_flat = rho.reshape(n_modals * n_modals, n_modals * n_modals)
            rho_flat = 0.5 * (rho_flat + rho_flat.conj().T)
            rho = rho_flat.reshape(n_modals, n_modals, n_modals, n_modals)

            out[(i, j)] = rho

    return out


def two_mode_rdms_from_statevector(
    statevector: Statevector | np.ndarray,
    n_modes: int,
    n_modals: int,
) -> dict[tuple[int, int], np.ndarray]:
    coeffs = physical_amplitude_tensor_from_statevector(
        statevector,
        n_modes=n_modes,
        n_modals=n_modals,
        normalize=True,
    )
    return two_mode_rdms_from_physical_coefficients(coeffs)


def joint_probabilities_from_two_mode_rdms(
    two_rdms: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """
    Extract diagonal joint probabilities from two-mode RDMs.

    Returns:
        P[(i,j)][a,b] = rho_ij[a,b,a,b].real
    """
    probs: dict[tuple[int, int], np.ndarray] = {}

    for pair, rho in two_rdms.items():
        n_modals = rho.shape[0]
        p = np.zeros((n_modals, n_modals), dtype=float)

        for a in range(n_modals):
            for b in range(n_modals):
                p[a, b] = float(np.real(rho[a, b, a, b]))

        # Remove tiny negative numerical noise.
        p[p < 0.0] = 0.0

        total = float(np.sum(p))
        if total > 1e-14:
            p /= total

        probs[pair] = p

    return probs


def connected_pair_correlations_from_probabilities(
    one_rdms: Sequence[np.ndarray],
    joint_probs: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """
    Compute connected modal occupation correlations:

        C_ij[a,b] = P_ij[a,b] - p_i[a] p_j[b]

    where:
        p_i[a] = rho_i[a,a]
    """
    marginals = []

    for rho in one_rdms:
        p = np.real(np.diag(rho)).astype(float)
        p[p < 0.0] = 0.0
        total = float(np.sum(p))
        if total > 1e-14:
            p /= total
        marginals.append(p)

    connected: dict[tuple[int, int], np.ndarray] = {}

    for (i, j), pij in joint_probs.items():
        connected[(i, j)] = pij - np.outer(marginals[i], marginals[j])

    return connected


def jastrow_angles_from_two_mode_statistics(
    one_rdms: Sequence[np.ndarray],
    two_rdms: dict[tuple[int, int], np.ndarray],
    n_modes: int,
    n_modals: int,
    method: str = "log_odds",
    scale: float = 1.0,
    clip: float = 0.5,
    eps: float = 1e-10,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Build Jastrow angle guesses from two-mode RDM statistics.

    Returns:
        angles[(i,j)] has shape (n_modals, n_modals)

    Supported methods:

    1. method='connected'
        angle[a,b] = scale * (P_ab - p_a p_b)

    2. method='normalized_connected'
        angle[a,b] = scale * (P_ab - p_a p_b)
                           / sqrt(p_a(1-p_a) p_b(1-p_b))

    3. method='log_odds'
        angle[a,b] = scale * log((P_ab + eps) / (p_a p_b + eps))

    Notes:
        These angles initialize phase correlations, not amplitudes directly.
        They are heuristic but physically motivated by two-mode correlations.
    """
    joint_probs = joint_probabilities_from_two_mode_rdms(two_rdms)
    connected = connected_pair_correlations_from_probabilities(one_rdms, joint_probs)

    marginals = []
    for rho in one_rdms:
        p = np.real(np.diag(rho)).astype(float)
        p[p < 0.0] = 0.0
        total = float(np.sum(p))
        if total > 1e-14:
            p /= total
        marginals.append(p)

    angles: dict[tuple[int, int], np.ndarray] = {}

    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            pij = joint_probs[(i, j)]
            cij = connected[(i, j)]
            pi = marginals[i]
            pj = marginals[j]

            theta = np.zeros((n_modals, n_modals), dtype=float)

            for a in range(n_modals):
                for b in range(n_modals):
                    if method == "connected":
                        val = cij[a, b]

                    elif method == "normalized_connected":
                        denom = np.sqrt(
                            max(pi[a] * (1.0 - pi[a]) * pj[b] * (1.0 - pj[b]), eps)
                        )
                        val = cij[a, b] / denom

                    elif method == "log_odds":
                        val = np.log((pij[a, b] + eps) / (pi[a] * pj[b] + eps))

                    else:
                        raise ValueError(
                            "method must be one of: connected, "
                            "normalized_connected, log_odds"
                        )

                    theta[a, b] = scale * val

            theta = np.clip(theta, -clip, clip)
            angles[(i, j)] = theta

    return angles


def insert_adjacent_same_modal_jastrow_angles_into_vlucj_x0(
    x0: np.ndarray,
    jastrow_angles: dict[tuple[int, int], np.ndarray],
    n_modes: int,
    n_modals: int,
    layers: int,
    include_rz: bool = True,
    distribute_over_layers: bool = True,
) -> np.ndarray:
    """
    Insert Jastrow angles into an existing VLUCJ x0 vector.

    Matches build_parameterized_vlucj order:

    per layer:
        first intramode block:
            n_modes * (n_modals - 1)
        adjacent-mode same-modal Jastrow:
            (n_modes - 1) * n_modals
        second intramode block:
            n_modes * (n_modals - 1)
        optional RZ:
            n_modes * n_modals

    Current VLUCJ only has adjacent-mode same-modal Jastrow terms:
        pair = (upper, upper+1), modal = a
        angle = jastrow_angles[(upper, upper+1)][a,a]
    """
    x0 = np.array(x0, dtype=float, copy=True)

    n_intramode_per_block = n_modes * (n_modals - 1)
    n_jastrow = max(n_modes - 1, 0) * n_modals
    n_rz = n_modes * n_modals if include_rz else 0
    params_per_layer = 2 * n_intramode_per_block + n_jastrow + n_rz

    expected = layers * params_per_layer
    if x0.shape != (expected,):
        raise ValueError(f"x0 has shape {x0.shape}; expected {(expected,)}")

    layer_scale = 1.0 / layers if distribute_over_layers else 1.0

    for layer in range(layers):
        base = layer * params_per_layer
        jastrow_offset = base + n_intramode_per_block

        p = jastrow_offset
        for upper in range(n_modes - 1):
            lower = upper + 1
            matrix = jastrow_angles.get((upper, lower))

            for modal in range(n_modals):
                if matrix is None:
                    val = 0.0
                else:
                    val = float(matrix[modal, modal])

                x0[p] = layer_scale * val
                p += 1

    return x0


def natural_modal_and_jastrow_initialization_from_statevector(
    statevector: Statevector | np.ndarray,
    n_modes: int,
    n_modals: int,
    vlucj_layers: int,
    include_rz: bool = True,
    placement: Literal["first_block", "split_blocks", "all_layers"] = "first_block",
    jastrow_method: str = "log_odds",
    jastrow_scale: float = 0.1,
    jastrow_clip: float = 0.3,
) -> NaturalModalInitialization:
    """
    Full two-stage initialization:

        1. One-mode RDM -> natural modal chain angles.
        2. Two-mode RDM -> Jastrow phase initialization.

    Returns:
        NaturalModalInitialization with vlucj_x0 containing both
        intramode natural rotations and adjacent same-modal Jastrow angles.
    """
    coeffs = physical_amplitude_tensor_from_statevector(
        statevector,
        n_modes=n_modes,
        n_modals=n_modals,
        normalize=True,
    )

    one_rdms = one_mode_rdms_from_physical_coefficients(coeffs)
    two_rdms = two_mode_rdms_from_physical_coefficients(coeffs)

    occupations, natural_orbitals = natural_orbitals_from_rdms(one_rdms)

    chain_angles = tuple(
        chain_angles_from_dominant_vector(vecs[:, 0]) for vecs in natural_orbitals
    )

    x0 = vlucj_x0_from_chain_angles(
        chain_angles=chain_angles,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=vlucj_layers,
        include_rz=include_rz,
        placement=placement,
    )

    jastrow_angles = jastrow_angles_from_two_mode_statistics(
        one_rdms=one_rdms,
        two_rdms=two_rdms,
        n_modes=n_modes,
        n_modals=n_modals,
        method=jastrow_method,
        scale=jastrow_scale,
        clip=jastrow_clip,
    )

    x0 = insert_adjacent_same_modal_jastrow_angles_into_vlucj_x0(
        x0=x0,
        jastrow_angles=jastrow_angles,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=vlucj_layers,
        include_rz=include_rz,
        distribute_over_layers=True,
    )

    return NaturalModalInitialization(
        rdms=tuple(one_rdms),
        occupations=tuple(occupations),
        natural_orbitals=tuple(natural_orbitals),
        chain_angles=chain_angles,
        vlucj_x0=x0,
    )


def natural_orbitals_from_rdms(
    rdms: Sequence[np.ndarray],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """
    Diagonalize each one-mode RDM.

    Returns:
        occupations:
            Eigenvalues sorted descending.
        natural_orbitals:
            Eigenvector matrices with columns sorted descending.
    """
    occupations = []
    natural_orbitals = []

    for rho in rdms:
        vals, vecs = np.linalg.eigh(rho)

        order = np.argsort(vals)[::-1]
        vals = np.real(vals[order])
        vecs = vecs[:, order]

        # Make dominant column mostly real with deterministic phase.
        for col in range(vecs.shape[1]):
            pivot = np.argmax(np.abs(vecs[:, col]))
            phase = np.exp(-1j * np.angle(vecs[pivot, col]))
            vecs[:, col] *= phase

        occupations.append(vals)
        natural_orbitals.append(vecs)

    return tuple(occupations), tuple(natural_orbitals)


def chain_angles_from_dominant_vector(
    vector: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert the dominant natural modal vector into adjacent-chain Givens angles.

    Your VLUCJ intramode block applies adjacent rotations:
        0 <-> 1, 1 <-> 2, 2 <-> 3, ...

    Starting from |0>, after angles θ0, θ1, ...:

        amp[0] = cos θ0
        amp[1] = sin θ0 cos θ1
        amp[2] = sin θ0 sin θ1 cos θ2
        ...

    This function extracts angles that approximately prepare the real part of
    the dominant natural orbital from |0>.

    Notes:
        - Complex phases are ignored here; VLUCJ local RZ phases can later absorb
          phase information if needed.
        - The sign is taken from each current component.
    """
    v = np.asarray(vector, dtype=complex)
    v = np.real_if_close(v).real.astype(float)

    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros(len(v) - 1, dtype=float)

    v = v / norm

    # Choose sign convention so v[0] is nonnegative.
    if v[0] < 0:
        v = -v

    n_modals = len(v)
    angles = np.zeros(n_modals - 1, dtype=float)

    remaining_norm = 1.0

    for a in range(n_modals - 1):
        if remaining_norm < eps:
            angles[a] = 0.0
            continue

        current = v[a] / remaining_norm
        tail = np.linalg.norm(v[a + 1 :]) / remaining_norm

        current = float(np.clip(current, -1.0, 1.0))
        tail = float(np.clip(tail, 0.0, 1.0))

        theta = np.arctan2(tail, current)

        # Preserve sign of the next branch if possible.
        if a + 1 < n_modals and v[a + 1] < 0:
            theta = -theta

        angles[a] = theta
        remaining_norm *= abs(np.sin(theta))

    return angles


def vlucj_x0_from_chain_angles(
    chain_angles: Sequence[np.ndarray],
    n_modes: int,
    n_modals: int,
    layers: int,
    include_rz: bool = True,
    placement: Literal["first_block", "split_blocks", "all_layers"] = "first_block",
) -> np.ndarray:
    """
    Build a flat VLUCJ parameter vector from per-mode adjacent-chain angles.

    Must match build_parameterized_vlucj order:

    per layer:
        first intramode block:
            for mode:
                for adjacent modal pair:
        adjacent-mode same-modal Jastrow:
            for upper in range(n_modes - 1):
                lower = upper + 1
                for modal:
        second intramode block:
            same as first
        optional local RZ phases

    Jastrow and RZ are initialized to zero.
    """
    if layers < 1:
        raise ValueError("layers must be >= 1.")

    n_intramode_per_block = n_modes * (n_modals - 1)
    n_jastrow = max(n_modes - 1, 0) * n_modals
    n_rz = n_modes * n_modals if include_rz else 0
    params_per_layer = 2 * n_intramode_per_block + n_jastrow + n_rz

    x0 = np.zeros(layers * params_per_layer, dtype=float)

    def write_intramode_block(
        layer: int, block: Literal["first", "second"], scale: float
    ) -> None:
        base = layer * params_per_layer

        if block == "first":
            offset = 0
        else:
            offset = n_intramode_per_block + n_jastrow

        p = base + offset
        for mode in range(n_modes):
            angles = np.asarray(chain_angles[mode], dtype=float)
            if angles.shape != (n_modals - 1,):
                raise ValueError(
                    f"chain_angles[{mode}] shape {angles.shape}; "
                    f"expected {(n_modals - 1,)}"
                )

            for a in range(n_modals - 1):
                x0[p] = scale * angles[a]
                p += 1

    if placement == "first_block":
        write_intramode_block(layer=0, block="first", scale=1.0)

    elif placement == "split_blocks":
        write_intramode_block(layer=0, block="first", scale=0.5)
        write_intramode_block(layer=0, block="second", scale=0.5)

    elif placement == "all_layers":
        for layer in range(layers):
            write_intramode_block(layer=layer, block="first", scale=1.0 / layers)

    else:
        raise ValueError(f"Unknown placement={placement!r}")

    return x0


def natural_modal_initialization_from_statevector(
    statevector: Statevector | np.ndarray,
    n_modes: int,
    n_modals: int,
    vlucj_layers: int,
    include_rz: bool = True,
    placement: Literal["first_block", "split_blocks", "all_layers"] = "first_block",
) -> NaturalModalInitialization:
    """
    Full natural-modal initialization pipeline:

        statevector
          -> physical coefficient tensor
          -> one-mode RDMs
          -> natural orbitals
          -> dominant natural modal chain angles
          -> VLUCJ x0
    """
    rdms = one_mode_rdms_from_statevector(
        statevector,
        n_modes=n_modes,
        n_modals=n_modals,
    )

    occupations, natural_orbitals = natural_orbitals_from_rdms(rdms)

    chain_angles = tuple(
        chain_angles_from_dominant_vector(vecs[:, 0]) for vecs in natural_orbitals
    )

    x0 = vlucj_x0_from_chain_angles(
        chain_angles=chain_angles,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=vlucj_layers,
        include_rz=include_rz,
        placement=placement,
    )

    return NaturalModalInitialization(
        rdms=tuple(rdms),
        occupations=tuple(occupations),
        natural_orbitals=tuple(natural_orbitals),
        chain_angles=chain_angles,
        vlucj_x0=x0,
    )
