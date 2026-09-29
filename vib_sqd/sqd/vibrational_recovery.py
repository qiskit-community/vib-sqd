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

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vib_sqd.sqd.simple_projection import encoded_to_dense_matrix


@dataclass
class VibrationalSQDLoopResult:
    converged: bool
    best_energy_cm_inv: float
    energies_by_iter: list[float]
    selected_basis_bitstrings: list[str]
    post_sqd_probabilities: dict[str, float]
    spectrum_df: pd.DataFrame
    recovered_probabilities: dict[str, float]
    iterations: int


def _bitstring_to_index(bitstring: str) -> int:
    return int(bitstring, 2)


def _modal_indices_from_onehot(
    bitstring: str, n_modes: int, n_modals: int
) -> list[int]:
    out: list[int] = []
    for mode in range(n_modes):
        block = bitstring[mode * n_modals : (mode + 1) * n_modals]
        if block.count("1") != 1:
            raise ValueError(
                f"Non-one-hot block in bitstring {bitstring}: block={block}"
            )
        out.append(block.index("1"))
    return out


def _onehot_from_modal_indices(indices: list[int], n_modals: int) -> str:
    chunks: list[str] = []
    for idx in indices:
        block = ["0"] * n_modals
        block[int(idx)] = "1"
        chunks.append("".join(block))
    return "".join(chunks)


def _is_physical_onehot(bitstring: str, n_modes: int, n_modals: int, vmax: int) -> bool:
    if len(bitstring) != n_modes * n_modals:
        return False
    for mode in range(n_modes):
        block = bitstring[mode * n_modals : (mode + 1) * n_modals]
        if block.count("1") != 1:
            return False
        if block.index("1") > vmax:
            return False
    return True


def _repair_block_by_occupancy(
    block: str,
    occ: np.ndarray,
    rng: np.random.Generator,
    vmax: int,
    randomize: bool,
) -> int:
    ones = [i for i, bit in enumerate(block) if bit == "1" and i <= vmax]

    if len(ones) == 1:
        return int(ones[0])

    valid_occ = np.asarray(occ[: vmax + 1], dtype=float)
    valid_occ = np.clip(valid_occ, 1e-12, None)
    valid_occ = valid_occ / float(np.sum(valid_occ))

    if randomize:
        return int(rng.choice(np.arange(vmax + 1), p=valid_occ))

    return int(np.argmax(valid_occ))


def repair_direct_onehot_bitstring(
    bitstring: str,
    avg_modal_occupancies: np.ndarray,
    n_modes: int,
    n_modals: int,
    vmax: int,
    rng: np.random.Generator,
    randomize: bool = True,
) -> str:
    if len(bitstring) != n_modes * n_modals:
        raise ValueError(
            f"Expected bitstring length {n_modes * n_modals}, got {len(bitstring)}: {bitstring}"
        )

    modal_indices: list[int] = []
    for mode in range(n_modes):
        block = bitstring[mode * n_modals : (mode + 1) * n_modals]
        idx = _repair_block_by_occupancy(
            block=block,
            occ=avg_modal_occupancies[mode],
            rng=rng,
            vmax=vmax,
            randomize=randomize,
        )
        modal_indices.append(idx)

    return _onehot_from_modal_indices(modal_indices, n_modals)


def recover_direct_onehot_distribution(
    counts_or_probs: dict[str, int | float],
    avg_modal_occupancies: np.ndarray,
    n_modes: int,
    n_modals: int,
    vmax: int,
    seed: int | None = None,
    randomize: bool = True,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    recovered: dict[str, float] = {}

    for bitstring, weight in counts_or_probs.items():
        b = str(bitstring).replace(" ", "")
        w = float(weight)
        if w <= 0.0:
            continue

        repaired = repair_direct_onehot_bitstring(
            b,
            avg_modal_occupancies=avg_modal_occupancies,
            n_modes=n_modes,
            n_modals=n_modals,
            vmax=vmax,
            rng=rng,
            randomize=randomize,
        )
        recovered[repaired] = recovered.get(repaired, 0.0) + w

    total = float(sum(recovered.values()))
    if total <= 0.0:
        raise ValueError("Recovery produced zero total probability/count.")

    return {bitstring: float(weight) / total for bitstring, weight in recovered.items()}


def sample_basis_from_distribution(
    probabilities: dict[str, float],
    basis_size: int,
    rng: np.random.Generator,
    always_include: list[str] | None = None,
) -> list[str]:
    keys = list(probabilities.keys())
    p = np.array([float(probabilities[key]) for key in keys], dtype=float)
    p = p / np.sum(p)

    basis: list[str] = []
    if always_include:
        for bitstring in always_include:
            if bitstring in probabilities and bitstring not in basis:
                basis.append(bitstring)

    target = min(int(basis_size), len(keys))
    while len(basis) < target:
        sampled = str(rng.choice(keys, p=p))
        if sampled not in basis:
            basis.append(sampled)

    return basis


def diagonalize_basis(
    dense_h: np.ndarray,
    basis: list[str],
    n_roots: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.array(
        [_bitstring_to_index(bitstring) for bitstring in basis], dtype=int
    )
    h_sub = dense_h[np.ix_(indices, indices)]
    evals, evecs = np.linalg.eigh(h_sub)
    order = np.argsort(evals)
    return evals[order][:n_roots], evecs[:, order][:, :n_roots]


def occupancies_from_ground_state(
    basis: list[str],
    ground_vec: np.ndarray,
    n_modes: int,
    n_modals: int,
    vmax: int,
    smoothing: float = 1e-6,
) -> np.ndarray:
    occ = np.full((n_modes, n_modals), float(smoothing), dtype=float)
    probs = np.abs(np.asarray(ground_vec, dtype=complex)) ** 2

    for bitstring, prob in zip(basis, probs):
        modal_indices = _modal_indices_from_onehot(bitstring, n_modes, n_modals)
        for mode, idx in enumerate(modal_indices):
            if idx <= vmax:
                occ[mode, idx] += float(prob)

    for mode in range(n_modes):
        total = float(np.sum(occ[mode, : vmax + 1]))
        occ[mode, : vmax + 1] /= total
        if vmax + 1 < n_modals:
            occ[mode, vmax + 1 :] = 0.0

    return occ


def occupancies_from_raw_counts(
    counts: dict[str, int | float],
    n_modes: int,
    n_modals: int,
    vmax: int,
    smoothing: float = 1e-6,
) -> np.ndarray:
    """Per-mode occupation marginal from a RAW (possibly non-one-hot) bitstring-
    weight dict, e.g. unrepaired hardware counts. Unlike
    occupancies_from_ground_state (which requires every block to be exactly
    one-hot and raises ValueError otherwise via _modal_indices_from_onehot),
    this tolerates malformed blocks via a fractional-split policy: a block with
    k>=1 bits set at indices i_1..i_k (each <= vmax) contributes weight/k to
    each occ[mode, i_j]; a block with 0 bits set within [0, vmax] contributes
    nothing for that mode (skipped, not an error). Degrades to an exact one-hot
    decode when the input is already well-formed (k always 1), so this is a
    strict superset of occupancies_from_ground_state's per-bitstring logic and
    safe to use on both "before" (noisy) and "after" (recovered, well-formed)
    populations."""
    occ = np.full((n_modes, n_modals), float(smoothing), dtype=float)
    for bitstring, weight in counts.items():
        b = str(bitstring).replace(" ", "")
        w = float(weight)
        if w <= 0.0 or len(b) != n_modes * n_modals:
            continue
        for mode in range(n_modes):
            block = b[mode * n_modals : (mode + 1) * n_modals]
            ones = [i for i, bit in enumerate(block) if bit == "1" and i <= vmax]
            if not ones:
                continue
            share = w / len(ones)
            for idx in ones:
                occ[mode, idx] += share

    for mode in range(n_modes):
        total = float(np.sum(occ[mode, : vmax + 1]))
        occ[mode, : vmax + 1] /= total
        if vmax + 1 < n_modals:
            occ[mode, vmax + 1 :] = 0.0

    return occ


def initial_uniform_modal_occupancies(
    n_modes: int,
    n_modals: int,
    vmax: int,
) -> np.ndarray:
    occ = np.zeros((n_modes, n_modals), dtype=float)
    occ[:, : vmax + 1] = 1.0 / float(vmax + 1)
    return occ


def run_vibrational_sqd_recovery_loop(
    encoded_hamiltonian: Any,
    counts_or_probs: dict[str, int | float],
    n_modes: int,
    n_modals: int,
    vmax: int,
    n_roots: int = 5,
    max_iter: int = 100,
    conv_tol: float = 1e-3,
    basis_size: int = 64,
    n_batches: int = 16,
    seed: int | None = 1234,
    init_bitstring: str | None = None,
    resample_if_not_converged: bool = True,
    max_resample_attempts: int = 3,
    randomize_repair: bool = True,
) -> VibrationalSQDLoopResult:
    dense_h = encoded_to_dense_matrix(encoded_hamiltonian)
    dim = dense_h.shape[0]
    n_qubits = n_modes * n_modals

    if 2**n_qubits != dim:
        raise ValueError(
            f"Hamiltonian dimension mismatch: expected {2**n_qubits}, got {dim}."
        )

    rng_master = np.random.default_rng(seed)
    attempts = max_resample_attempts if resample_if_not_converged else 1

    best_global_energy = float("inf")
    best_global_basis: list[str] = []
    best_global_vec: np.ndarray | None = None
    best_global_evals: np.ndarray | None = None
    best_recovered: dict[str, float] = {}

    final_energies_by_iter: list[float] = []
    converged = False
    total_iterations = 0

    for _attempt in range(attempts):
        occ = initial_uniform_modal_occupancies(n_modes, n_modals, vmax)
        prev_energy: float | None = None
        energies_by_iter: list[float] = []

        for _iter in range(max_iter):
            total_iterations += 1
            iter_seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(iter_seed)

            recovered = recover_direct_onehot_distribution(
                counts_or_probs=counts_or_probs,
                avg_modal_occupancies=occ,
                n_modes=n_modes,
                n_modals=n_modals,
                vmax=vmax,
                seed=iter_seed,
                randomize=randomize_repair,
            )

            always_include: list[str] = []
            if init_bitstring is not None and _is_physical_onehot(
                init_bitstring,
                n_modes,
                n_modals,
                vmax,
            ):
                always_include.append(init_bitstring)

            batch_best_energy = float("inf")
            batch_best_basis: list[str] = []
            batch_best_vec: np.ndarray | None = None
            batch_best_evals: np.ndarray | None = None

            for _ in range(n_batches):
                basis = sample_basis_from_distribution(
                    recovered,
                    basis_size=basis_size,
                    rng=rng,
                    always_include=always_include,
                )
                evals, evecs = diagonalize_basis(dense_h, basis, n_roots=n_roots)

                if float(evals[0]) < batch_best_energy:
                    batch_best_energy = float(evals[0])
                    batch_best_basis = basis
                    batch_best_vec = evecs[:, 0]
                    batch_best_evals = evals

            if batch_best_vec is None or batch_best_evals is None:
                raise RuntimeError("No SQD batch was diagonalized.")

            occ = occupancies_from_ground_state(
                batch_best_basis,
                batch_best_vec,
                n_modes=n_modes,
                n_modals=n_modals,
                vmax=vmax,
            )

            energies_by_iter.append(batch_best_energy)

            if batch_best_energy < best_global_energy:
                best_global_energy = batch_best_energy
                best_global_basis = list(batch_best_basis)
                best_global_vec = np.array(batch_best_vec, copy=True)
                best_global_evals = np.array(batch_best_evals, copy=True)
                best_recovered = dict(recovered)

            if prev_energy is not None:
                if abs(batch_best_energy - prev_energy) < conv_tol:
                    converged = True
                    final_energies_by_iter = energies_by_iter
                    break

            prev_energy = batch_best_energy

        final_energies_by_iter = energies_by_iter
        if converged:
            break

    if best_global_vec is None or best_global_evals is None:
        raise RuntimeError("SQD recovery loop failed to produce a result.")

    gs_probs = np.abs(best_global_vec) ** 2
    post_probs = {
        bitstring: float(prob) for bitstring, prob in zip(best_global_basis, gs_probs)
    }

    rows: list[dict[str, Any]] = []
    for root, energy in enumerate(best_global_evals[:n_roots]):
        rows.append(
            {
                "state": f"root_{root}",
                "stage": "post_sqd_recovery_loop",
                "energy_cm_inv": float(energy),
            }
        )

    return VibrationalSQDLoopResult(
        converged=bool(converged),
        best_energy_cm_inv=float(best_global_energy),
        energies_by_iter=[float(x) for x in final_energies_by_iter],
        selected_basis_bitstrings=best_global_basis,
        post_sqd_probabilities=post_probs,
        spectrum_df=pd.DataFrame(rows),
        recovered_probabilities=best_recovered,
        iterations=int(total_iterations),
    )
