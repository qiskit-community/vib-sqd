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
QSCI+X reference-space augmentation for vibrational SQD.

The sampled/selected configurations from a quantum circuit tend to concentrate on
a few dominant configs and MISS the small-weight "tail" that carries correlation
energy (our VLUCJ finding; and the LUCJ+SQD literature). A cheap, ansatz-agnostic
remedy: treat each sampled configuration as a PARENT and generate its
single/double vibrational excitations classically, add them to the projection
basis, then diagonalize on the enlarged (but still small) space.

This is the vibrational analogue of QSCI+X / reference-space augmentation:
  - QSCI-PT / QSCI+X, PMC12423809: sampled configs become parent configurations
    generating complementary SD/SDT excitations, keeping the space compact.
  - Closed-loop SQD, arXiv:2511.00224: recovered samples + augmentation feed the
    projection/diagonalization.

A vibrational configuration is an occupation tuple (v_1, ..., v_N) with
0 <= v_i <= vmax. A "single" changes one mode's level by any amount within range;
a "double" changes two modes. (We generate all in-range single/double moves from
each parent — the tail configs coupled to a parent by the QFF are exactly these.)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

Config = tuple[int, ...]


def singles(parent: Config, vmax: int) -> set[Config]:
    """All configs differing from parent in exactly ONE mode (any in-range level)."""
    out: set[Config] = set()
    for i in range(len(parent)):
        for v in range(vmax + 1):
            if v == parent[i]:
                continue
            c = list(parent)
            c[i] = v
            out.add(tuple(c))
    return out


def doubles(parent: Config, vmax: int) -> set[Config]:
    """All configs differing from parent in exactly TWO modes (in-range levels)."""
    out: set[Config] = set()
    n = len(parent)
    for i, j in combinations(range(n), 2):
        for vi in range(vmax + 1):
            if vi == parent[i]:
                continue
            for vj in range(vmax + 1):
                if vj == parent[j]:
                    continue
                c = list(parent)
                c[i] = vi
                c[j] = vj
                out.add(tuple(c))
    return out


def augment_configs(
    parents: Iterable[Config],
    vmax: int,
    level: str = "SD",
    max_size: int | None = None,
) -> list[Config]:
    """Augment a set of parent configs with their excitations.

    level: "S" (singles), "D" (doubles), or "SD" (both). Returns the union of
    parents + generated configs, capped at ``max_size`` (parents kept first).
    """
    parents = [tuple(p) for p in parents]
    space: set[Config] = set(parents)
    for p in parents:
        if "S" in level:
            space |= singles(p, vmax)
        if "D" in level:
            space |= doubles(p, vmax)
    ordered = list(dict.fromkeys(parents))  # parents first, dedup, order-stable
    for c in space:
        if c not in set(ordered):
            ordered.append(c)
    if max_size is not None and len(ordered) > max_size:
        ordered = ordered[:max_size]
    return ordered


def config_to_index(occ: Config, n_modals: int) -> int:
    """Compact modal-product index (row-major over product(range(M))), matching
    the ordering used by the modal-space Hamiltonian in modal_basis / the
    ci_circuit analysis. index = sum_i v_i * M^{N-1-i}."""
    idx = 0
    for v in occ:
        idx = idx * n_modals + int(v)
    return idx


def augmented_ground_energy(
    H_compact,
    parent_indices: Sequence[int],
    n_modes: int,
    n_modals: int,
    level: str = "SD",
    max_size: int | None = None,
):
    """Diagonalize H_compact projected onto the augmented basis.

    ``H_compact`` is the modal-product Hamiltonian (dim M^N, product order).
    ``parent_indices`` are compact indices of the sampled configs. Returns
    (ground_energy_compact_units, n_basis, augmented_index_list).
    """
    import numpy as np

    # decode parent indices -> occupation tuples
    def decode(idx: int) -> Config:
        occ = []
        for _ in range(n_modes):
            occ.append(idx % n_modals)
            idx //= n_modals
        return tuple(reversed(occ))

    parents = [decode(i) for i in parent_indices]
    aug = augment_configs(parents, n_modals - 1, level=level, max_size=max_size)
    idxs = sorted({config_to_index(c, n_modals) for c in aug})
    sub = np.asarray(H_compact)[np.ix_(idxs, idxs)]
    w = np.linalg.eigvalsh(sub)
    return float(w[0].real), len(idxs), idxs


def iterative_subspace_growth(
    H_compact,
    parent_indices: Sequence[int],
    n_modes: int,
    n_modals: int,
    max_rounds: int = 6,
    tol_cm: float = 1e-2,
    cm2au: float = 4.556335252912e-06,
    max_size: int | None = 6000,
    weights: dict | None = None,
):
    """SQD-style recovery: start from the SAMPLED parents and repeatedly GROW the
    diagonalization subspace (each round adds single+double excitations of the
    CURRENT subspace, so round k reaches ~k-fold excitations), re-diagonalizing
    after each round.  This is "recover by increasing the diagonalization
    subspace" done as an iterative loop rather than one-shot.

    Truncation to ``max_size`` keeps the MOST IMPORTANT configs, ranked by
    ``weights`` (a dict config_index -> sampled probability).  This is what real
    SQD does: the sample selects WHICH configs enter the subspace (highest
    sampled weight first), then H is diagonalized EXACTLY in that subspace -- the
    noisy weights only rank the set, they do not enter the energy.  Configs
    generated by growth (with no sampled weight) rank after all sampled parents.
    Passing ``weights=None`` falls back to config-index order (only correct when
    the subspace fits under the cap).

    Returns a per-round trace: list of (n_basis, ground_energy_au).  Stops when
    the energy change between rounds is < tol_cm (converged) or the subspace hits
    max_size / max_rounds.
    """
    import numpy as np

    def decode(idx: int):
        occ = []
        for _ in range(n_modes):
            occ.append(idx % n_modals)
            idx //= n_modals
        return tuple(reversed(occ))

    def keep_indices(idx_set):
        """Order config indices for truncation: by descending sampled weight
        (importance), then by index for anything without a weight."""
        idxs = list(idx_set)
        if weights is None:
            idxs.sort()
        else:
            idxs.sort(key=lambda i: (-float(weights.get(i, 0.0)), i))
        if max_size is not None and len(idxs) > max_size:
            idxs = idxs[:max_size]
        return sorted(idxs)  # sort ascending for the H submatrix slice

    Hc = np.asarray(H_compact)
    current = {tuple(decode(i)) for i in parent_indices}
    trace = []
    prev_e = None

    def ground(sub):
        """Lowest eigenvalue: sparse Lanczos for large subspaces, dense for small
        (Lanczos is far cheaper than full eigvalsh when we only need E0)."""
        m = sub.shape[0]
        if m <= 400:
            return float(np.linalg.eigvalsh(sub)[0].real)
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        w = eigsh(
            csr_matrix(sub),
            k=1,
            which="SA",
            maxiter=5000,
            tol=1e-9,
            return_eigenvectors=False,
        )
        return float(w[0].real)

    for _round in range(max_rounds):
        # keep_indices ranks by sampled weight and applies the max_size cap;
        # `current` may be larger than the cap -- the cap decides what is
        # diagonalized this round.
        idxs = keep_indices({config_to_index(c, n_modals) for c in current})
        sub = Hc[np.ix_(idxs, idxs)]
        e = ground(sub)
        trace.append((len(idxs), e))
        if prev_e is not None and abs(e - prev_e) / cm2au < tol_cm:
            break
        prev_e = e
        # grow by UNION: add S+D excitations of the current configs WITHOUT ever
        # dropping what we already have (no max_size here -- truncation is done
        # only by the weight-aware keep_indices above, so high-weight sampled
        # configs like the reference are never discarded by index-based growth).
        grown = augment_configs(list(current), n_modals - 1, level="SD", max_size=None)
        new = current | set(map(tuple, grown))
        if new == current:  # nothing new to add (space saturated)
            break
        current = new
    return trace
