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

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class SampleSet:
    bitstrings: np.ndarray  # shape (n_samples, n_qubits), dtype=np.uint8
    weights: np.ndarray  # shape (n_samples,), dtype=float


def counts_to_samples(counts: dict[str, int]) -> SampleSet:
    bits = []
    weights = []
    for bitstring, count in counts.items():
        bits.append(np.array([int(b) for b in bitstring], dtype=np.uint8))
        weights.append(float(count))
    return SampleSet(bitstrings=np.array(bits), weights=np.array(weights, dtype=float))


def deduplicate_samples(samples: SampleSet) -> SampleSet:
    ctr = Counter()
    for b, w in zip(samples.bitstrings, samples.weights):
        ctr[tuple(int(x) for x in b)] += float(w)
    new_bits = np.array(
        [np.array(k, dtype=np.uint8) for k in ctr.keys()], dtype=np.uint8
    )
    new_weights = np.array(list(ctr.values()), dtype=float)
    return SampleSet(bitstrings=new_bits, weights=new_weights)


def filter_samples(
    samples: SampleSet, predicate: Callable[[np.ndarray], bool]
) -> SampleSet:
    keep = [predicate(b) for b in samples.bitstrings]
    keep = np.array(keep, dtype=bool)
    return SampleSet(samples.bitstrings[keep], samples.weights[keep])


def binary_physicality_predicate(n_modes: int, n_qubits_per_mode: int, vmax: int):
    def _pred(bitrow: np.ndarray) -> bool:
        for m in range(n_modes):
            block = bitrow[m * n_qubits_per_mode : (m + 1) * n_qubits_per_mode]
            idx = int("".join(str(int(x)) for x in block), 2)
            if idx > vmax:
                return False
        return True

    return _pred


def direct_physicality_predicate(n_modes: int, n_modals: int):
    def _pred(bitrow: np.ndarray) -> bool:
        for m in range(n_modes):
            block = bitrow[m * n_modals : (m + 1) * n_modals]
            if int(np.sum(block)) != 1:
                return False
        return True

    return _pred
