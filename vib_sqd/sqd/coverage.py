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
Configuration-coverage metrics for the SQD workflow.

Under sample -> project -> diagonalize (SQD), the operative success criterion is
NOT state fidelity or variational energy, but whether the RIGHT configurations
end up in the sampled/selected set — "finding good determinants" that form a
compact, accurate CI expansion.

Refs:
  - "Exposing a Fatal Flaw in Sample-based Quantum Diagonalization",
    arXiv:2501.07231 (+ JCTC 10.1021/acs.jctc.5c00375): the metric is
    determinant-importance ranking / coverage, not fidelity.
  - "Noise and Configuration Recovery Impact on QSCI", arXiv:2605.23697:
    LUCJ concentrates weight on a few dominant configs; controlled Hilbert-space
    exploration (coverage) matters more than ansatz fidelity.

Compares a reference exact distribution over configurations (|c_exact|^2,
computed in ci_distribution / run_ci_circuit_comparison) against a
sampled/selected set of configurations and reports how much exact weight / how
many exact top-K configs are covered. Configs are labeled by occupation string
"v1v2...".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoverageResult:
    top_k: int
    n_exact_configs: int
    n_sampled_configs: int
    topk_covered: int
    topk_coverage_frac: float
    weight_covered: float
    configs_for_99pct_weight: int
    missing_topk: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "n_exact_configs": self.n_exact_configs,
            "n_sampled_configs": self.n_sampled_configs,
            "topk_covered": self.topk_covered,
            "topk_coverage_frac": self.topk_coverage_frac,
            "weight_covered": self.weight_covered,
            "configs_for_99pct_weight": self.configs_for_99pct_weight,
            "missing_topk": self.missing_topk,
        }


def config_label(occ: Sequence[int]) -> str:
    return "".join(str(int(v)) for v in occ)


def exact_weight_by_config(
    p_exact: np.ndarray, config_labels: Sequence[str]
) -> dict[str, float]:
    p = np.asarray(p_exact, dtype=float)
    return {lbl: float(p[i]) for i, lbl in enumerate(config_labels)}


def sampled_config_set(
    counts_or_probs: dict[str, float], decode: Callable, threshold: float = 0.0
) -> set:
    """Set of config labels present in a sampled distribution.

    ``decode(bitstring) -> occ tuple or None`` (None if non-physical).
    """
    total = sum(counts_or_probs.values()) or 1.0
    out = set()
    for bits, w in counts_or_probs.items():
        if w / total < threshold:
            continue
        occ = decode(bits)
        if occ is None:
            continue
        out.add(config_label(occ))
    return out


def coverage(
    p_exact: np.ndarray,
    config_labels: Sequence[str],
    sampled_labels: Iterable[str],
    top_k: int = 10,
) -> CoverageResult:
    """Coverage of the exact distribution by a sampled/selected config set."""
    w_by_cfg = exact_weight_by_config(p_exact, config_labels)
    sampled = set(sampled_labels)
    order = sorted(w_by_cfg, key=lambda c: w_by_cfg[c], reverse=True)
    topk = order[:top_k]
    covered = [c for c in topk if c in sampled]
    missing = [c for c in topk if c not in sampled]
    weight_covered = float(sum(w_by_cfg[c] for c in sampled if c in w_by_cfg))
    tot = sum(w_by_cfg.values()) or 1.0
    cum, n99 = 0.0, 0
    for c in order:
        cum += w_by_cfg[c] / tot
        n99 += 1
        if cum >= 0.99:
            break
    return CoverageResult(
        top_k=top_k,
        n_exact_configs=len(config_labels),
        n_sampled_configs=len(sampled),
        topk_covered=len(covered),
        topk_coverage_frac=len(covered) / max(1, len(topk)),
        weight_covered=weight_covered,
        configs_for_99pct_weight=n99,
        missing_topk=missing,
    )


def coverage_from_amplitude(
    p_exact: np.ndarray,
    config_labels: Sequence[str],
    p_circuit: np.ndarray,
    sample_threshold: float = 1e-3,
    top_k: int = 10,
) -> CoverageResult:
    """Treat a circuit's |c|^2 (same config order as p_exact) as the sampled set
    by thresholding — which configs the circuit realistically puts weight on."""
    labels = list(config_labels)
    sampled = {labels[i] for i, w in enumerate(p_circuit) if w >= sample_threshold}
    return coverage(p_exact, labels, sampled, top_k=top_k)
