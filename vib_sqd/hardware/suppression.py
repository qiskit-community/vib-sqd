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

from dataclasses import asdict, dataclass
from typing import Any

from qiskit_ibm_runtime import SamplerV2 as Sampler


@dataclass
class RuntimeSuppressionConfig:
    enable_dd: bool = False
    dd_sequence_type: str = "XX"
    enable_pauli_twirling: bool = False
    num_randomizations: int | None = None
    shots_per_randomization: int | None = None
    extra_options: dict[str, Any] | None = None


def apply_sampler_suppression_options(
    sampler: Sampler,
    cfg: RuntimeSuppressionConfig,
) -> Sampler:
    """
    Apply IBM Runtime suppression / mitigation-related options to a SamplerV2 object.
    """
    sampler.options.dynamical_decoupling.enable = bool(cfg.enable_dd)
    if cfg.enable_dd:
        sampler.options.dynamical_decoupling.sequence_type = cfg.dd_sequence_type

    sampler.options.twirling.enable_gates = bool(cfg.enable_pauli_twirling)
    if cfg.enable_pauli_twirling:
        if cfg.num_randomizations is not None:
            sampler.options.twirling.num_randomizations = int(cfg.num_randomizations)
        if cfg.shots_per_randomization is not None:
            sampler.options.twirling.shots_per_randomization = int(
                cfg.shots_per_randomization
            )

    if cfg.extra_options:
        for key, value in cfg.extra_options.items():
            obj = sampler.options
            parts = key.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    return sampler


def summarize_sampler_options(cfg: RuntimeSuppressionConfig):
    return asdict(cfg)
