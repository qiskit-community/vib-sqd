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

"""Hardware execution layer: local Aer sampling and IBM Quantum Runtime sampling,
plus backend transpilation and error-suppression helpers.

Public entry points:
    exact_probabilities_from_circuit, run_on_aer_with_exact   (aer.py)
    run_on_ibm_runtime_with_exact, sampler_result_to_counts   (runtime.py)
    RuntimeSuppressionConfig, apply_sampler_suppression_options (suppression.py)
    choose_best_connected_line, transpile_default_with_layout  (transpilation.py)
"""

from __future__ import annotations

from .aer import exact_probabilities_from_circuit, run_on_aer_with_exact
from .runtime import run_on_ibm_runtime_with_exact, sampler_result_to_counts
from .suppression import (
    RuntimeSuppressionConfig,
    apply_sampler_suppression_options,
)
from .transpilation import (
    choose_best_connected_line,
    transpile_default_with_layout,
)

__all__ = [
    "RuntimeSuppressionConfig",
    "apply_sampler_suppression_options",
    "choose_best_connected_line",
    "exact_probabilities_from_circuit",
    "run_on_aer_with_exact",
    "run_on_ibm_runtime_with_exact",
    "sampler_result_to_counts",
    "transpile_default_with_layout",
]
