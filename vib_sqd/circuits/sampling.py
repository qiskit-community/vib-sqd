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

"""Backward-compatibility shim.

Sampling now lives in the :mod:`vib_sqd.hardware` layer. This module
re-exports the public entry points so existing imports keep working; new code
should import from ``vib_sqd.hardware`` directly.
"""

from __future__ import annotations

from vib_sqd.hardware.aer import (
    exact_probabilities_from_circuit,
    run_on_aer_with_exact,
)
from vib_sqd.hardware.runtime import (
    run_on_ibm_runtime_with_exact,
    sampler_result_to_counts,
)

__all__ = [
    "exact_probabilities_from_circuit",
    "run_on_aer_with_exact",
    "run_on_ibm_runtime_with_exact",
    "sampler_result_to_counts",
]
