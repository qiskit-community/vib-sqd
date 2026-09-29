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

"""IBM Quantum Runtime sampling (SamplerV2).

The real-hardware / Runtime-simulator sampling path. Credentials are read from
the environment only (``QISKIT_IBM_TOKEN``, ``QISKIT_IBM_INSTANCE``) — never
hardcode a token. ``qiskit-ibm-runtime`` is an optional dependency (install via
the ``hardware`` extra); it is imported lazily inside the entry point so the
package imports without it.
"""

from __future__ import annotations

import os
from typing import Any

from .aer import exact_probabilities_from_circuit
from .suppression import (
    RuntimeSuppressionConfig,
    apply_sampler_suppression_options,
)
from .transpilation import choose_best_connected_line


def run_on_ibm_runtime_with_exact(
    circuits,
    shots: int = 4096,
    backend_name: str | None = None,
    use_real: bool = True,
    config: dict | None = None,
) -> tuple[str, str, Any, list[Any], list[dict[str, float]], str]:
    """Sample ``circuits`` on IBM Runtime (SamplerV2, job mode).

    Steps: compute exact probabilities from the ORIGINAL logical measured
    circuits (before ISA transpilation), optionally pick a best-connected
    physical-qubit line, transpile to the backend ISA, apply runtime suppression
    (dynamical decoupling / Pauli twirling), and run.

    Returns
    -------
    (backend_name_used, job_id, primitive_result, isa_circuits,
     exact_probs_list, exact_probs_source)
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    config = config or {}

    if use_real:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=os.environ["QISKIT_IBM_TOKEN"],
            instance=os.environ["QISKIT_IBM_INSTANCE"],
        )
        chosen_backend_name = (
            backend_name
            or config.get("BACKEND_NAME")
            or os.environ.get("BACKEND_NAME", "ibm_marrakesh")
        )
        backend = service.backend(chosen_backend_name)
        backend_name = backend.name
    else:
        service = QiskitRuntimeService()
        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False)
        backend_name = backend.name

    # Exact probabilities from the ORIGINAL logical measured circuits (before ISA).
    exact_probs_list = [exact_probabilities_from_circuit(c) for c in circuits]
    exact_probs_source = "original_logical_measured_circuits"

    # Optional best-line physical-qubit selection.
    initial_layout = None
    if bool(config.get("USE_BEST_LINE", False)):
        initial_layout = choose_best_connected_line(
            backend, logical_width=circuits[0].num_qubits
        )
        print("Chosen best connected physical line:", initial_layout)

    # Backend ISA transpilation.
    pm = generate_preset_pass_manager(
        target=backend.target,
        optimization_level=int(config.get("OPTIMIZATION_LEVEL", 3)),
        initial_layout=initial_layout,
    )
    isa_circuits = [pm.run(c) for c in circuits]

    # Runtime suppression / mitigation toggles.
    sampler = Sampler(mode=backend)
    rt_cfg = config.get("runtime_suppression", {})
    supp_cfg = RuntimeSuppressionConfig(
        enable_dd=bool(rt_cfg.get("enable_dd", False)),
        dd_sequence_type=str(rt_cfg.get("dd_sequence_type", "XX")),
        enable_pauli_twirling=bool(rt_cfg.get("enable_pauli_twirling", False)),
        num_randomizations=rt_cfg.get("num_randomizations"),
        shots_per_randomization=rt_cfg.get("shots_per_randomization"),
    )
    sampler = apply_sampler_suppression_options(sampler, supp_cfg)
    print("Applied runtime suppression config:", supp_cfg)

    job = sampler.run(isa_circuits, shots=shots)
    result = job.result()

    return (
        backend_name,
        job.job_id(),
        result,
        isa_circuits,
        exact_probs_list,
        exact_probs_source,
    )


def sampler_result_to_counts(result) -> list[dict[str, int]]:
    """Convert a SamplerV2 result into a list of plain ``{bitstring: count}`` dicts."""
    all_counts = []
    for pub_result in result:
        data = pub_result.data
        if hasattr(data, "meas"):
            counts = data.meas.get_counts()
        else:
            counts = None
            for attr in dir(data):
                obj = getattr(data, attr)
                if hasattr(obj, "get_counts"):
                    counts = obj.get_counts()
                    break
            if counts is None:
                raise RuntimeError(
                    "Could not find measurement counts in SamplerV2 result."
                )
        all_counts.append({str(k): int(v) for k, v in counts.items()})
    return all_counts
