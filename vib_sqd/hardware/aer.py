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

"""Local statevector / Aer sampling primitives.

These have no dependency on IBM Runtime and are used both for exact reference
probabilities and for local (simulator) sampling in the runner's ``aer`` mode.
"""

from __future__ import annotations

from qiskit import transpile
from qiskit.quantum_info import Statevector

# qiskit_aer is imported lazily inside run_on_aer_with_exact so this module (and
# the whole vib_sqd.hardware package) imports even when Aer is absent;
# exact_probabilities_from_circuit needs only qiskit's Statevector.


def exact_probabilities_from_circuit(circuit, max_qubits: int = 20) -> dict[str, float]:
    """Exact computational-basis probabilities via statevector simulation.

    Final measurements are removed before building the statevector. Refuses
    circuits wider than ``max_qubits`` (an exact statevector is exponential).
    """
    bare = circuit.remove_final_measurements(inplace=False)
    if bare.num_qubits > max_qubits:
        raise ValueError(
            f"Cannot compute statevector exactly for circuit with "
            f"{bare.num_qubits} qubits (max_qubits={max_qubits})."
        )
    sv = Statevector.from_instruction(bare)
    probs = sv.probabilities_dict()
    return {str(k): float(v) for k, v in probs.items() if float(v) > 0.0}


def run_on_aer_with_exact(circuits, shots: int = 4096):
    """Aer path: transpile for Aer, sample, and compute exact probabilities from
    the SAME transpiled circuits.

    Returns
    -------
    (transpiled_circuits, counts_list, exact_probs_list, exact_probs_source)
    """
    from qiskit_aer import AerSimulator

    backend = AerSimulator()
    compiled = transpile(circuits, backend=backend)

    exact_probs_list = [exact_probabilities_from_circuit(c) for c in compiled]
    exact_probs_source = "same_aer_transpiled_circuits"

    job = backend.run(compiled, shots=shots)
    result = job.result()
    counts_list = [result.get_counts(i) for i in range(len(compiled))]

    return compiled, counts_list, exact_probs_list, exact_probs_source
