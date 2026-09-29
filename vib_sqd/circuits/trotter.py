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

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter


def build_initial_state_from_bitstring(bitstring: str) -> QuantumCircuit:
    """Prepare a computational basis state from a bitstring.

    The leftmost character in the bitstring is treated as the most significant bit.
    Qiskit qubit indexing is little-endian, so we reverse when applying X gates.
    """
    qc = QuantumCircuit(len(bitstring))
    for i, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            qc.x(i)
    return qc


def build_trotter_step(
    hamiltonian: SparsePauliOp,
    dt: float,
    reps: int = 1,
    order: int = 2,
) -> QuantumCircuit:
    """Construct one Suzuki-Trotter time-evolution step."""
    synthesis = SuzukiTrotter(order=order, reps=reps)
    evo = PauliEvolutionGate(hamiltonian, time=dt, synthesis=synthesis)
    qc = QuantumCircuit(hamiltonian.num_qubits)
    qc.append(evo, range(hamiltonian.num_qubits))
    return qc


def build_measured_trotter_family(
    initial_state: QuantumCircuit,
    step_circuit: QuantumCircuit,
    n_steps: int,
) -> list[QuantumCircuit]:
    """Return measured circuits after 1, 2, ..., n_steps Trotter steps."""
    circuits: list[QuantumCircuit] = []

    n_qubits = initial_state.num_qubits
    running = QuantumCircuit(n_qubits, n_qubits)
    running.compose(initial_state, inplace=True)

    for _ in range(n_steps):
        running.compose(step_circuit, inplace=True)
        measured = running.copy()
        measured.measure(range(n_qubits), range(n_qubits))
        circuits.append(measured)

    return circuits
