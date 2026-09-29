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

from typing import Any, Protocol

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from vib_sqd.circuits.trotter import build_measured_trotter_family, build_trotter_step

# Import optimized ansatz builders
try:
    from vib_sqd.circuits.ansatz_optimized import OptimizedVUCJAnsatz

    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False

try:
    # Reuse the lightweight resource templates available in the repo for CHC/VLUCJ prototypes
    from vib_sqd.analysis.resource_estimation import (
        build_binary_chc_circuit,
        build_binary_vlucj_circuit,
        build_direct_chc_circuit,
        build_direct_vlucj_circuit,
    )
except Exception:
    build_direct_chc_circuit = None  # type: ignore
    build_binary_chc_circuit = None  # type: ignore
    build_direct_vlucj_circuit = None  # type: ignore
    build_binary_vlucj_circuit = None  # type: ignore


class AnsatzBuilder(Protocol):
    def build(
        self,
        hamiltonian: SparsePauliOp,
        initial_state: QuantumCircuit,
        cfg: dict[str, Any],
    ) -> list[QuantumCircuit]:
        """Construct measured circuits for the requested ansatz.

        - `hamiltonian` may be unused by some ansatz builders (e.g., variational hardware-efficient)
        - `initial_state` is a prepared QuantumCircuit describing the reference state
        - `cfg` contains keys from the `circuit` section of the run config
        """


class TrotterAnsatz:
    """Thin wrapper that produces the current Trotterized evolution circuits.

    This preserves existing behaviour while exposing a pluggable interface.
    """

    def build(
        self,
        hamiltonian: SparsePauliOp,
        initial_state: QuantumCircuit,
        cfg: dict[str, Any],
    ) -> list[QuantumCircuit]:
        dt = float(cfg.get("dt", 10.0))
        reps = int(cfg.get("reps", 1))
        order = int(cfg.get("order", 2))
        n_steps = int(cfg.get("n_steps", 1))

        step = build_trotter_step(hamiltonian, dt=dt, reps=reps, order=order)
        return build_measured_trotter_family(
            initial_state=initial_state, step_circuit=step, n_steps=n_steps
        )


class CHCAnsatz:
    """Prototype CHC ansatz that delegates to the repo's resource templates when available.

    This builder returns measured circuits and accepts the same `cfg` keys as Trotter where applicable.
    """

    def __init__(self, encoding: str | None = None):
        self.encoding = encoding or "direct"

    def build(
        self,
        hamiltonian: SparsePauliOp,
        initial_state: QuantumCircuit,
        cfg: dict[str, Any],
    ) -> list[QuantumCircuit]:
        n_layers = int(cfg.get("layers", cfg.get("n_layers", 4)))
        if self.encoding == "direct":
            if build_direct_chc_circuit is None:
                raise RuntimeError("CHC resource templates are unavailable")
            qc = build_direct_chc_circuit(
                n_modes=int(cfg.get("n_modes", 3)),
                n_modals=int(cfg.get("n_modals", cfg.get("vmax", 3) + 1)),
                n_layers=n_layers,
            )
        else:
            if build_binary_chc_circuit is None:
                raise RuntimeError("CHC resource templates are unavailable")
            qc = build_binary_chc_circuit(
                n_modes=int(cfg.get("n_modes", 3)),
                n_levels=int(cfg.get("n_levels", cfg.get("vmax", 3) + 1)),
                n_layers=n_layers,
            )

        # Compose initial_state (if non-empty) and measure
        full = QuantumCircuit(qc.num_qubits, qc.num_qubits)
        full.compose(initial_state, inplace=True)
        full.compose(qc, inplace=True)
        full.measure(range(full.num_qubits), range(full.num_qubits))
        return [full]


class VUCJAnsatz:
    """Builder for the Vibrational UCJ (VUCJ) style ansatz.

    The implementation here builds symmetry-preserving layers using XX+YY (exchange)
    gates between adjacent modal qubits and diagonal CP rotations for Jastrow terms.
    This follows the design notes in the repo's resource_estimation.py.
    """

    def __init__(self, encoding: str | None = "direct"):
        self.encoding = encoding or "direct"

    def build(
        self,
        hamiltonian: SparsePauliOp,
        initial_state: QuantumCircuit,
        cfg: dict[str, Any],
    ) -> list[QuantumCircuit]:
        n_layers = int(cfg.get("layers", 4))
        n_modes = int(cfg.get("n_modes", 3))
        ansatz_params = cfg.get("ansatz_params", {}) if cfg is not None else {}
        # Optional parameter maps to set final orbital rotations and Jastrow angles
        r_f_angles = ansatz_params.get("r_f_angles")
        jastrow_angles = ansatz_params.get("jastrow_angles")
        if self.encoding == "direct":
            n_modals = int(cfg.get("n_modals", cfg.get("vmax", 3) + 1))
            n_qubits = n_modes * n_modals
            qc = QuantumCircuit(n_qubits)

            # compose reference state
            qc.compose(initial_state, inplace=True)

            # build layers
            from qiskit.circuit.library import XXPlusYYGate

            for _ in range(n_layers):
                # mode-local Givens/XX+YY rotations
                for m in range(n_modes):
                    base = m * n_modals
                    for a in range(n_modals - 1):
                        qc.append(XXPlusYYGate(0.2, 0.0), [base + a, base + a + 1])

                # diagonal Jastrow between modes (pairwise CP on modal pairs)
                for upper in range(n_modes - 1):
                    for lower in range(upper + 1, n_modes):
                        for a in range(n_modals):
                            # allow overriding of jastrow angles via ansatz params
                            angle = 0.12
                            if jastrow_angles is not None:
                                if isinstance(jastrow_angles, (int, float)):
                                    angle = float(jastrow_angles)
                                else:
                                    try:
                                        angle = float(jastrow_angles[upper][lower][a])
                                    except Exception:
                                        pass
                            qc.cp(angle, upper * n_modals + a, lower * n_modals + a)

                # optional second rotation block
                for m in range(n_modes):
                    base = m * n_modals
                    for a in range(n_modals - 1):
                        qc.append(XXPlusYYGate(0.2, 0.0), [base + a, base + a + 1])

            # optionally add final orbital rotation R_f (placeholder using same local rotations)
            for m in range(n_modes):
                base = m * n_modals
                for a in range(1, n_modals):
                    # allow overriding R_f angles via ansatz params
                    theta = 0.15
                    if r_f_angles is not None:
                        try:
                            theta = float(r_f_angles[m][a - 1])
                        except Exception:
                            pass
                    qc.append(XXPlusYYGate(theta, 0.0), [base + 0, base + a])

            # If UVCCSD initialization is requested, try to apply provided t1 amplitudes
            uvcc_cfg = ansatz_params.get("uvccsd_init")
            if uvcc_cfg:
                # Prefer user-supplied t1 amplitudes (explicit), otherwise attempt Qiskit Nature
                t1 = uvcc_cfg.get("t1")
                if t1 is not None:
                    # Expect t1 as a nested sequence: t1[mode][modal_index]
                    for m in range(min(n_modes, len(t1))):
                        base = m * n_modals
                        for a in range(1, min(n_modals, len(t1[m]))):
                            theta = float(t1[m][a])
                            qc.append(XXPlusYYGate(theta, 0.0), [base + 0, base + a])
                elif bool(uvcc_cfg.get("use_qiskit_nature", False)):
                    try:
                        # Attempt to construct an initial UVCC/UVCCSD circuit using Qiskit Nature
                        from qiskit_nature.second_q.drivers import GaussianForcesDriver
                        from qiskit_nature.second_q.mappers import DirectMapper
                        from qiskit_nature.second_q.problems import HarmonicBasis

                        gfile = uvcc_cfg.get("gaussian_logfile")
                        num_modals = uvcc_cfg.get(
                            "num_modals_per_mode", [n_modals] * n_modes
                        )
                        trunc = int(uvcc_cfg.get("truncation_order", 2))

                        driver = GaussianForcesDriver(logfile=str(gfile))
                        basis = HarmonicBasis(num_modals)
                        vib_problem = driver.run(basis=basis)
                        vib_problem.hamiltonian.truncation_order = trunc

                        main_op, _ = vib_problem.second_q_ops()
                        qubit_op = DirectMapper().map(main_op)

                        # Try to import UVCC/UVCCSD ansatz class
                        try:
                            from qiskit_nature.second_q.circuit.library import UVCCSD

                            uvcc_cls = UVCCSD
                        except Exception:
                            try:
                                from qiskit_nature.second_q.circuit.library.ansatzes.uvccsd import (
                                    UVCCSD,
                                )

                                uvcc_cls = UVCCSD
                            except Exception:
                                uvcc_cls = None

                        if uvcc_cls is not None:
                            # Build a parameterized UVCCSD circuit and append it as R_f
                            uvcc = uvcc_cls(qubit_op.num_qubits)
                            qc.compose(uvcc, inplace=True)
                        else:
                            print(
                                "UVCCSD class not found in qiskit_nature; falling back to VSCF reference rotations"
                            )
                            from qiskit_nature.second_q.circuit.library.initial_states import (
                                VSCF,
                            )

                            vscf = VSCF(num_modals)
                            qc.compose(vscf, inplace=True)
                    except Exception as e:
                        print(
                            "Failed to build UVCCSD initialization from Qiskit Nature:",
                            e,
                        )
                        print("Proceeding with default R_f placeholder")

            # measured circuit — ensure classical bits exist by using measure_all()
            meas = qc.copy()
            meas.measure_all()
            return [meas]

        else:
            # For binary encoding we provide a VLUCJ-like template using modal logical blocks
            # Fallback to existing resource template if available
            if build_binary_vlucj_circuit is not None:
                qc = build_binary_vlucj_circuit(
                    n_modes=n_modes,
                    n_levels=int(cfg.get("n_levels", cfg.get("vmax", 3) + 1)),
                    n_layers=n_layers,
                )
                full = QuantumCircuit(qc.num_qubits, qc.num_qubits)
                full.compose(initial_state, inplace=True)
                full.compose(qc, inplace=True)
                full.measure(range(full.num_qubits), range(full.num_qubits))
                return [full]

            # generic minimal binary VLUCJ fallback
            raise RuntimeError("Binary VUCJ template not available")


def get_ansatz_builder(
    name: str = "trotter", encoding: str | None = None, use_optimized: bool = False
) -> AnsatzBuilder:
    """
    Get an ansatz builder by name.

    Args:
        name: Ansatz type ("trotter", "chc", "vucj")
        encoding: Encoding type ("direct" or "binary")
        use_optimized: If True, use hardware-optimized implementations when available

    Returns:
        Ansatz builder instance
    """
    name = (name or "trotter").lower()

    if name in ("trotter", "time-evolution"):
        return TrotterAnsatz()

    if name in ("chc",):
        return CHCAnsatz(encoding=encoding)

    if name in ("vucj", "vlucj", "vucj_ansatz"):
        # Use optimized version if requested and available
        if use_optimized and OPTIMIZED_AVAILABLE:
            return OptimizedVUCJAnsatz(encoding=encoding or "direct")
        return VUCJAnsatz(encoding=encoding)

    # default
    return TrotterAnsatz()
