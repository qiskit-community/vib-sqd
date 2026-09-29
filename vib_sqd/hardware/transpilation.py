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

"""
Utility helpers for hardware-aware transpilation experiments.

Place this file under utility/ in your repo.
Assumes qopt_best_practices is installed.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.providers.backend import BackendV2
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qiskit.visualization import plot_gate_map

# NOTE: qopt_best_practices is an OPTIONAL dependency, installed separately
# (`pip install qopt-best-practices`) -- it is not bundled in any extra here.
# It is only needed by the QAOA/SAT-mapper helper functions below
# (optimize_with_qopt_swap_strategy / optimize_with_sat_mapper). To keep this
# module importable without it — the core hardware path (choose_best_connected_line,
# transpile_default_with_layout, circuit reports, drawings) does not use qopt — the
# qopt imports are done lazily inside those two functions.


@dataclass
class DepthReport:
    label: str
    depth: int
    depth_1q: int
    depth_2q: int
    counts: dict[str, int]
    n_1q_ops: int
    n_2q_ops: int
    size: int
    num_qubits: int


def _oneq_depth(circ: QuantumCircuit) -> int:
    return circ.depth(
        filter_function=lambda inst: getattr(inst.operation, "num_qubits", 0) == 1
    )


def _twoq_depth(circ: QuantumCircuit) -> int:
    return circ.depth(
        filter_function=lambda inst: getattr(inst.operation, "num_qubits", 0) == 2
    )


def _n_qubit_ops(circ: QuantumCircuit, nq: int) -> int:
    total = 0
    for inst, qargs, _ in circ.data:
        if getattr(inst, "num_qubits", len(qargs)) == nq:
            total += 1
    return total


def circuit_report(circ: QuantumCircuit, label: str) -> DepthReport:
    counts = {str(k): int(v) for k, v in circ.count_ops().items()}
    return DepthReport(
        label=label,
        depth=int(circ.depth()),
        depth_1q=int(_oneq_depth(circ)),
        depth_2q=int(_twoq_depth(circ)),
        counts=counts,
        n_1q_ops=_n_qubit_ops(circ, 1),
        n_2q_ops=_n_qubit_ops(circ, 2),
        size=int(circ.size()),
        num_qubits=int(circ.num_qubits),
    )


def print_depth_comparison(*circuits_with_labels: tuple[str, QuantumCircuit]) -> None:
    for label, circ in circuits_with_labels:
        rep = circuit_report(circ, label)
        print(
            f"[{rep.label}] depth={rep.depth}, "
            f"1q_depth={rep.depth_1q}, "
            f"2q_depth={rep.depth_2q}, "
            f"n1q={rep.n_1q_ops}, "
            f"n2q={rep.n_2q_ops}, "
            f"ops={rep.counts}"
        )


def save_depth_reports_json(path: str | Path, reports: list[DepthReport]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "label": r.label,
            "depth": r.depth,
            "depth_1q": r.depth_1q,
            "depth_2q": r.depth_2q,
            "counts": r.counts,
            "n_1q_ops": r.n_1q_ops,
            "n_2q_ops": r.n_2q_ops,
            "size": r.size,
            "num_qubits": r.num_qubits,
        }
        for r in reports
    ]
    path.write_text(json.dumps(payload, indent=2))


def line_swap_strategy(n_qubits: int):
    """
    Create the line swap strategy and alternating edge coloring
    used in the qopt examples.
    """
    swap_strategy = SwapStrategy.from_line(list(range(n_qubits)))
    edge_coloring = {(idx, idx + 1): idx % 2 for idx in range(n_qubits - 1)}
    return swap_strategy, edge_coloring


def choose_best_connected_line(
    backend: BackendV2,
    logical_width: int,
    weight_readout: float = 1.0,
    weight_2q: float = 5.0,
) -> list[int]:
    """
    Simple heuristic to choose a low-error connected line of physical qubits.

    Score = weight_readout * sum(readout_error) + weight_2q * sum(2q_error)
    Lower is better.

    This is a lightweight heuristic, not a replacement for real-time
    characterization or benchmarking.
    """
    coupling = backend.coupling_map
    edges = list(coupling.get_edges()) if coupling is not None else []
    if not edges:
        raise ValueError("Backend does not expose a coupling map.")

    adj: dict[int, set] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    try:
        props = backend.properties()
    except Exception:
        props = None

    def readout_err(q: int) -> float:
        if props is None:
            return 0.0
        try:
            return float(props.readout_error(q))
        except Exception:
            return 0.0

    def edge_err(a: int, b: int) -> float:
        if props is None:
            return 0.0
        for gate in ("cz", "ecr", "cx"):
            for pair in ([a, b], [b, a]):
                try:
                    return float(props.gate_error(gate, pair))
                except Exception:
                    pass
        return 0.0

    # Exhaustive DFS over every simple path of length `logical_width`,
    # starting from every node, is exponential in the coupling graph's
    # branching factor -- fine for a handful of qubits, but for a
    # ~3-regular 100+-qubit device coupling map and logical_width in the
    # 30s-70s this never finishes in practice (observed: 10+ CPU-minutes
    # and still running). Bound the search instead: explore neighbors in
    # best-edge-first order (a strong heuristic on these regular graphs, so
    # the first few thousand explored nodes almost always include a
    # complete, good-scoring path) and cap total DFS calls so a bad
    # starting node can't blow up the search. Falls back to the best
    # *complete* path found within budget; small logical_width still
    # searches the full space (the budget is far larger than what a short
    # line needs), so this changes nothing there.
    max_dfs_calls = 200_000
    calls = 0
    best_path = None
    best_score = math.inf

    def dfs(path: list[int]):
        nonlocal best_path, best_score, calls
        calls += 1
        if calls > max_dfs_calls:
            return
        if len(path) == logical_width:
            score = weight_readout * sum(readout_err(q) for q in path)
            score += weight_2q * sum(
                edge_err(path[i], path[i + 1]) for i in range(len(path) - 1)
            )
            if score < best_score:
                best_score = score
                best_path = path.copy()
            return

        path_set = set(path)
        candidates = [nxt for nxt in adj[path[-1]] if nxt not in path_set]
        candidates.sort(key=lambda nxt: edge_err(path[-1], nxt))
        for nxt in candidates:
            path.append(nxt)
            dfs(path)
            path.pop()
            if calls > max_dfs_calls:
                return

    starts = sorted(adj, key=readout_err)
    for q in starts:
        dfs([q])
        if best_path is not None and calls > max_dfs_calls:
            break

    if best_path is None:
        raise RuntimeError(
            f"Could not find a connected line of length {logical_width}."
        )

    return best_path


def transpile_default_with_layout(
    circuit: QuantumCircuit,
    backend: BackendV2,
    optimization_level: int = 3,
    initial_layout: list[int] | None = None,
) -> QuantumCircuit:
    """
    Standard Qiskit transpilation with an optional fixed initial layout.
    """
    pm = generate_preset_pass_manager(
        target=backend.target,
        optimization_level=optimization_level,
        initial_layout=initial_layout,
    )
    return pm.run(circuit)


def transpile_interaction_aware(
    circuit: QuantumCircuit,
    backend: BackendV2,
    optimization_level: int = 3,
) -> QuantumCircuit:
    """
    Transpile WITHOUT a forced initial_layout, letting Qiskit's own preset
    layout stage (VF2Layout, falling back to SabreLayout) place logical
    qubits based on the circuit's REAL 2-qubit interaction graph, scored by
    the backend's calibrated error rates via `target` -- both interaction-
    aware AND error-aware, unlike choose_best_connected_line's straight
    physical line (which maps logical qubit i -> physical path[i] purely
    positionally, ignoring which logical qubits actually interact).

    Verified on ibm_kingston (2026-09-06) against real ansatz circuits from
    this repo (UVCCSD/VIm-uCJ/CHC, H2O/CH2O at several modal counts): this
    path consistently beats choose_best_connected_line's forced line by
    15-48% fewer CZ gates and 15-48% less depth. Use this instead of
    transpile_default_with_layout(..., initial_layout=choose_best_connected_line(...))
    for every ansatz/cell where the circuit is large enough that layout
    quality matters (i.e. always, for this pipeline's real hardware
    submissions) -- choose_best_connected_line is kept in this module for
    any caller that still wants a bare best-effort connected line (e.g. as
    a diagnostic), but is no longer the recommended default.
    """
    pm = generate_preset_pass_manager(
        target=backend.target,
        optimization_level=optimization_level,
    )
    return pm.run(circuit)


def optimize_with_qopt_swap_strategy(
    cost_op,
    backend: BackendV2,
    reps: int = 1,
):
    """
    Compare default transpilation against qopt swap-strategy transpilation
    for a QAOA-style commuting cost layer.

    Requires the optional ``qopt_best_practices`` dependency (``hardware`` extra).
    """
    from qopt_best_practices.circuit_library import annotated_qaoa_ansatz
    from qopt_best_practices.transpilation import generate_preset_qaoa_pass_manager

    ansatz = annotated_qaoa_ansatz(cost_op, reps=reps)
    ansatz.measure_all()

    swap_strategy, edge_coloring = line_swap_strategy(cost_op.num_qubits)

    qaoa_pm = generate_preset_qaoa_pass_manager(
        backend,
        swap_strategy,
        edge_coloring=edge_coloring,
    )
    optimized_circ = qaoa_pm.run(ansatz)

    pm_ref = generate_preset_pass_manager(
        target=backend.target,
        optimization_level=3,
    )
    ref_optimized = pm_ref.run(ansatz)

    rep_ref = circuit_report(ref_optimized, "default_preset")
    rep_opt = circuit_report(optimized_circ, "qopt_swap_strategy")

    return ref_optimized, optimized_circ, rep_ref, rep_opt


def optimize_with_sat_mapper(
    cost_op,
    reps: int = 1,
    sat_timeout: int = 10,
):
    """
    Apply SATMapper to the cost operator and compare depths before/after.

    Requires the optional ``qopt_best_practices`` dependency (``hardware`` extra).
    """
    from qopt_best_practices.circuit_library import annotated_qaoa_ansatz
    from qopt_best_practices.sat_mapping import SATMapper
    from qopt_best_practices.transpilation.annotated_transpilation_passes import (
        AnnotatedCommuting2qGateRouter,
        AnnotatedPrepareCostLayer,
        AnnotatedSwapToFinalMapping,
        UnrollBoxes,
    )

    swap_strategy, _ = line_swap_strategy(cost_op.num_qubits)

    pre_init = PassManager(
        [
            AnnotatedPrepareCostLayer(),
            AnnotatedCommuting2qGateRouter(swap_strategy),
            AnnotatedSwapToFinalMapping(),
            UnrollBoxes(),
        ]
    )

    ansatz = pre_init.run(annotated_qaoa_ansatz(cost_op, reps=reps))

    sm = SATMapper(timeout=sat_timeout)
    sat_cost_op, sat_map, min_sat_layers = sm.remap_graph_with_sat(
        graph=cost_op,
        swap_strategy=swap_strategy,
    )

    sat_ansatz = pre_init.run(annotated_qaoa_ansatz(sat_cost_op, reps=reps))

    rep_orig = circuit_report(ansatz, "sat_before")
    rep_sat = circuit_report(sat_ansatz, "sat_after")

    return ansatz, sat_ansatz, sat_map, int(min_sat_layers), rep_orig, rep_sat


def save_circuit_drawings(circuit: QuantumCircuit, output_dir: str | Path, stem: str):
    """
    Save .txt and .png circuit drawings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"{stem}.txt").write_text(str(circuit.draw(output="text", fold=-1)))

    try:
        fig = circuit.draw("mpl", fold=-1)
        fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    except Exception:
        pass


def save_gate_map(
    backend: BackendV2, output_dir: str | Path, stem: str = "backend_gate_map"
):
    """
    Save a backend gate map image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig = plot_gate_map(backend, figsize=(6, 6))
        fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    except Exception:
        pass
