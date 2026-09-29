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

"""choose_best_connected_line must stay fast on a sparse, ~3-regular,
many-qubit coupling graph (a real device's shape) -- the original
exhaustive-DFS-over-every-simple-path implementation was observed to run
for 10+ CPU-minutes without finishing at logical_width in the 30s-70s on a
real 156-qubit backend. Uses a synthetic heavy-hex coupling map (no IBM
credentials needed) since that topology is exactly the failure mode:
sparse and large enough for the search space to explode.
"""

import time

import pytest
from qiskit.transpiler import CouplingMap

from vib_sqd.hardware.transpilation import choose_best_connected_line


class _FakeBackend:
    """Minimal duck-typed BackendV2 stand-in: only .coupling_map and
    .properties() are read by choose_best_connected_line."""

    def __init__(self, coupling_map):
        self.coupling_map = coupling_map

    def properties(self):
        raise NotImplementedError  # exercises the props=None fallback path


@pytest.fixture(scope="module")
def heavy_hex_backend():
    # d=9 -> 193 qubits, degree<=3 -- close to a real ~156-qubit device's
    # size/sparsity (and its slack relative to the widths tested below:
    # a near-Hamiltonian path request, e.g. width close to the graph's
    # total qubit count, is a much harder search than the realistic
    # "request 72 of 156+ qubits" case this function is actually used for).
    return _FakeBackend(CouplingMap.from_heavy_hex(9))


@pytest.mark.parametrize("width", [6, 30, 72])
def test_line_is_valid_and_fast(heavy_hex_backend, width):
    edges = set(heavy_hex_backend.coupling_map.get_edges())
    edges |= {(b, a) for a, b in edges}

    t0 = time.time()
    line = choose_best_connected_line(heavy_hex_backend, logical_width=width)
    elapsed = time.time() - t0

    assert len(line) == width
    assert len(set(line)) == width, "line must not repeat a qubit"
    for i in range(len(line) - 1):
        assert (line[i], line[i + 1]) in edges, "consecutive qubits must be coupled"
    # The old exhaustive-DFS implementation took 10+ minutes and never
    # finished at these widths; a generous bound confirms the fix without
    # being flaky on a slow CI machine.
    assert (
        elapsed < 15.0
    ), f"took {elapsed:.1f}s, expected the bounded search to be fast"


def test_raises_when_width_exceeds_graph_size(heavy_hex_backend):
    with pytest.raises(RuntimeError):
        choose_best_connected_line(heavy_hex_backend, logical_width=10_000)
