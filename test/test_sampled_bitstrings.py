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

"""Sampled-bitstring / one-hot bookkeeping: encode/decode round-trip, qubit
endianness, one-hot post-selection, and config-index ordering consistency.
"""

from itertools import product

import numpy as np
import pytest

from vib_sqd.sqd import qsci_augment as qa


def _onehot_bits_big(occ, n_modals):
    bits = []
    for v in occ:
        blk = [0] * n_modals
        blk[v] = 1
        bits.extend(blk)
    return bits


def _decode_onehot(bitstr_big, n_modes, n_modals):
    occ = []
    for m in range(n_modes):
        block = bitstr_big[m * n_modals : (m + 1) * n_modals]
        if block.count("1") != 1:
            return None
        occ.append(block.index("1"))
    return tuple(occ)


@pytest.mark.parametrize("n_modes,n_modals", [(2, 2), (3, 3), (4, 3), (2, 4)])
def test_onehot_encode_decode_roundtrip(n_modes, n_modals):
    for occ in product(range(n_modals), repeat=n_modes):
        bits = _onehot_bits_big(occ, n_modals)
        assert sum(bits) == n_modes  # exactly one per mode
        s = "".join(map(str, bits))
        assert _decode_onehot(s, n_modes, n_modals) == occ


def test_config_to_index_matches_product_order():
    """qsci_augment.config_to_index must equal the position in product(range(M))
    order -- the SAME order the compact modal Hamiltonian rows use."""
    n_modes, n_modals = 4, 3
    occs = list(product(range(n_modals), repeat=n_modes))
    for k, occ in enumerate(occs):
        assert qa.config_to_index(occ, n_modals) == k


def test_config_index_decode_roundtrip():
    n_modes, n_modals = 3, 4

    def decode(idx):
        occ = []
        for _ in range(n_modes):
            occ.append(idx % n_modals)
            idx //= n_modals
        return tuple(reversed(occ))

    for occ in product(range(n_modals), repeat=n_modes):
        assert decode(qa.config_to_index(occ, n_modals)) == occ


def test_endianness_reverse_qargs_contract():
    """A one-hot state prepared with X gates on the big-endian bit positions must,
    after Statevector(...).reverse_qargs(), land on the big-endian index."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    n_modes, n_modals = 2, 3
    nq = n_modes * n_modals
    occ = (1, 2)  # excited config
    bits = _onehot_bits_big(occ, n_modals)  # big-endian bit list
    qc = QuantumCircuit(nq)
    for q, b in enumerate(bits):  # qubit q holds big-endian bit q
        if b:
            qc.x(q)
    sv = np.asarray(Statevector(qc).reverse_qargs().data)
    big_idx = 0
    for b in bits:
        big_idx = (big_idx << 1) | b
    assert np.argmax(np.abs(sv) ** 2) == big_idx
    assert np.abs(sv[big_idx]) ** 2 == pytest.approx(1.0)


def test_onehot_postselection_filters_invalid():
    """Post-selection keeps only valid one-hot-per-mode bitstrings and computes
    the retained fraction correctly."""
    n_modes, n_modals = 2, 2
    # counts keyed by big-endian bitstring; two valid one-hot, one invalid (0011)
    counts = {"1010": 60, "0101": 30, "0011": 10}  # 0011 -> mode0 has zero 1s
    kept = 0
    valid = {}
    for b, c in counts.items():
        occ = _decode_onehot(b, n_modes, n_modals)
        if occ is not None:
            valid[occ] = valid.get(occ, 0) + c
            kept += c
    assert kept == 90
    assert set(valid) == {(0, 0), (1, 1)}  # 1010->(0,0), 0101->(1,1)
    assert kept / sum(counts.values()) == pytest.approx(0.9)
