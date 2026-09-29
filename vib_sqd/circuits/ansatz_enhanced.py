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

"""
Enhanced ansatz implementations for direct-encoded vibrational Hamiltonians.

Key implementation detail:
    We avoid symbolic ParameterExpression objects inside SparsePauliOp passed
    to PauliEvolutionGate, because some Qiskit versions reject that pattern.

    Instead, every Pauli-string rotation

        exp(-i * angle/2 * P)

    is decomposed manually into basis changes, CZ-based CNOT ladders, and RZ.

Baseline Heron-style gate set:
    RZ, SX, X, CZ

Notes:
    - H is implemented as RZ(pi/2) SX RZ(pi/2), up to global phase.
    - CX is implemented as H(target) CZ H(target).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# ---------------------------------------------------------------------
# Native-gate helpers
# ---------------------------------------------------------------------


def _is_numeric_zero(x, atol: float = 1e-12) -> bool:
    try:
        return abs(float(x)) < atol
    except Exception:
        return False


def append_h_native(qc: QuantumCircuit, q: int) -> None:
    """
    Append H using Heron-style native single-qubit gates.

    H = RZ(pi/2) SX RZ(pi/2), up to global phase.
    """
    qc.rz(np.pi / 2, q)
    qc.sx(q)
    qc.rz(np.pi / 2, q)


def append_cx_via_cz_native(qc: QuantumCircuit, control: int, target: int) -> None:
    """
    Append CX(control, target) using CZ and native H decompositions.

    CX = H(target) CZ(control, target) H(target)
    """
    append_h_native(qc, target)
    qc.cz(control, target)
    append_h_native(qc, target)


def _basis_change_to_z(qc: QuantumCircuit, q: int, pauli: str) -> None:
    """
    Basis change so that a Pauli P measurement/rotation is mapped to Z.

    Standard convention:
        X: H
        Y: Sdg then H
        Z: identity

    Sdg is RZ(-pi/2), S is RZ(pi/2).
    """
    if pauli == "X":
        append_h_native(qc, q)
    elif pauli == "Y":
        qc.rz(-np.pi / 2, q)
        append_h_native(qc, q)
    elif pauli == "Z" or pauli == "I":
        return
    else:
        raise ValueError(f"Unsupported Pauli: {pauli!r}")


def _undo_basis_change_from_z(qc: QuantumCircuit, q: int, pauli: str) -> None:
    """
    Undo _basis_change_to_z.
    """
    if pauli == "X":
        append_h_native(qc, q)
    elif pauli == "Y":
        append_h_native(qc, q)
        qc.rz(np.pi / 2, q)
    elif pauli == "Z" or pauli == "I":
        return
    else:
        raise ValueError(f"Unsupported Pauli: {pauli!r}")


def append_pauli_string_rotation_native(
    qc: QuantumCircuit,
    paulis: dict[int, str],
    angle,
) -> None:
    """
    Append exp(-i * angle/2 * P) for a Pauli string P.

    Uses:
        - RZ/SX for basis changes
        - CZ-based CNOT ladder
        - RZ(angle) on the parity target

    Args:
        qc:
            Circuit to modify.
        paulis:
            Mapping qubit -> "X", "Y", "Z". Identity entries can be omitted.
        angle:
            Numeric or symbolic Qiskit parameter expression.

    Example:
        append_pauli_string_rotation_native(qc, {0: "X", 1: "Y"}, theta)
        implements exp(-i * theta/2 * X0 Y1).
    """
    clean = {int(q): str(p).upper() for q, p in paulis.items() if str(p).upper() != "I"}

    if not clean or _is_numeric_zero(angle):
        return

    active = sorted(clean)

    # Basis change P -> Z-string.
    for q in active:
        _basis_change_to_z(qc, q, clean[q])

    # Entangle parity onto final active qubit.
    target = active[-1]
    controls = active[:-1]

    for c in controls:
        append_cx_via_cz_native(qc, c, target)

    # Z-string rotation.
    qc.rz(angle, target)

    # Uncompute parity.
    for c in reversed(controls):
        append_cx_via_cz_native(qc, c, target)

    # Undo basis change.
    for q in reversed(active):
        _undo_basis_change_from_z(qc, q, clean[q])


def append_givens_rotation_native(
    qc: QuantumCircuit,
    q_ref: int,
    q_exc: int,
    theta,
) -> None:
    """
    Append the real single-excitation/Givens rotation

        exp(theta * (a_exc^† a_ref - a_ref^† a_exc))

    using Heron-native symbolic Pauli-string rotations.

    Generator:
        G = i/2 (X_exc Y_ref - Y_exc X_ref)

    Therefore:
        U = exp(theta G)
          = exp(i theta/2 X_exc Y_ref) exp(-i theta/2 Y_exc X_ref)

    In exp(-i angle/2 P) convention:
        exp(i theta/2 X_exc Y_ref)  -> angle = -theta
        exp(-i theta/2 Y_exc X_ref) -> angle = +theta
    """
    if _is_numeric_zero(theta):
        return

    append_pauli_string_rotation_native(
        qc,
        {q_exc: "X", q_ref: "Y"},
        -theta,
    )
    append_pauli_string_rotation_native(
        qc,
        {q_exc: "Y", q_ref: "X"},
        theta,
    )


# ---------------------------------------------------------------------
# UVCCSD
# ---------------------------------------------------------------------


class UVCCSDAnsatz:
    """
    Unitary Vibrational Coupled Cluster Singles and Doubles.

    Direct encoding only.

    This implementation is symbolic-parameter safe and avoids PauliEvolutionGate
    with parameterized SparsePauliOp.
    """

    def __init__(self, encoding: str = "direct"):
        if encoding != "direct":
            raise NotImplementedError("UVCCSD currently only supports direct encoding.")
        self.encoding = encoding

    @staticmethod
    def count_parameters(n_modes: int, n_modals: int) -> tuple[int, int]:
        n_singles = n_modes * (n_modals - 1)
        n_doubles = (n_modes * (n_modes - 1) // 2) * (n_modals - 1) ** 2
        return n_singles, n_doubles

    def _apply_single_excitation(
        self,
        qc: QuantumCircuit,
        q_ref: int,
        q_exc: int,
        theta,
    ) -> None:
        append_givens_rotation_native(qc, q_ref=q_ref, q_exc=q_exc, theta=theta)

    def _apply_double_excitation(
        self,
        qc: QuantumCircuit,
        q_ref1: int,
        q_exc1: int,
        q_ref2: int,
        q_exc2: int,
        theta,
    ) -> None:
        """
        Apply the Eq. 21 8-term double excitation using symbolic-safe
        Pauli-string rotations.

        Previous PauliEvolutionGate coefficients were:
            coeff_s = -2 * theta * sign

        Pauli rotation convention here:
            exp(-i angle/2 P)

        To implement exp(-i coeff_s P), use:
            angle = 2 * coeff_s = -4 * theta * sign
        """
        if _is_numeric_zero(theta):
            return

        i, j, k, l = q_exc1, q_exc2, q_ref1, q_ref2

        terms = [
            (+1, {i: "X", j: "Y", k: "X", l: "X"}),
            (+1, {i: "Y", j: "X", k: "X", l: "X"}),
            (+1, {i: "Y", j: "Y", k: "X", l: "Y"}),
            (+1, {i: "Y", j: "Y", k: "Y", l: "X"}),
            (-1, {i: "X", j: "X", k: "X", l: "Y"}),
            (-1, {i: "X", j: "X", k: "Y", l: "X"}),
            (-1, {i: "X", j: "Y", k: "Y", l: "Y"}),
            (-1, {i: "Y", j: "X", k: "Y", l: "Y"}),
        ]

        for sign, paulis in terms:
            append_pauli_string_rotation_native(
                qc,
                paulis,
                -0.25 * theta * sign,
            )

    def build(
        self,
        hamiltonian: SparsePauliOp,
        initial_state: QuantumCircuit,
        cfg: dict[str, Any],
    ) -> list[QuantumCircuit]:
        n_modes = int(cfg.get("n_modes", 3))
        n_modals = int(cfg.get("n_modals", cfg.get("vmax", 3) + 1))
        parameterized = bool(cfg.get("parameterized", True))
        measure = bool(cfg.get("measure", False))
        # selected_doubles: optional iterable of (mode1, mode2, exc1, exc2)
        # tuples. When given, ONLY those double-excitation blocks are emitted
        # (a "sparse"/selected UVCCSD), reducing depth. Default None keeps every
        # double, preserving the original behavior and parameter count/order.
        selected_doubles_cfg = cfg.get("selected_doubles", None)
        selected_doubles = (
            None
            if selected_doubles_cfg is None
            else {tuple(int(x) for x in d) for d in selected_doubles_cfg}
        )

        n_qubits = n_modes * n_modals
        qc = QuantumCircuit(n_qubits)
        qc.compose(initial_state, inplace=True)

        n_singles, n_doubles_all = self.count_parameters(n_modes, n_modals)
        # Effective number of doubles actually emitted (all, or the selected set).
        n_doubles = n_doubles_all if selected_doubles is None else len(selected_doubles)

        # ------------------------------------------------------------------
        # Initialize all branch-dependent variables so Pyright knows they exist.
        # ------------------------------------------------------------------
        params: ParameterVector | None = None
        pidx = 0

        t1: np.ndarray | None = None
        t2: np.ndarray | None = None

        if parameterized:
            params = ParameterVector("theta", n_singles + n_doubles)
        else:
            ansatz_params = cfg.get("ansatz_params", {})

            t1_input = ansatz_params.get("t1")
            t2_input = ansatz_params.get("t2")

            if t1_input is None:
                t1 = np.zeros((n_modes, n_modals - 1), dtype=float)
            else:
                t1 = np.asarray(t1_input, dtype=float)

            if t2_input is None:
                t2 = np.zeros(
                    (n_modes, n_modes, n_modals - 1, n_modals - 1),
                    dtype=float,
                )
            else:
                t2 = np.asarray(t2_input, dtype=float)

            expected_t1_shape = (n_modes, n_modals - 1)
            expected_t2_shape = (n_modes, n_modes, n_modals - 1, n_modals - 1)

            if t1.shape != expected_t1_shape:
                raise ValueError(
                    f"t1 has shape {t1.shape}, expected {expected_t1_shape}."
                )

            if t2.shape != expected_t2_shape:
                raise ValueError(
                    f"t2 has shape {t2.shape}, expected {expected_t2_shape}."
                )

        # ------------------------------------------------------------------
        # Singles: modal 0 -> modal a within each mode.
        # ------------------------------------------------------------------
        for mode in range(n_modes):
            base = mode * n_modals
            q_ref = base

            for exc_idx in range(n_modals - 1):
                q_exc = base + exc_idx + 1

                if parameterized:
                    if params is None:
                        raise RuntimeError(
                            "Internal error: params was not initialized."
                        )
                    theta = params[pidx]
                    pidx += 1
                else:
                    if t1 is None:
                        raise RuntimeError("Internal error: t1 was not initialized.")
                    theta = float(t1[mode][exc_idx])

                self._apply_single_excitation(qc, q_ref, q_exc, theta)

        # ------------------------------------------------------------------
        # Doubles: simultaneous modal 0 -> modal a in two different modes.
        # ------------------------------------------------------------------
        for mode1 in range(n_modes):
            for mode2 in range(mode1 + 1, n_modes):
                base1 = mode1 * n_modals
                base2 = mode2 * n_modals

                for exc1_idx in range(n_modals - 1):
                    for exc2_idx in range(n_modals - 1):
                        # Skip doubles not in the selected set (sparse UVCCSD).
                        if (
                            selected_doubles is not None
                            and (mode1, mode2, exc1_idx, exc2_idx)
                            not in selected_doubles
                        ):
                            continue

                        q_ref1 = base1
                        q_exc1 = base1 + exc1_idx + 1
                        q_ref2 = base2
                        q_exc2 = base2 + exc2_idx + 1

                        if parameterized:
                            if params is None:
                                raise RuntimeError(
                                    "Internal error: params was not initialized."
                                )
                            theta = params[pidx]
                            pidx += 1
                        else:
                            if t2 is None:
                                raise RuntimeError(
                                    "Internal error: t2 was not initialized."
                                )
                            theta = float(t2[mode1][mode2][exc1_idx][exc2_idx])

                        self._apply_double_excitation(
                            qc,
                            q_ref1=q_ref1,
                            q_exc1=q_exc1,
                            q_ref2=q_ref2,
                            q_exc2=q_exc2,
                            theta=theta,
                        )

        if parameterized and pidx != n_singles + n_doubles:
            raise RuntimeError(
                f"Internal parameter-count mismatch: used {pidx}, "
                f"expected {n_singles + n_doubles}."
            )

        if measure:
            measured = qc.copy()
            measured.measure_all()
            return [measured]

        return [qc]


# Backward-compatible typo alias.
UVCCSDAnswatz = UVCCSDAnsatz


# ---------------------------------------------------------------------
# Enhanced VUCJ
# ---------------------------------------------------------------------


class EnhancedVUCJAnsatz:
    """
    Parameterized VUCJ-style ansatz using symbolic-safe Heron-native Givens
    rotations and CP Jastrow gates.

    Note:
        CP will transpile to the requested backend basis. If you need strict
        CZ/RZ/SX only at construction time, transpile with basis_gates after
        circuit construction.
    """

    def __init__(self, encoding: str = "direct"):
        if encoding != "direct":
            raise NotImplementedError("EnhancedVUCJ only supports direct encoding.")
        self.encoding = encoding

    def build(
        self,
        hamiltonian: SparsePauliOp,
        initial_state: QuantumCircuit,
        cfg: dict[str, Any],
    ) -> list[QuantumCircuit]:
        n_modes = int(cfg.get("n_modes", 3))
        n_modals = int(cfg.get("n_modals", cfg.get("vmax", 3) + 1))
        n_layers = int(cfg.get("layers", 1))
        measure = bool(cfg.get("measure", False))
        include_rz = bool(cfg.get("include_rz", True))
        # adjacent_jastrow=True restricts the Jastrow block to adjacent mode
        # pairs (i, i+1) only, matching the parameter layout produced by
        # vib_sqd.initialization.vcc_initialization.map_vccsd_to_vlucj_params
        # (which fills t2[mode, mode+1, exc, exc]). Default False keeps the
        # original all-pairs behavior, so existing callers/tests are unchanged.
        adjacent_jastrow = bool(cfg.get("adjacent_jastrow", False))

        n_qubits = n_modes * n_modals

        n_intramode_per_block = n_modes * (n_modals - 1)
        if adjacent_jastrow:
            n_jastrow = max(n_modes - 1, 0) * n_modals
        else:
            n_jastrow = (n_modes * (n_modes - 1) // 2) * n_modals
        n_rz = n_modes * n_modals if include_rz else 0
        params_per_layer = 2 * n_intramode_per_block + n_jastrow + n_rz

        params = ParameterVector("vucj", n_layers * params_per_layer)

        qc = QuantumCircuit(n_qubits)
        qc.compose(initial_state, inplace=True)

        mode_qubits = [
            [m * n_modals + a for a in range(n_modals)] for m in range(n_modes)
        ]

        pidx = 0

        for _layer in range(n_layers):
            # First intramode rotations.
            for mode in range(n_modes):
                qubits = mode_qubits[mode]
                for a in range(n_modals - 1):
                    append_givens_rotation_native(
                        qc,
                        q_ref=qubits[a],
                        q_exc=qubits[a + 1],
                        theta=params[pidx],
                    )
                    pidx += 1

            # Same-modal Jastrow CP gates. adjacent_jastrow restricts to
            # (i, i+1) pairs to match the VCC->VLUCJ parameter mapper layout;
            # otherwise all mode pairs are used.
            if adjacent_jastrow:
                for upper in range(n_modes - 1):
                    lower = upper + 1
                    for modal in range(n_modals):
                        qc.cp(
                            params[pidx],
                            mode_qubits[upper][modal],
                            mode_qubits[lower][modal],
                        )
                        pidx += 1
            else:
                for upper in range(n_modes - 1):
                    for lower in range(upper + 1, n_modes):
                        for modal in range(n_modals):
                            qc.cp(
                                params[pidx],
                                mode_qubits[upper][modal],
                                mode_qubits[lower][modal],
                            )
                            pidx += 1

            # Second intramode rotations.
            for mode in range(n_modes):
                qubits = mode_qubits[mode]
                for a in range(n_modals - 1):
                    append_givens_rotation_native(
                        qc,
                        q_ref=qubits[a],
                        q_exc=qubits[a + 1],
                        theta=params[pidx],
                    )
                    pidx += 1

            if include_rz:
                for mode in range(n_modes):
                    for modal in range(n_modals):
                        qc.rz(params[pidx], mode_qubits[mode][modal])
                        pidx += 1

        if measure:
            measured = qc.copy()
            measured.measure_all()
            return [measured]

        return [qc]


def get_enhanced_ansatz_builder(name: str = "uvccsd", encoding: str = "direct"):
    name = (name or "uvccsd").lower()

    if name in ("uvccsd", "uvcc"):
        return UVCCSDAnsatz(encoding=encoding)

    if name in ("enhanced_vucj", "vucj_enhanced", "evucj", "vlucj"):
        return EnhancedVUCJAnsatz(encoding=encoding)

    raise ValueError(f"Unknown enhanced ansatz: {name}")
