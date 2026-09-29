#!/usr/bin/env python3

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
Write a mode-subset MidasCpp ``.mop`` from a parsed :class:`MopDocument`.

Used to run MidasCpp VCC on a small active subset of a large molecule (e.g. a
few glycine modes) so the classical VCC — and the downstream statevector VQE —
stay tractable. Selected modes are renumbered ``Q0..Q(k-1)`` and only operator
terms whose factors all lie within the subset are kept (so the subset operator
is self-contained, matching what ``mop_to_forcefield(doc, selected_modes=...)``
produces on the Python side).
"""

from __future__ import annotations

from pathlib import Path

from vib_sqd.classical.mop_forcefield import MopDocument


def write_subset_mop(
    doc: MopDocument,
    selected_modes: tuple[int, ...],
    out_path: str | Path,
) -> Path:
    """Write a subset ``.mop`` keeping only ``selected_modes`` (1-based)."""
    out_path = Path(out_path)
    name_to_global = {n: i for i, n in enumerate(doc.mode_names)}
    selected_global = [m - 1 for m in selected_modes]
    for g in selected_global:
        if g < 0 or g >= doc.n_modes:
            raise ValueError(f"selected mode {g + 1} out of range")
    # Renumber: global mode -> new Q index.
    global_to_new = {g: k for k, g in enumerate(selected_global)}
    new_mode_names = [f"Q{k}" for k in range(len(selected_global))]

    # Filter/renumber terms.
    kept: list[str] = []
    for coeff, factors in doc.terms:
        try:
            parts = []
            for mode_name, power in factors:
                g = name_to_global[mode_name]
                if g not in global_to_new:
                    raise KeyError
                parts.append(f"Q^{power}(Q{global_to_new[g]})")
        except KeyError:
            continue
        kept.append(f" {coeff:.16e} " + "   ".join(parts))

    lines = ["#0 MidasOperator", "#1 ModeNames", " ".join(new_mode_names)]
    if doc.scale_factors is not None:
        sf = [doc.scale_factors[g] for g in selected_global]
        lines += ["#1 ScaleFactors", " ".join(f"{s:.16e}" for s in sf)]
    lines += ["#1 OperatorTerms", *kept, "#0 MidasOperatorEnd", ""]

    out_path.write_text("\n".join(lines))
    return out_path
