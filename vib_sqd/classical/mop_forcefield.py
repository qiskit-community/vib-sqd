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
Parse a MidasCpp ``.mop`` operator file into a repo ``VibrationalForceField``.

Why this exists
---------------
The collaborator's ``.mop`` files are the authoritative vibrational force field
(atomic units, dimensionless normal coordinates ``Q``, with an optional
``#1 ScaleFactors`` block). The repo's direct-encoding Hamiltonian builder
(:func:`vib_sqd.encoding.direct.build_direct_encoded_vibrational_hamiltonian`)
consumes a ``VibrationalForceField`` of ``(omega, ForceTerm(coeff, modes, powers))``
in a *dimensionless* coordinate convention where ``q = (a + a†)/sqrt(2)``.

Verified coordinate mapping (see plan / numeric check)
------------------------------------------------------
MidasCpp's dimensionless ``Q`` is *the same reduced normal coordinate* as the
repo's ``q = (a + a†)/sqrt(2)``. The ``.mop`` harmonic term is written as
``0.5 * omega_au * Q^2``; matrix elements of ``0.5*omega*(a+a†)^2/2`` equal
``omega*(n+1/2)``. Hence a ``.mop`` term::

    c_au * Q^k(Qi) Q^m(Qj) ...

maps *directly* to::

    ForceTerm(coeff=c_au, modes=(i+1, j+1, ...), powers=(k, m, ...))

with no extra scaling, and ``omega_i_au = 2 * coeff(Q^2(Qi))``. Building the
Hamiltonian with ``harmonic_source="polynomial_quadratic"`` then reproduces the
``.mop`` operator term-for-term, so the VCC amplitudes and the circuit
Hamiltonian come from exactly the same force field.

Usage
-----
    from mop_to_forcefield import parse_mop, mop_to_forcefield
    doc = parse_mop("inputs/h2o_dimless_qff_numeric.mop")
    ff = mop_to_forcefield(doc)                 # all modes
    ff = mop_to_forcefield(doc, selected_modes=(1, 2, 3, 4))  # subset (1-based)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vib_sqd.classical.forcefield_types import (
    ForceTerm,
    VibrationalForceField,
)

CM2AU = 4.556335252912000e-06

_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"
# One factor like "Q^2(Q0)" or "Q^1(Q12)". Power defaults to 1 if "^p" absent.
_FACTOR_RE = re.compile(r"Q(?:\^(\d+))?\((Q\d+)\)")


@dataclass
class MopDocument:
    """Structured contents of a parsed ``.mop`` file."""

    mode_names: list[str]  # e.g. ["Q0", "Q1", ...] in file order
    scale_factors: list[float] | None  # sqrt(omega_cm * CM2AU) per mode, or None
    # Each term: (coeff_au, [(mode_name, power), ...])
    terms: list[tuple[float, list[tuple[str, int]]]]

    @property
    def n_modes(self) -> int:
        return len(self.mode_names)


def _strip_comment(line: str) -> str:
    """Remove a trailing ``// ...`` comment."""
    idx = line.find("//")
    return line[:idx] if idx != -1 else line


def _is_tag(token: str, name: str) -> bool:
    """Match ``#1 ModeNames`` and the compact ``#1ModeNames`` variant."""
    compact = token.replace(" ", "").upper()
    return compact == f"#1{name}".upper() or compact == f"#0{name}".upper()


def parse_mop(path: str | Path) -> MopDocument:
    """Parse a ``.mop`` file into a :class:`MopDocument`.

    Handles both spaced (``#1 ModeNames``) and compact (``#1ModeNames``) tags,
    ``//`` comments, and terms spread across the ``#1 OperatorTerms`` section.
    """
    text = Path(path).read_text()

    mode_names: list[str] = []
    scale_factors: list[float] | None = None
    terms: list[tuple[float, list[tuple[str, int]]]] = []

    section: str | None = None

    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue

        compact = line.replace(" ", "").upper()

        # Section headers (#0/#1 ...). A line that starts with '#' is a tag.
        if line.startswith("#"):
            if compact.startswith("#1MODENAMES"):
                section = "modenames"
            elif compact.startswith("#1SCALEFACTORS"):
                section = "scalefactors"
            elif compact.startswith("#1OPERATORTERMS"):
                section = "operatorterms"
            elif compact.startswith("#0"):
                # #0 MidasOperator / #0 MidasOperatorEnd -> not a data section
                section = None
            else:
                section = None
            continue

        if section == "modenames":
            mode_names.extend(line.split())
        elif section == "scalefactors":
            if scale_factors is None:
                scale_factors = []
            scale_factors.extend(float(x) for x in line.split())
        elif section == "operatorterms":
            term = _parse_term_line(line)
            if term is not None:
                terms.append(term)

    if not mode_names:
        raise ValueError(f"No #1 ModeNames section found in {path}")
    if not terms:
        raise ValueError(f"No #1 OperatorTerms parsed from {path}")

    return MopDocument(
        mode_names=mode_names,
        scale_factors=scale_factors,
        terms=terms,
    )


def _parse_term_line(line: str) -> tuple[float, list[tuple[str, int]]] | None:
    """Parse one operator term line: ``<coeff> Q^p(Qi) Q^m(Qj) ...``."""
    m = re.match(rf"\s*({_FLOAT})\s+(.*)", line)
    if not m:
        return None
    coeff = float(m.group(1))
    rest = m.group(2)

    factors = _FACTOR_RE.findall(rest)  # list of (power_str, mode_name)
    if not factors:
        return None

    parsed: list[tuple[str, int]] = []
    for power_str, mode_name in factors:
        power = int(power_str) if power_str else 1
        parsed.append((mode_name, power))
    return coeff, parsed


def mop_to_forcefield(
    doc: MopDocument,
    selected_modes: tuple[int, ...] | None = None,
    *,
    omega_rel_tol: float = 1e-6,
) -> VibrationalForceField:
    """Convert a :class:`MopDocument` to a :class:`VibrationalForceField`.

    Parameters
    ----------
    doc
        Parsed ``.mop`` document.
    selected_modes
        1-based mode indices to keep (relative to the ``.mop`` mode order). If
        ``None``, keep all modes. Terms that reference any dropped mode are
        discarded (so the subset Hamiltonian is self-contained).
    omega_rel_tol
        When both a ``Q^2`` diagonal coefficient and a ScaleFactor exist for a
        mode, assert they imply the same ``omega_au`` to this relative tolerance.

    Returns
    -------
    VibrationalForceField
        Atomic-unit, dimensionless-coordinate force field ready for
        ``build_direct_encoded_vibrational_hamiltonian(..., harmonic_source=
        "polynomial_quadratic")``.
    """
    n_all = doc.n_modes
    name_to_global = {name: i for i, name in enumerate(doc.mode_names)}  # 0-based

    if selected_modes is None:
        selected_global = list(range(n_all))
    else:
        selected_global = [m - 1 for m in selected_modes]
        for g in selected_global:
            if g < 0 or g >= n_all:
                raise ValueError(f"selected mode {g + 1} out of range 1..{n_all}")

    # Map global 0-based mode -> new 0-based position in the subset.
    global_to_local = {g: k for k, g in enumerate(selected_global)}
    n_modes = len(selected_global)

    # --- Derive omega_au (atomic units) per selected mode from Q^2 coeffs. ---
    omega_from_q2: dict[int, float] = {}
    for coeff, factors in doc.terms:
        if len(factors) == 1 and factors[0][1] == 2:
            g = name_to_global.get(factors[0][0])
            if g is not None:
                omega_from_q2[g] = 2.0 * coeff  # 0.5*omega*Q^2 => omega = 2*coeff

    omega: list[float] = []
    for g in selected_global:
        w = omega_from_q2.get(g)
        # Cross-check / fallback via ScaleFactors: scale = sqrt(omega_cm*CM2AU),
        # and omega_cm*CM2AU = omega_au  => omega_au = scale**2.
        if doc.scale_factors is not None and g < len(doc.scale_factors):
            w_scale = doc.scale_factors[g] ** 2
            if w is None:
                w = w_scale
            elif abs(w - w_scale) > omega_rel_tol * max(abs(w), abs(w_scale)):
                raise ValueError(
                    f"omega mismatch for mode {g} ({doc.mode_names[g]}): "
                    f"from Q^2 coeff = {w:.6e} a.u., from ScaleFactor = "
                    f"{w_scale:.6e} a.u. (rel tol {omega_rel_tol})."
                )
        if w is None:
            raise ValueError(
                f"Could not determine omega for mode {g} "
                f"({doc.mode_names[g]}): no Q^2 term and no ScaleFactor."
            )
        if w <= 0.0:
            raise ValueError(f"Non-positive omega {w} for mode {doc.mode_names[g]}.")
        omega.append(w)

    # --- Build ForceTerms (1-based, subset-local), dropping cross-subset terms. ---
    terms: list[ForceTerm] = []
    n_dropped = 0
    for coeff, factors in doc.terms:
        try:
            local_modes = []
            powers = []
            for mode_name, power in factors:
                g = name_to_global[mode_name]
                if g not in global_to_local:
                    raise KeyError(mode_name)  # references a dropped mode
                local_modes.append(global_to_local[g] + 1)  # 1-based
                powers.append(power)
        except KeyError:
            n_dropped += 1
            continue
        terms.append(
            ForceTerm(coeff=coeff, modes=tuple(local_modes), powers=tuple(powers))
        )

    metadata: dict[str, Any] = {
        "source_format": "midascpp_mop",
        "coefficient_units": "a.u.",
        "omega_units": "a.u.",
        "coordinate_convention": "dimensionless reduced normal coordinates "
        "(q=(a+a+)/sqrt2), same as MidasCpp Q",
        "cm2au": CM2AU,
        "n_modes_full": n_all,
        "selected_modes_1based": [g + 1 for g in selected_global],
        "mode_names": [doc.mode_names[g] for g in selected_global],
        "has_scale_factors": doc.scale_factors is not None,
        "n_terms_dropped_cross_subset": n_dropped,
        "harmonic_source_recommended": "polynomial_quadratic",
    }

    return VibrationalForceField(
        omega=tuple(omega),
        terms=tuple(terms),
        coordinate_kind="dimensionless",
        source="midascpp_mop",
        metadata=metadata,
    )


def omega_cm_from_forcefield(ff: VibrationalForceField) -> list[float]:
    """Return per-mode frequencies in cm^-1 (omega_au / CM2AU)."""
    return [w / CM2AU for w in ff.omega]


def _self_check(path: str | Path) -> None:
    """Convention self-check: recover omega from Q^2 coeffs and ScaleFactors."""
    doc = parse_mop(path)
    ff = mop_to_forcefield(doc)
    print(f"Parsed {path}")
    print(f"  modes:        {doc.n_modes}  {doc.mode_names}")
    print(f"  terms:        {len(doc.terms)}")
    print(f"  scalefactors: {doc.scale_factors}")
    print(f"  omega (a.u.): {[f'{w:.6e}' for w in ff.omega]}")
    print(f"  omega (cm-1): {[f'{w:.3f}' for w in omega_cm_from_forcefield(ff)]}")
    if doc.scale_factors is not None:
        for g, sf in enumerate(doc.scale_factors[: doc.n_modes]):
            w_q2 = ff.omega[g]
            w_sf = sf**2
            rel = abs(w_q2 - w_sf) / max(abs(w_q2), abs(w_sf))
            status = "OK" if rel < 1e-6 else "MISMATCH"
            print(
                f"    mode {g} ({doc.mode_names[g]}): omega_Q2={w_q2:.6e} "
                f"omega_SF={w_sf:.6e} rel={rel:.2e} [{status}]"
            )


if __name__ == "__main__":
    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "mop_files"
            / "h2o_dimless_qff_numeric.mop"
        )
    )
    _self_check(target)
