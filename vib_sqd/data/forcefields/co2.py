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
CO2 vibrational forcefield and Table 2 benchmark case definitions.

Important:
    The approximate forcefields in this file are NOT paper-authentic.
    The PySCF-fitted path produces an ab-initio regenerated forcefield, but it
    is still not guaranteed to match Table 2 unless coordinate conventions,
    selected modes, basis, geometry, modal construction, and PES fitting match
    the paper exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

# ForceTerm/VibrationalForceField are molecule-agnostic; the canonical
# definition lives in classical.forcefield_types. Re-exported here for
# backward compatibility with existing importers of this module.
from vib_sqd.classical.forcefield_types import (
    ForceTerm,
    VibrationalForceField,
)


class ForceFieldSource:
    APPROXIMATE = "approximate"
    PYSCF_FITTED = "pyscf_fitted"
    PAPER_AUTHENTIC = "paper_authentic"
    CALIBRATED_TABLE2 = "calibrated_table2"


@dataclass(frozen=True)
class Table2CaseSpec:
    case: str
    source: str
    selected_modes: tuple[int, ...]
    n_modes: int
    vmax: int
    reference_gaps_cm: tuple[float, ...]
    notes: str
    forcefield: VibrationalForceField


def load_co2_approximate_forcefield() -> VibrationalForceField:
    """
    Approximate 4-mode CO2 forcefield.

    This is only a pipeline/debug forcefield, not a paper-authentic model.
    """
    omega = (
        667.38,
        667.38,
        1388.19,
        2349.14,
    )

    terms = (
        ForceTerm(-5.0, (1,), (3,)),
        ForceTerm(-5.0, (2,), (3,)),
        ForceTerm(-8.0, (3,), (3,)),
        ForceTerm(-12.0, (4,), (3,)),
        ForceTerm(-3.0, (1, 2), (2, 1)),
        ForceTerm(-3.0, (1, 2), (1, 2)),
        ForceTerm(-4.0, (1, 3), (2, 1)),
        ForceTerm(-4.0, (2, 3), (2, 1)),
        ForceTerm(-6.0, (1, 4), (2, 1)),
        ForceTerm(-6.0, (2, 4), (2, 1)),
        ForceTerm(-10.0, (3, 4), (2, 1)),
        ForceTerm(1.0, (1,), (4,)),
        ForceTerm(1.0, (2,), (4,)),
        ForceTerm(2.0, (3,), (4,)),
        ForceTerm(3.0, (4,), (4,)),
        ForceTerm(0.5, (1, 2), (3, 1)),
        ForceTerm(0.5, (1, 2), (1, 3)),
        ForceTerm(1.0, (1, 2), (2, 2)),
        ForceTerm(1.5, (1, 3), (2, 2)),
        ForceTerm(1.5, (2, 3), (2, 2)),
        ForceTerm(2.0, (1, 4), (2, 2)),
        ForceTerm(2.0, (2, 4), (2, 2)),
        ForceTerm(3.0, (3, 4), (2, 2)),
    )

    return VibrationalForceField(
        omega=omega,
        terms=terms,
        coordinate_kind="dimensionless",
        source=ForceFieldSource.APPROXIMATE,
        metadata={
            "warning": "Approximate placeholder, not Table 2 authentic.",
            "molecule": "CO2",
        },
    )


def load_co2_approximate_bending_only() -> VibrationalForceField:
    omega = (667.38, 667.38)
    terms = (
        ForceTerm(-5.0, (1,), (3,)),
        ForceTerm(-5.0, (2,), (3,)),
        ForceTerm(-3.0, (1, 2), (2, 1)),
        ForceTerm(-3.0, (1, 2), (1, 2)),
        ForceTerm(1.0, (1,), (4,)),
        ForceTerm(1.0, (2,), (4,)),
        ForceTerm(0.5, (1, 2), (3, 1)),
        ForceTerm(0.5, (1, 2), (1, 3)),
        ForceTerm(1.0, (1, 2), (2, 2)),
    )
    return VibrationalForceField(
        omega=omega,
        terms=terms,
        coordinate_kind="dimensionless",
        source=ForceFieldSource.APPROXIMATE,
        metadata={
            "warning": "Approximate bending-only placeholder, not Table 2 authentic.",
            "molecule": "CO2",
        },
    )


def load_co2_forcefield() -> VibrationalForceField:
    """Backward-compatible alias for the approximate 4-mode model."""
    return load_co2_approximate_forcefield()


def load_co2_bending_only() -> VibrationalForceField:
    """Backward-compatible alias for the approximate 2-mode bending model."""
    return load_co2_approximate_bending_only()


def select_forcefield_modes(
    ff: VibrationalForceField,
    selected_modes: tuple[int, ...],
) -> VibrationalForceField:
    """
    Select a subset of 1-indexed modes and remap to compact 1-indexed labels.
    """
    selected_modes = tuple(selected_modes)
    mode_map = {
        original_mode: new_mode
        for new_mode, original_mode in enumerate(selected_modes, start=1)
    }

    omega = tuple(ff.omega[mode - 1] for mode in selected_modes)
    filtered_terms = []

    for term in ff.terms:
        if all(mode in mode_map for mode in term.modes):
            filtered_terms.append(
                ForceTerm(
                    coeff=term.coeff,
                    modes=tuple(mode_map[mode] for mode in term.modes),
                    powers=term.powers,
                )
            )

    metadata = dict(ff.metadata)
    metadata["selected_modes_original"] = selected_modes
    metadata["mode_remap"] = dict(mode_map)

    return VibrationalForceField(
        omega=omega,
        terms=tuple(filtered_terms),
        coordinate_kind=ff.coordinate_kind,
        source=ff.source,
        metadata=metadata,
    )


def load_co2_table2_case(
    case: str,
    source: str = ForceFieldSource.APPROXIMATE,
) -> Table2CaseSpec:
    """
    Load a case-labeled CO2 Table 2 benchmark config.

    Warning:
        Unless source=PAPER_AUTHENTIC is implemented, these are not guaranteed
        to reproduce Table 2.
    """
    normalized = case.strip().upper()

    selected_modes_by_case = {
        "A": (1, 2),  # Table 2: two bending modes, 2 modals
        "B": (1, 2),  # Table 2: two bending modes, 4 modals
        "C": (1, 2, 3, 4),  # Table 2: all four modes, 2 modals
    }

    if normalized not in selected_modes_by_case:
        raise ValueError(f"Unknown CO2 Table 2 case: {case!r}")

    selected_modes = selected_modes_by_case[normalized]

    if source == ForceFieldSource.APPROXIMATE:
        forcefield = select_forcefield_modes(
            load_co2_approximate_forcefield(), selected_modes
        )

    elif source == ForceFieldSource.PYSCF_FITTED:
        from vib_sqd.classical.pyscf_forcefield import (
            default_co2_spec,
            load_forcefield_from_pyscf,
        )

        forcefield = load_forcefield_from_pyscf(
            molecule=default_co2_spec(),
            selected_modes=selected_modes,
            max_degree=4,
            grid_points=(-0.08, -0.04, 0.0, 0.04, 0.08),
            max_tensor_points=5000,
        )

    else:
        raise NotImplementedError(
            f"CO2 Table 2 source '{source}' is not implemented yet."
        )

    if normalized == "A":
        return Table2CaseSpec(
            case="A",
            source=source,
            selected_modes=selected_modes,
            n_modes=len(selected_modes),
            vmax=1,
            reference_gaps_cm=(574.441, 1438.778, 2063.261),
            notes=(
                "2 modes, 2 modals. Current selected modes are placeholder "
                "bend + symmetric stretch. Not paper-authentic unless "
                "source=PAPER_AUTHENTIC is implemented."
            ),
            forcefield=forcefield,
        )

    if normalized == "B":
        return Table2CaseSpec(
            case="B",
            source=source,
            selected_modes=selected_modes,
            n_modes=len(selected_modes),
            vmax=3,
            reference_gaps_cm=(
                496.697,
                1073.420,
                1460.074,
                1642.996,
                2024.187,
                2498.060,
            ),
            notes=(
                "2 modes, 4 modals. Same placeholder mode selection as A but "
                "larger modal truncation."
            ),
            forcefield=forcefield,
        )

    if normalized == "C":
        return Table2CaseSpec(
            case="C",
            source=source,
            selected_modes=selected_modes,
            n_modes=len(selected_modes),
            vmax=1,
            reference_gaps_cm=(
                534.908,
                559.330,
                1098.527,
                1267.081,
                1855.895,
                1880.816,
            ),
            notes=(
                "4 modes, 2 modals. Not Table 2-authentic until paper forcefield "
                "or calibrated/modal Hamiltonian is implemented."
            ),
            forcefield=forcefield,
        )

    raise ValueError(f"Unknown CO2 Table 2 case: {case!r}")


def load_co2_custom_case(
    selected_modes: tuple[int, ...],
    n_modals: int,
    source: str = ForceFieldSource.APPROXIMATE,
) -> Table2CaseSpec:
    """
    Load a custom CO2 benchmark case.

    Args:
        selected_modes:
            1-indexed CO2 modes to include.
        n_modals:
            Number of modals per mode. vmax = n_modals - 1.
        source:
            approximate or pyscf_fitted.
    """
    if n_modals < 2:
        raise ValueError("n_modals must be at least 2.")

    selected_modes = tuple(int(m) for m in selected_modes)
    vmax = n_modals - 1

    if source == ForceFieldSource.APPROXIMATE:
        forcefield = select_forcefield_modes(
            load_co2_approximate_forcefield(),
            selected_modes,
        )

    elif source == ForceFieldSource.PYSCF_FITTED:
        from vib_sqd.classical.pyscf_forcefield import (
            default_co2_spec,
            load_forcefield_from_pyscf,
        )

        forcefield = load_forcefield_from_pyscf(
            molecule=default_co2_spec(),
            selected_modes=selected_modes,
            max_degree=4,
            grid_points=(-0.08, -0.04, 0.0, 0.04, 0.08),
            max_tensor_points=5000,
        )

    else:
        raise NotImplementedError(f"CO2 source {source!r} is not implemented.")

    return Table2CaseSpec(
        case=f"CUSTOM_{len(selected_modes)}m_{n_modals}modals",
        source=source,
        selected_modes=selected_modes,
        n_modes=len(selected_modes),
        vmax=vmax,
        reference_gaps_cm=(),
        notes=(
            f"CO2 custom case: selected_modes={selected_modes}, "
            f"n_modals={n_modals}."
        ),
        forcefield=forcefield,
    )


def load_co2_fig2_case(source: str = ForceFieldSource.APPROXIMATE) -> Table2CaseSpec:
    """
    Fig. 2 / Fig. 3 style ground-state VQE case:
    bending + symmetric stretching modes, 2 modals per mode.
    """
    selected_modes = (1, 3)

    if source == ForceFieldSource.APPROXIMATE:
        forcefield = select_forcefield_modes(
            load_co2_approximate_forcefield(), selected_modes
        )
    elif source == ForceFieldSource.PYSCF_FITTED:
        from vib_sqd.classical.pyscf_forcefield import (
            default_co2_spec,
            load_forcefield_from_pyscf,
        )

        forcefield = load_forcefield_from_pyscf(
            molecule=default_co2_spec(),
            selected_modes=selected_modes,
            max_degree=4,
            grid_points=(-0.08, -0.04, 0.0, 0.04, 0.08),
            max_tensor_points=5000,
        )
    else:
        raise NotImplementedError(f"source={source!r} not implemented")

    return Table2CaseSpec(
        case="FIG2",
        source=source,
        selected_modes=selected_modes,
        n_modes=2,
        vmax=1,
        reference_gaps_cm=(),
        notes="Fig. 2/Fig. 3 ground-state VQE case: bend + symmetric stretch.",
        forcefield=forcefield,
    )
