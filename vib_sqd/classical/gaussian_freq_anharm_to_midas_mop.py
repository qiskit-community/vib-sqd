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
Convert Gaussian freq=anharmonic force constants to a MIDASCPP .mop file.

This converter is intended for the MIDASCPP operator format

    #0 MidasOperator
    #1 ModeNames
    Q0 Q1 Q2 ...
    #1 ScaleFactors
    ...
    #1 OperatorTerms
    <numeric coefficient> Q^n(Qm) ...
    #0 MidasOperatorEnd

Default convention
------------------
Gaussian's freq=anharmonic output prints normal-mode force constants in three
useful columns: FI, k, and K.  This script uses the FI column by default:

    quadratic FI  = harmonic frequency, cm^-1
    cubic FI      = reduced cubic constants, cm^-1
    quartic FI    = reduced quartic constants, cm^-1

The output .mop coefficients are explicit numerical coefficients in Hartree,
obtained as

    coefficient = FI_cm^-1 * CM_TO_HARTREE / prod_a factorial(n_a)

where n_a is the number of repeated appearances of mode a in the term.  This is
what you want if the potential is interpreted as a Taylor expansion in
dimensionless normal coordinates q_i,

    V(q) = 1/2 sum_i nu_i q_i^2
         + 1/6 sum_ijk F_ijk q_i q_j q_k
         + 1/24 sum_ijkl F_ijkl q_i q_j q_k q_l.

The #1 ScaleFactors line is written as

    sqrt(nu_i_cm^-1 * CM_TO_HARTREE)

which is the same convention used in the MIDASCPP input examples we have been
using for dimensionless/frequency-scaled normal coordinates.

Examples
--------
    python gaussian_freq_anharm_to_midas_mop.py gaussian.log -o molecule.mop

    # Use one-based MIDAS mode labels Q1..QN instead of Q0..Q(N-1)
    python gaussian_freq_anharm_to_midas_mop.py gaussian.log -o molecule.mop --one-based

    # Keep only modes 1, 2, and 3 from the Gaussian table, and relabel to Q0,Q1,Q2
    python gaussian_freq_anharm_to_midas_mop.py gaussian.log -o active3.mop --modes 1-3

    # Keep modes 7, 8, 12 without compact relabeling, giving Q6,Q7,Q11 by default
    python gaussian_freq_anharm_to_midas_mop.py gaussian.log -o subset.mop --modes 7,8,12 --no-compact

Notes
-----
* Gaussian mode indices are one-based.  By default, MIDAS labels are zero-based
  Q0..Q(N-1), matching the format requested in the conversation.
* The script writes only force constants that Gaussian printed.  Gaussian usually
  omits constants below its internal print threshold; omitted constants are
  treated as zero.
* The script does not expand symmetry-equivalent permutations into separate
  lines.  It writes one MIDAS monomial per unique Gaussian row, with the Taylor
  multiplicity/factorial already included in the numeric coefficient.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

CM_TO_HARTREE = 1.0 / 219474.6313705

SECTION_TITLES = {
    2: "QUADRATIC FORCE CONSTANTS IN NORMAL MODES",
    3: "CUBIC FORCE CONSTANTS IN NORMAL MODES",
    4: "QUARTIC FORCE CONSTANTS IN NORMAL MODES",
}

FLOAT_RE = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[DdEe][-+]?\d+)?"


@dataclasses.dataclass(frozen=True)
class ForceConstant:
    order: int
    indices: tuple[int, ...]  # Gaussian one-based mode indices
    fi_cm: float
    k_value: float
    K_value: float
    source_line_no: int


def parse_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def factorial_denominator(indices: Iterable[int]) -> int:
    denom = 1
    for count in collections.Counter(indices).values():
        denom *= math.factorial(count)
    return denom


def find_section_ranges(lines: Sequence[str]) -> dict[int, tuple[int, int]]:
    """Return approximate [start, end) line-number ranges for Gaussian FC sections."""
    starts: dict[int, int] = {}
    for i, line in enumerate(lines):
        for order, title in SECTION_TITLES.items():
            if title in line:
                starts[order] = i

    ranges: dict[int, tuple[int, int]] = {}
    for order, start in starts.items():
        end = len(lines)
        for other_start in starts.values():
            if other_start > start:
                end = min(end, other_start)
        # Prefer the Gaussian summary line, when present.
        for j in range(start, end):
            if re.search(r"Num\. of\s+\d+(?:st|nd|rd|th) derivatives", lines[j]):
                end = j + 1
                break
        ranges[order] = (start, end)
    return ranges


def parse_force_constants(lines: Sequence[str], order: int) -> list[ForceConstant]:
    """Parse one Gaussian force-constant table."""
    title = SECTION_TITLES[order]
    try:
        start = next(i for i, line in enumerate(lines) if title in line)
    except StopIteration:
        return []

    # Data begin after the line containing the column headings FI(...), k(...), K(...).
    header_idx = None
    for i in range(start, min(start + 60, len(lines))):
        if "FI(" in lines[i] and "K(" in lines[i]:
            header_idx = i
            break
    if header_idx is None:
        return []

    pattern = re.compile(
        r"^\s*"
        + r"\s+".join([r"(\d+)"] * order)
        + r"\s+"
        + r"\s+".join([f"({FLOAT_RE})"] * 3)
        + r"\s*$"
    )

    constants: list[ForceConstant] = []
    for line_no in range(header_idx + 1, len(lines)):
        line = lines[line_no]
        if "Num. of" in line and "derivatives" in line:
            break
        if (
            any(other in line for other in SECTION_TITLES.values())
            and line_no > header_idx + 1
        ):
            break
        match = pattern.match(line)
        if not match:
            continue
        indices = tuple(int(match.group(i + 1)) for i in range(order))
        fi_cm = parse_float(match.group(order + 1))
        k_value = parse_float(match.group(order + 2))
        K_value = parse_float(match.group(order + 3))
        constants.append(
            ForceConstant(
                order=order,
                indices=indices,
                fi_cm=fi_cm,
                k_value=k_value,
                K_value=K_value,
                source_line_no=line_no + 1,
            )
        )
    return constants


def parse_modes_spec(spec: str | None) -> list[int] | None:
    """Parse strings like '1,2,5-8'. Gaussian mode numbers are one-based."""
    if not spec:
        return None
    modes: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"Invalid decreasing mode range: {part}")
            modes.extend(range(lo, hi + 1))
        else:
            modes.append(int(part))
    if not modes:
        raise ValueError("No modes parsed from --modes")
    if any(m <= 0 for m in modes):
        raise ValueError("Gaussian mode numbers in --modes must be positive")
    # Preserve user order but remove duplicates.
    seen: set[int] = set()
    unique: list[int] = []
    for mode in modes:
        if mode not in seen:
            seen.add(mode)
            unique.append(mode)
    return unique


def infer_nmodes(constants_by_order: dict[int, list[ForceConstant]]) -> int:
    max_mode = 0
    for constants in constants_by_order.values():
        for fc in constants:
            max_mode = max(max_mode, *fc.indices)
    return max_mode


def build_selected_modes(
    constants_by_order: dict[int, list[ForceConstant]], modes_spec: str | None
) -> list[int]:
    selected = parse_modes_spec(modes_spec)
    if selected is not None:
        return selected
    nmodes = infer_nmodes(constants_by_order)
    if nmodes <= 0:
        raise ValueError(
            "Could not infer number of modes from Gaussian force-constant tables"
        )
    return list(range(1, nmodes + 1))


def get_quadratic_frequencies(
    constants_by_order: dict[int, list[ForceConstant]],
) -> dict[int, float]:
    freqs: dict[int, float] = {}
    for fc in constants_by_order.get(2, []):
        if len(fc.indices) == 2 and fc.indices[0] == fc.indices[1]:
            freqs[fc.indices[0]] = fc.fi_cm
    return freqs


def make_label_map(
    selected_modes: Sequence[int], *, one_based: bool, compact: bool
) -> dict[int, str]:
    labels: dict[int, str] = {}
    if compact:
        for pos, gaussian_mode in enumerate(selected_modes):
            labels[gaussian_mode] = f"Q{pos + 1 if one_based else pos}"
    else:
        for gaussian_mode in selected_modes:
            labels[gaussian_mode] = (
                f"Q{gaussian_mode if one_based else gaussian_mode - 1}"
            )
    return labels


def coefficient_hartree(fc: ForceConstant) -> float:
    return fc.fi_cm * CM_TO_HARTREE / factorial_denominator(fc.indices)


def term_string(fc: ForceConstant, label_map: dict[int, str], precision: int) -> str:
    counts = collections.Counter(fc.indices)
    parts: list[str] = []
    # Use selected-mode order through label_map insertion order if possible; otherwise numeric.
    for mode in sorted(counts):
        parts.append(f"Q^{counts[mode]}({label_map[mode]})")
    return f"{coefficient_hartree(fc): .{precision}e} " + " ".join(parts)


def wrap_values(values: Sequence[str], *, per_line: int = 5) -> str:
    out: list[str] = []
    for i in range(0, len(values), per_line):
        out.append(" ".join(values[i : i + per_line]))
    return "\n".join(out)


def write_mop(
    output: Path,
    constants_by_order: dict[int, list[ForceConstant]],
    selected_modes: Sequence[int],
    *,
    one_based: bool,
    compact: bool,
    threshold_cm: float,
    threshold_au: float,
    include_comments: bool,
    precision: int,
) -> dict[str, int | float]:
    selected_set = set(selected_modes)
    label_map = make_label_map(selected_modes, one_based=one_based, compact=compact)
    mode_names = [label_map[m] for m in selected_modes]
    freqs = get_quadratic_frequencies(constants_by_order)

    missing = [m for m in selected_modes if m not in freqs]
    if missing:
        raise ValueError(
            "Missing diagonal quadratic FI frequencies for selected Gaussian modes: "
            + ", ".join(map(str, missing))
        )

    scale_factors: list[float] = []
    for mode in selected_modes:
        nu = freqs[mode]
        if nu <= 0.0:
            raise ValueError(
                f"Mode {mode} has non-positive harmonic frequency {nu} cm^-1; "
                "cannot form sqrt(nu*CM_TO_HARTREE)."
            )
        scale_factors.append(math.sqrt(nu * CM_TO_HARTREE))

    written_by_order = {2: 0, 3: 0, 4: 0}
    skipped_by_mode = {2: 0, 3: 0, 4: 0}
    skipped_by_threshold = {2: 0, 3: 0, 4: 0}

    with output.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("#0 MidasOperator\n\n")
        fh.write("#1 ModeNames\n")
        fh.write(" ".join(mode_names) + "\n\n")
        fh.write("#1 ScaleFactors\n")
        fh.write(" ".join(f"{x:.{precision}e}" for x in scale_factors) + "\n\n")
        fh.write("#1 OperatorTerms\n\n")
        if include_comments:
            fh.write("/* Generated by gaussian_freq_anharm_to_midas_mop.py. */\n")
            fh.write(
                "/* Gaussian FI constants are interpreted as cm^-1 constants in dimensionless normal coordinates. */\n"
            )
            fh.write(f"/* CM_TO_HARTREE = {CM_TO_HARTREE:.16e}. */\n")
            if compact:
                fh.write(
                    "/* Mode mapping: MIDAS label <- Gaussian normal-mode index. */\n"
                )
                for mode in selected_modes:
                    fh.write(
                        f"/*   {label_map[mode]} <- Gaussian mode {mode}, harmonic FI = {freqs[mode]:.8f} cm^-1 */\n"
                    )
            fh.write(
                "/* Coefficients include Taylor factorial/permutation factors. */\n\n"
            )

        for order in (2, 3, 4):
            constants = constants_by_order.get(order, [])
            if not constants:
                continue
            if include_comments:
                fh.write(f"/* Order {order} terms */\n")
            for fc in constants:
                if any(idx not in selected_set for idx in fc.indices):
                    skipped_by_mode[order] += 1
                    continue
                coeff_au = coefficient_hartree(fc)
                if abs(fc.fi_cm) < threshold_cm or abs(coeff_au) < threshold_au:
                    skipped_by_threshold[order] += 1
                    continue
                fh.write(term_string(fc, label_map, precision) + "\n")
                written_by_order[order] += 1
            fh.write("\n")
        fh.write("#0 MidasOperatorEnd\n")

    return {
        "nmodes_written": len(selected_modes),
        "terms2": written_by_order[2],
        "terms3": written_by_order[3],
        "terms4": written_by_order[4],
        "skipped_by_mode": sum(skipped_by_mode.values()),
        "skipped_by_threshold": sum(skipped_by_threshold.values()),
        "zpe_harm_cm": 0.5 * sum(freqs[m] for m in selected_modes),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract Gaussian freq=anharmonic FI force constants and write a numeric MIDASCPP .mop file.",
    )
    parser.add_argument(
        "logfile",
        type=Path,
        help="Gaussian .log/.out file from a freq=anharmonic calculation",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("gaussian_qff_numeric.mop"),
        help="Output MIDASCPP .mop file",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Gaussian normal modes to include, e.g. '1-3' or '1,2,5-7'. Default: all modes found",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Use MIDAS mode labels Q1..QN instead of Q0..Q(N-1)",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Do not relabel a selected subset compactly. Example: Gaussian mode 7 becomes Q6/Q7 rather than Q0/Q1",
    )
    parser.add_argument(
        "--threshold-cm",
        type=float,
        default=0.0,
        help="Skip terms with abs(raw FI in cm^-1) below this value",
    )
    parser.add_argument(
        "--threshold-au",
        type=float,
        default=0.0,
        help="Skip terms with abs(final Hartree coefficient) below this value",
    )
    parser.add_argument(
        "--comments",
        action="store_true",
        help="Include explanatory comments in the .mop file",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=16,
        help="Digits after decimal in scientific notation",
    )
    args = parser.parse_args(argv)

    if not args.logfile.exists():
        parser.error(f"Input file does not exist: {args.logfile}")
    text = args.logfile.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    constants_by_order = {
        order: parse_force_constants(lines, order) for order in (2, 3, 4)
    }
    if not constants_by_order[2]:
        raise SystemExit(
            "Could not find/parse the Gaussian quadratic force-constant table."
        )
    if not constants_by_order[3] and not constants_by_order[4]:
        raise SystemExit(
            "Could not find/parse Gaussian cubic or quartic force-constant tables."
        )

    selected_modes = build_selected_modes(constants_by_order, args.modes)
    compact = not args.no_compact
    try:
        stats = write_mop(
            args.output,
            constants_by_order,
            selected_modes,
            one_based=args.one_based,
            compact=compact,
            threshold_cm=args.threshold_cm,
            threshold_au=args.threshold_au,
            include_comments=args.comments,
            precision=args.precision,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ranges = find_section_ranges(lines)
    print(f"Wrote: {args.output}")
    print(f"Gaussian log: {args.logfile}")
    print(f"Modes written: {stats['nmodes_written']}")
    print(f"Harmonic ZPE from selected modes: {stats['zpe_harm_cm']:.8f} cm^-1")
    print("Parsed Gaussian rows:")
    for order in (2, 3, 4):
        if order in ranges:
            start, end = ranges[order]
            loc = f"lines {start + 1}-{end}"
        else:
            loc = "section not found"
        print(f"  order {order}: {len(constants_by_order[order])} rows ({loc})")
    print("Written MIDAS terms:")
    print(f"  quadratic: {stats['terms2']}")
    print(f"  cubic:     {stats['terms3']}")
    print(f"  quartic:   {stats['terms4']}")
    if stats["skipped_by_mode"]:
        print(f"Skipped because outside selected modes: {stats['skipped_by_mode']}")
    if stats["skipped_by_threshold"]:
        print(f"Skipped by threshold: {stats['skipped_by_threshold']}")
    print(
        "Coefficient convention: final MIDAS coefficient = FI_cm^-1 / 219474.6313705 / repeated-index-factorials."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
