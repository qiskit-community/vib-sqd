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
Generalized VCC t1/t2 amplitude extractor for MidasCpp ``.mout`` output.

Generalizes ``h2o_3m_4modal/extract_vcc_t1_t2_from_midas.py`` (which hardcodes
n_modes=3, n_modals=4) to arbitrary systems, and correctly selects the
**converged ground-state** VCC solution vector when the ``.mout`` contains many
VCC blocks (e.g. one per VSCF state from ``UseAllVscf`` + ``OccAllFund``).

Amplitude conventions (match vib_sqd.classical.vcc_solver.VCCAmplitudes):
    t1[i, a]      : mode i, ref modal 0 -> modal a+1 ;  shape (n_modes, n_modals-1)
    t2[i, j, a, b]: modes i,j both 0 -> a+1, b+1     ;  shape (n_modes,n_modes,n_exc,n_exc)

MidasCpp solution-vector lines look like::

     1.0000000000000000E+00 * Psi_Ref  ()               [0,0,0]
    +3.4879304795543152E-02 * Psi_41    (0:1, 2:2)       [1,0,2]

where ``(mode:modal, ...)`` are 0-based mode / 1-based modal excitations.

Usage:
    python generalized_vcc_extractor.py <mout> --n-modes 3 --n-modals 4 \
        --layers 4 --outdir <dir> [--label water]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from vib_sqd.classical.amplitude_io import (
    save_vcc_amplitudes_npz,
)
from vib_sqd.classical.vcc_solver import VCCAmplitudes
from vib_sqd.initialization.vcc_initialization import (
    map_vccsd_to_chc_params,
    map_vccsd_to_uvccsd_initial_point,
    map_vccsd_to_vlucj_params,
)

FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"
_AMP_RE = re.compile(rf"({FLOAT_RE})\s*\*\s*Psi_(?:\d+|Ref)\s*\(([^)]*)\)")
_HDR_RE = re.compile(r"Analysis of solution vector - the\s+(\d+)\s+largest")


def parse_excitation_label(label: str) -> list[tuple[int, int]]:
    """Parse ``"0:1, 2:2"`` -> ``[(0,1),(2,2)]`` (mode:modal)."""
    label = label.strip()
    if not label:
        return []
    out = []
    for p in label.split(","):
        p = p.strip()
        if ":" not in p:
            continue
        m, mo = p.split(":")
        out.append((int(m), int(mo)))
    return out


def _find_ground_state_vcc_block(text: str) -> str:
    """Return the converged **ground-state** VCC solution-vector block.

    Strategy: split the file at ``"Analysis of solution vector"`` headers, keep
    only blocks that (a) are preceded by ``"Non-linear VCC equations
    converged"`` (i.e. VCC, not the early VSCF-analysis print), and whose
    reference line ``Psi_Ref ( ) [...]`` has an **all-zero occupation** (the
    ground state). Among those, pick the one printing the most elements.
    """
    headers = list(_HDR_RE.finditer(text))
    if not headers:
        raise ValueError("No 'Analysis of solution vector' block found in .mout")

    best_block = None
    best_count = -1
    for i, h in enumerate(headers):
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        n_printed = int(h.group(1))

        # Must be a converged VCC block (preceded by the convergence banner).
        preamble = text[max(0, start - 400) : start]
        if "Non-linear VCC equations converged" not in preamble:
            continue

        # Ground state: the Psi_Ref line's second bracket [..] is all zeros.
        ref_match = re.search(r"Psi_Ref\s*\(\s*\)\s*\[([0-9,\s]*)\]", block)
        if ref_match:
            occ = [int(x) for x in ref_match.group(1).split(",") if x.strip() != ""]
            if any(occ):
                continue  # excited-state VCC block; skip

        if n_printed > best_count:
            best_count = n_printed
            best_block = block

    if best_block is None:
        raise ValueError(
            "Could not locate a converged ground-state VCC solution block. "
            "Check that the input ran VCC and printed the solution vector."
        )
    return best_block


def extract_amplitudes(
    mout_path: Path,
    n_modes: int,
    n_modals: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract ground-state t1/t2 from a MidasCpp ``.mout``."""
    n_exc = n_modals - 1
    t1 = np.zeros((n_modes, n_exc))
    t2 = np.zeros((n_modes, n_modes, n_exc, n_exc))

    text = mout_path.read_text()
    block = _find_ground_state_vcc_block(text)

    # VCC energy (a.u.) and convergence, from the whole file.
    vcc_energy = None
    for line in text.splitlines():
        if "Final VCC energy" in line and "a.u" in line:
            m = re.search(FLOAT_RE, line)
            if m:
                vcc_energy = float(m.group())
                break
    converged = "Non-linear VCC equations converged" in text
    it_match = re.search(r"converged in\s+(\d+)\s+iterations", text)
    iterations = int(it_match.group(1)) if it_match else None

    n_t1 = n_t2 = 0
    n_out_mode = n_out_modal = n_higher = 0

    for line in block.splitlines():
        m = _AMP_RE.search(line)
        if not m:
            continue
        coeff = float(m.group(1))
        exc = parse_excitation_label(m.group(2))

        if len(exc) == 0:
            continue  # reference
        elif len(exc) == 1:
            mode, modal = exc[0]
            if not (0 <= mode < n_modes):
                n_out_mode += 1
                continue
            a = modal - 1
            if not (0 <= a < n_exc):
                n_out_modal += 1
                continue
            t1[mode, a] = coeff
            n_t1 += 1
        elif len(exc) == 2:
            (m1, mo1), (m2, mo2) = exc
            if not (0 <= m1 < n_modes and 0 <= m2 < n_modes):
                n_out_mode += 1
                continue
            a, b = mo1 - 1, mo2 - 1
            if not (0 <= a < n_exc and 0 <= b < n_exc):
                n_out_modal += 1
                continue
            if m1 <= m2:
                t2[m1, m2, a, b] = coeff
            else:
                t2[m2, m1, b, a] = coeff
            n_t2 += 1
        else:
            n_higher += 1

    exp_t1 = n_modes * n_exc
    exp_t2 = (n_modes * (n_modes - 1) // 2) * n_exc * n_exc
    complete = n_t1 == exp_t1 and n_t2 == exp_t2 and n_out_mode == 0 and n_higher == 0

    meta = {
        "source": "MidasCpp",
        "amplitude_source": "printed_ground_state_solution_vector_components",
        "n_modes": n_modes,
        "n_modals": n_modals,
        "expected_t1_count": exp_t1,
        "expected_t2_count": exp_t2,
        "parsed_t1_count": n_t1,
        "parsed_t2_count": n_t2,
        "active_space_amplitudes_complete": complete,
        "ignored_outside_mode_space": n_out_mode,
        "ignored_outside_modal_space": n_out_modal,
        "ignored_higher_excitations": n_higher,
        "vcc_energy_au": vcc_energy,
        "converged": converged,
        "iterations": iterations,
        "t1_norm": float(np.linalg.norm(t1)),
        "t2_norm": float(np.linalg.norm(t2)),
    }
    return t1, t2, meta


def _find_vcc_block_for_occupation(text: str, occupation: list[int]) -> str:
    """Generalizes ``_find_ground_state_vcc_block`` to an arbitrary
    reference occupation (the ``Psi_Ref`` bracket must equal `occupation`
    exactly) -- needed for a ``#3 Occup`` excited-state VCC run, whose
    reference is not the all-zero ground state."""
    headers = list(_HDR_RE.finditer(text))
    if not headers:
        raise ValueError("No 'Analysis of solution vector' block found in .mout")

    target = list(occupation)
    best_block = None
    best_count = -1
    for i, h in enumerate(headers):
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        n_printed = int(h.group(1))

        preamble = text[max(0, start - 400) : start]
        if "Non-linear VCC equations converged" not in preamble:
            continue

        ref_match = re.search(r"Psi_Ref\s*\(\s*\)\s*\[([0-9,\s]*)\]", block)
        if not ref_match:
            continue
        occ = [int(x) for x in ref_match.group(1).split(",") if x.strip() != ""]
        if occ != target:
            continue

        if n_printed > best_count:
            best_count = n_printed
            best_block = block

    if best_block is None:
        raise ValueError(
            "Could not locate a converged VCC solution block whose reference "
            f"occupation matches {target}."
        )
    return best_block


def extract_amplitudes_for_occupation(
    mout_path: Path,
    n_modes: int,
    n_modals: int,
    occupation: list[int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract t1/t2 from a MidasCpp ``.mout`` whose reference is an
    arbitrary per-mode occupation (e.g. a ``#3 Occup`` excited-state run),
    not necessarily the all-zero ground state that :func:`extract_amplitudes`
    assumes.

    Amplitude convention: t1[i, a] / t2[i, j, a, b] use the same "logical"
    excitation-index convention as :func:`extract_amplitudes` -- index a is
    the a-th modal *other than* mode i's own reference occupation[i], in
    ascending absolute-modal order (matches
    ``vib_sqd.basis.modal_occupation.modal_permutation_for_occupation``).
    When ``occupation`` is all-zero this is identical to
    :func:`extract_amplitudes`'s convention.
    """
    from vib_sqd.basis.modal_occupation import modal_permutation_for_occupation

    n_exc = n_modals - 1
    t1 = np.zeros((n_modes, n_exc))
    t2 = np.zeros((n_modes, n_modes, n_exc, n_exc))
    perm = modal_permutation_for_occupation(occupation, n_modals)
    to_logical = [{abs_modal: a for a, abs_modal in enumerate(p)} for p in perm]

    text = mout_path.read_text()
    block = _find_vcc_block_for_occupation(text, occupation)

    vcc_energy = None
    for line in text.splitlines():
        if "Final VCC energy" in line and "a.u" in line:
            m = re.search(FLOAT_RE, line)
            if m:
                vcc_energy = float(m.group())
                break
    converged = "Non-linear VCC equations converged" in text
    it_match = re.search(r"converged in\s+(\d+)\s+iterations", text)
    iterations = int(it_match.group(1)) if it_match else None

    n_t1 = n_t2 = 0
    n_out_mode = n_out_modal = n_higher = 0

    for line in block.splitlines():
        m = _AMP_RE.search(line)
        if not m:
            continue
        coeff = float(m.group(1))
        exc = parse_excitation_label(m.group(2))

        if len(exc) == 0:
            continue  # reference
        elif len(exc) == 1:
            mode, abs_modal = exc[0]
            if not (0 <= mode < n_modes):
                n_out_mode += 1
                continue
            a = to_logical[mode].get(abs_modal)
            if a is None or a == 0:
                n_out_modal += 1
                continue
            t1[mode, a - 1] = coeff
            n_t1 += 1
        elif len(exc) == 2:
            (m1, mo1), (m2, mo2) = exc
            if not (0 <= m1 < n_modes and 0 <= m2 < n_modes):
                n_out_mode += 1
                continue
            a = to_logical[m1].get(mo1)
            b = to_logical[m2].get(mo2)
            if a is None or b is None or a == 0 or b == 0:
                n_out_modal += 1
                continue
            a, b = a - 1, b - 1
            if m1 <= m2:
                t2[m1, m2, a, b] = coeff
            else:
                t2[m2, m1, b, a] = coeff
            n_t2 += 1
        else:
            n_higher += 1

    exp_t1 = n_modes * n_exc
    exp_t2 = (n_modes * (n_modes - 1) // 2) * n_exc * n_exc
    complete = n_t1 == exp_t1 and n_t2 == exp_t2 and n_out_mode == 0 and n_higher == 0

    meta = {
        "source": "MidasCpp",
        "amplitude_source": "printed_occupation_referenced_solution_vector_components",
        "occupation": list(occupation),
        "n_modes": n_modes,
        "n_modals": n_modals,
        "expected_t1_count": exp_t1,
        "expected_t2_count": exp_t2,
        "parsed_t1_count": n_t1,
        "parsed_t2_count": n_t2,
        "active_space_amplitudes_complete": complete,
        "ignored_outside_mode_space": n_out_mode,
        "ignored_outside_modal_space": n_out_modal,
        "ignored_higher_excitations": n_higher,
        "vcc_energy_au": vcc_energy,
        "converged": converged,
        "iterations": iterations,
        "t1_norm": float(np.linalg.norm(t1)),
        "t2_norm": float(np.linalg.norm(t2)),
    }
    return t1, t2, meta


def extract_and_map(
    mout_path: str | Path,
    n_modes: int,
    n_modals: int,
    layers: int,
    outdir: str | Path,
    label: str = "system",
) -> dict:
    """Full extraction + init-parameter mapping. Returns a summary dict."""
    mout_path = Path(mout_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t1, t2, meta = extract_amplitudes(mout_path, n_modes, n_modals)

    amps = VCCAmplitudes(
        t1=t1,
        t2=t2,
        energy=meta["vcc_energy_au"],
        converged=meta["converged"],
        iterations=meta["iterations"],
        metadata=meta,
    )
    amps.validate()

    npz_path = outdir / f"{label}_vcc_amplitudes.npz"
    save_vcc_amplitudes_npz(amps, npz_path)
    (outdir / f"{label}_vcc_amplitudes.metadata.json").write_text(
        json.dumps(meta, indent=2)
    )

    # Map to ansatz init points (UVCCSD always; VLUCJ/CHC best-effort).
    init_paths: dict[str, str] = {}
    param_counts: dict[str, int] = {}

    uvccsd_x0 = map_vccsd_to_uvccsd_initial_point(
        amps,
        n_modes=n_modes,
        n_modals=n_modals,
        scaling_strategy="direct",
        clip=None,
    )
    np.save(outdir / f"{label}_uvccsd_vcc_init.npy", uvccsd_x0)
    init_paths["uvccsd"] = str(outdir / f"{label}_uvccsd_vcc_init.npy")
    param_counts["uvccsd"] = len(uvccsd_x0)

    chc_x0 = map_vccsd_to_chc_params(
        amps,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        scaling_strategy="direct",
        clip=np.pi,
        distribute_over_layers=True,
    )
    np.save(outdir / f"{label}_chc_l{layers}_vcc_init.npy", chc_x0)
    init_paths["chc"] = str(outdir / f"{label}_chc_l{layers}_vcc_init.npy")
    param_counts["chc"] = len(chc_x0)

    # VLUCJ: adjacent-only Jastrow layout (matches EnhancedVUCJAnsatz when
    # adjacent_jastrow=True). See reconcile note in run_pipeline.py.
    vlucj_x0 = map_vccsd_to_vlucj_params(
        amps,
        n_modes=n_modes,
        n_modals=n_modals,
        layers=layers,
        include_rz=True,
        scaling_strategy="direct",
        clip=np.pi,
        distribute_over_layers=True,
    )
    np.save(outdir / f"{label}_vlucj_l{layers}_vcc_init.npy", vlucj_x0)
    init_paths["vlucj"] = str(outdir / f"{label}_vlucj_l{layers}_vcc_init.npy")
    param_counts["vlucj"] = len(vlucj_x0)

    summary = {
        "label": label,
        "n_modes": n_modes,
        "n_modals": n_modals,
        "layers": layers,
        "amplitudes_meta": meta,
        "init_param_paths": init_paths,
        "init_param_counts": param_counts,
        "npz": str(npz_path),
    }
    (outdir / f"{label}_extraction_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mout")
    ap.add_argument("--n-modes", type=int, required=True)
    ap.add_argument("--n-modals", type=int, required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--outdir", default="vcc_init_params")
    ap.add_argument("--label", default="system")
    args = ap.parse_args()

    summary = extract_and_map(
        args.mout,
        args.n_modes,
        args.n_modals,
        args.layers,
        args.outdir,
        args.label,
    )
    m = summary["amplitudes_meta"]
    print(
        f"[{args.label}] VCC energy = {m['vcc_energy_au']} a.u.  "
        f"converged={m['converged']} iters={m['iterations']}"
    )
    print(
        f"  t1 parsed {m['parsed_t1_count']}/{m['expected_t1_count']}  "
        f"t2 parsed {m['parsed_t2_count']}/{m['expected_t2_count']}  "
        f"complete={m['active_space_amplitudes_complete']}"
    )
    print(f"  t1_norm={m['t1_norm']:.6e}  t2_norm={m['t2_norm']:.6e}")
    print(f"  init param counts: {summary['init_param_counts']}")


if __name__ == "__main__":
    main()
