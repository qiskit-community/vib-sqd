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

r"""
Thin convenience driver for README.md's tutorial: runs the exact
`run_excited_state_demo.py` command from that README for one concrete cell
(H2O, 6 modals/mode, modal basis, VLUCJ) and reports Delta E_1 = E1_sqd -
ZPE_sqd against the classical VCC(M=6) reference, so you can see the whole
excited-state pipeline work end-to-end.

This is NOT a reimplementation -- it invokes run_excited_state_demo.py
exactly as documented, as a subprocess, from this repo's examples/drivers/
directory. Unlike the ground-state driver, there is no --sim stage here
(run_excited_state_demo.py itself only supports --dry-run/--submit -- see
README.md Sec.3).

Usage:
    python reproduce_excited_state.py --dry-run   # circuit cost preview only
    python reproduce_excited_state.py --submit    # real IBM Quantum hardware
                                                    # (spends QPU quota!)
                                                    # requires the ground-state
                                                    # cell (see below) too.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DRIVERS_DIR = _HERE.parent / "drivers"

MOLECULE = "h2o"
MODALS = 6
BASIS = "modal"
ANSATZ = "VLUCJ"
# Real, verified VCC(M=6) classical reference Delta E_1 for water
# (tab:classicalrefs in the manuscript).
VCC_M6_DE1_REFERENCE_CM = 1552.22


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="circuit cost preview only"
    )
    mode.add_argument(
        "--submit",
        action="store_true",
        help="real IBM Quantum hardware -- spends QPU quota!",
    )
    args = ap.parse_args()
    stage = "--dry-run" if args.dry_run else "--submit"

    if not _DRIVERS_DIR.exists():
        raise SystemExit(f"drivers directory not found: {_DRIVERS_DIR}")

    cmd = [
        sys.executable,
        "run_excited_state_demo.py",
        "--modals",
        str(MODALS),
        "--molecules",
        MOLECULE,
        "--ansatze",
        ANSATZ,
        "--basis",
        BASIS,
        stage,
    ]
    print(f"=== running: {' '.join(cmd)}  (cwd={_DRIVERS_DIR}) ===")
    proc = subprocess.run(cmd, cwd=str(_DRIVERS_DIR))
    if proc.returncode != 0:
        raise SystemExit(
            f"run_excited_state_demo.py exited with code {proc.returncode}"
        )

    if args.dry_run:
        print(
            f"\n=== dry-run complete -- see "
            f"report/{MOLECULE}_excited/excited{'' if BASIS=='modal' else '_'+BASIS}"
            f"_modals{MODALS}_dryrun.json for transpiled circuit cost ==="
        )
        return

    exc_path = (
        _DRIVERS_DIR
        / "report"
        / f"{MOLECULE}_excited"
        / f"excited_{BASIS}_results_modals{MODALS}.json"
    )
    gs_path = (
        _DRIVERS_DIR / "report" / f"{MOLECULE}_{3 * MODALS}q" / f"{BASIS}_results.json"
    )

    if not exc_path.exists():
        print(
            f"\n! {exc_path} not found -- excited-state submission may not "
            "have completed."
        )
        return
    exc_data = json.loads(exc_path.read_text())
    e1 = exc_data.get("variants", {}).get(ANSATZ, {}).get("recovered_energy_cm")
    print(
        f"\n=== excited-state (E1_sqd) = {e1:.2f} cm^-1 (job {exc_data.get('job_id')}) ==="
    )

    if not gs_path.exists():
        print(
            f"\n! ground-state cell not found at {gs_path} -- run the "
            f"ground-state tutorial's --submit step first (--modals {MODALS} "
            f"--basis {BASIS}) to compute Delta E_1."
        )
        return
    gs_data = json.loads(gs_path.read_text())
    zpe = gs_data.get("variants", {}).get(ANSATZ, {}).get("recovered_energy_cm")
    if e1 is None or zpe is None:
        print("! missing recovered_energy_cm in one of the result files.")
        return
    de1 = e1 - zpe
    print(f"=== ground-state (ZPE_sqd)  = {zpe:.2f} cm^-1 ===")
    print(
        f"=== Delta E_1 (SQD)         = {de1:.2f} cm^-1  "
        f"(VCC(M={MODALS}) reference = {VCC_M6_DE1_REFERENCE_CM:.2f}, "
        f"Delta={de1 - VCC_M6_DE1_REFERENCE_CM:+.2f}) ==="
    )


if __name__ == "__main__":
    main()
