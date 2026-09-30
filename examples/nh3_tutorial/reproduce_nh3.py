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
`run_nh3.py` commands from that README for one concrete cell (NH3, 2
modals/mode, modal basis) and prints the recovered energy next to the
reference-config value, so you can see the whole pipeline work end to end
without copy-pasting each command by hand.

This is NOT a reimplementation -- it invokes run_nh3.py exactly as
documented, as a subprocess, from this repo's examples/drivers/ directory.

Usage:
    python reproduce_nh3.py --dry-run   # circuit cost preview only
    python reproduce_nh3.py --sim       # noiseless simulator (default)
    python reproduce_nh3.py --submit    # real IBM Quantum hardware
                                          # (spends QPU quota!)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DRIVERS_DIR = _HERE.parent / "drivers"

MODALS = 2
BASIS = "modal"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="circuit cost preview only"
    )
    mode.add_argument(
        "--sim", action="store_true", help="noiseless simulator (default)"
    )
    mode.add_argument(
        "--submit",
        action="store_true",
        help="real IBM Quantum hardware -- spends QPU quota!",
    )
    args = ap.parse_args()

    if not _DRIVERS_DIR.exists():
        raise SystemExit(f"drivers directory not found: {_DRIVERS_DIR}")

    stage = "--dry-run" if args.dry_run else "--submit" if args.submit else "--sim"
    cmd = [
        sys.executable,
        "run_nh3.py",
        "--modals",
        str(MODALS),
        "--basis",
        BASIS,
        stage,
    ]
    print(f"=== running: {' '.join(cmd)}  (cwd={_DRIVERS_DIR}) ===")
    proc = subprocess.run(cmd, cwd=str(_DRIVERS_DIR))
    if proc.returncode != 0:
        raise SystemExit(f"run_nh3.py exited with code {proc.returncode}")

    if args.dry_run:
        print(
            "\n=== dry-run complete -- see report/nh3_12q/modal_*_dryrun.json "
            "for transpiled circuit cost per ansatz ==="
        )
        return

    rep_dir = _DRIVERS_DIR / "report" / "nh3_12q"
    if args.sim:
        for path in sorted(rep_dir.glob(f"{BASIS}_*_results.json")):
            data = json.loads(path.read_text())
            ansatz = data.get("ansatz", path.stem)
            e = data.get("recovered_energy_cm")
            ref = data.get("reference_config_energy_cm")
            if e is None:
                continue
            print(
                f"  {ansatz:8s} recovered={e:9.2f} cm^-1  "
                f"(reference-config={ref:.2f})"
            )
    else:  # --submit
        results_path = rep_dir / f"{BASIS}_results.json"
        if results_path.exists():
            data = json.loads(results_path.read_text())
            for ansatz, v in data.get("variants", {}).items():
                e = v["recovered_energy_cm"]
                print(
                    f"  {ansatz:8s} recovered={e:9.2f} cm^-1  "
                    f"retained={v['onehot_retained_frac']*100:.2f}%"
                )


if __name__ == "__main__":
    main()
