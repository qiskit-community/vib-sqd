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
Thin convenience driver for README.md's tutorial: demonstrates the SQD
operating model on glycine-4 (warm start -> a deliberately unconverged
few-step optimization -> finite-shot sample -> iterative subspace-growth
recovery), then runs the noisy-hardware benchmark exactly as documented.

This is NOT a reimplementation -- the noiseless part inlines the same
calls the notebook this tutorial is based on made directly (build_vim_ucj,
imucj_init_vector, qsci_augment.iterative_subspace_growth); the noisy part
invokes run_noisy_benchmark.py exactly as documented, as a subprocess, from
this repo's examples/drivers/ directory.

Usage:
    python reproduce_sqd_workflow.py             # noiseless part only (fast)
    python reproduce_sqd_workflow.py --noisy      # also run the ~1-2 min
                                                    # noisy-hardware benchmark
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
_DRIVERS_DIR = _HERE.parent / "drivers"
sys.path.insert(0, str(_DRIVERS_DIR))

from qiskit.quantum_info import Statevector

import vib_sqd.basis.modal_transform as mb
import vib_sqd.sqd.qsci_augment as qa
from vib_sqd.circuits.initial_states import build_vscf_reference_state
from vib_sqd.circuits.ucj_correlators import (
    build_vim_ucj,
    imucj_init_vector,
)
from vib_sqd.classical.mop_forcefield import (
    CM2AU,
    mop_to_forcefield,
    parse_mop,
)
from vib_sqd.classical.vcc_extractor import extract_amplitudes
from vib_sqd.classical.vcc_solver import VCCAmplitudes
from vib_sqd.run.dualbasis import _assign, ensure_clean_vcc

GLYCINE_MOP = _DRIVERS_DIR.parent.parent / "data" / "mop_files" / "glycinehplus.mop"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--noisy",
        action="store_true",
        help="also run the noisy-hardware benchmark (~1-2 min)",
    )
    args = ap.parse_args()

    print("=== 1-2. Warm start, few steps, sample (glycine-4, modal, VIm-uCJ) ===\n")
    n_modes, n_modals, n_ho = 4, 3, 9
    mout, ffmop = ensure_clean_vcc(
        "glycine4",
        GLYCINE_MOP,
        (24, 25, 26, 27),
        n_modals,
        str(_HERE / "runs" / "notebook"),
    )
    ff = mop_to_forcefield(parse_mop(ffmop))
    rot = mb.build_modal_rotation(mout, n_modes, n_ho, n_modals)
    t1, t2, _ = extract_amplitudes(Path(mout), n_modes, n_modals)
    amps = VCCAmplitudes(t1=t1, t2=t2, converged=True)
    Hc = mb.modal_hamiltonian_compact(mb.ho_hamiltonian_full(ff, n_ho), rot)
    exact_cm = np.linalg.eigvalsh(Hc)[0].real / CM2AU

    occs = list(product(range(n_modals), repeat=n_modes))

    def onehot(o):
        bits = []
        for v in o:
            blk = [0] * n_modals
            blk[v] = 1
            bits.extend(blk)
        idx = 0
        for b in bits:
            idx = (idx << 1) | b
        return idx

    prod_idx = np.array([onehot(o) for o in occs])
    ref = build_vscf_reference_state(n_modes, n_modals, 0)
    print(f"exact = {exact_cm:.2f} cm^-1,  {len(occs)} one-hot configs")

    circ = build_vim_ucj(ref, n_modes, n_modals, layers=2, n_pairs=4, t2=amps.t2)
    x0 = imucj_init_vector(amps.t2, n_modes, n_modals, 2, 4)

    def energy(x):
        a = np.asarray(Statevector(_assign(circ, x)).reverse_qargs().data)[prod_idx]
        a = a / np.linalg.norm(a)
        return float(np.real(np.vdot(a, Hc @ a))) / CM2AU

    xs = minimize(energy, x0, method="COBYLA", options={"maxiter": 2, "rhobeg": 0.2}).x
    a = np.asarray(Statevector(_assign(circ, xs)).reverse_qargs().data)[prod_idx]
    p = np.abs(a) ** 2
    p /= p.sum()
    var_gap = energy(xs) - exact_cm

    shots = 4096
    rng = np.random.default_rng(20260714)
    counts = rng.multinomial(shots, p)
    sampled = np.nonzero(counts)[0]
    print(f"variational gap after 2 steps: {var_gap:.1f} cm^-1  (far from converged)")
    print(f"distinct sampled configs: {len(sampled)}")

    print("\n=== 3. Recover by growing the diagonalization subspace ===\n")
    parents = [qa.config_to_index(occs[i], n_modals) for i in sampled]
    weights = {qa.config_to_index(occs[i], n_modals): float(p[i]) for i in sampled}
    trace = qa.iterative_subspace_growth(
        Hc,
        parents,
        n_modes,
        n_modals,
        max_rounds=5,
        tol_cm=1e-2,
        cm2au=CM2AU,
        max_size=25,
        weights=weights,
    )
    for nb, e in trace:
        print(f"  subspace {nb:3d} configs -> gap {e/CM2AU - exact_cm:7.2f} cm^-1")
    print(
        f"\nrecovered gap: {trace[-1][1]/CM2AU - exact_cm:.2f} cm^-1  "
        f"(variational was {var_gap:.1f})"
    )

    if args.noisy:
        print("\n=== 4. Noisy-hardware benchmark (run_noisy_benchmark.py) ===\n")
        cmd = [sys.executable, "run_noisy_benchmark.py"]
        print(f"=== running: {' '.join(cmd)}  (cwd={_DRIVERS_DIR}) ===")
        proc = subprocess.run(cmd, cwd=str(_DRIVERS_DIR))
        if proc.returncode != 0:
            raise SystemExit(
                f"run_noisy_benchmark.py exited with code {proc.returncode}"
            )
        print(f"\nfigures written to {_DRIVERS_DIR}/runs/noisy_benchmark/")
    else:
        print("\n(pass --noisy to also run the noisy-hardware benchmark, ~1-2 min)")


if __name__ == "__main__":
    main()
