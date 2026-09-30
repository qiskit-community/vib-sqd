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
One excited-state result per molecule (H2O/CH2O/CH2ClF), at modals=2 --
the only modal count where ALL THREE molecules' qubit counts (6/12/18q)
are still noiseless-statevector-executable, keeping the comparison uniform.

Each molecule's excited state is its own lowest-frequency fundamental
(molecule.LOWEST_FUNDAMENTAL_MODE) excited to absolute modal 1, via a
separate MidasCpp ``#3 Occup`` VCC run producing that state's OWN t1/t2
amplitudes (see excited_state.py) -- not the ground-state amplitudes reused
with a different reference. The ansatz circuit is built exactly like the
ground state (build_vscf_reference_state(n_modes, n_modals, 0)); only the
warm-start amplitudes and the recovery loop's Hamiltonian-lookup step are
occupation-aware (see excited_state.py's module docstring).

All five ground-state ansatze (UVCCSD, VLUCJ, VIm-uCJ, Vg-uCJ, CHC) are
covered -- extended from the original UVCCSD/VLUCJ-only ANSATZE tuple, via
build_measured_excited's ucj_correlators dispatch (same LAYERS=2, NPAIRS=8
convention as run_ch2o.py/run_ch2fcl.py's ground-state VIm-uCJ/
Vg-uCJ). --dry-run transpiles all five and records L3 CZ/depth (mirrors
each run_*.py's dry_run_cell), since the excited-state circuit shares
identical ansatz-construction code with the ground state and previously
had no measured-cost data of its own at all.

NOTE on modals=2 specifically: VCC excitation operators only ever go
*upward* from the reference (there is no de-excitation operator in the
coupled-cluster excitation manifold). At modals=2 the excited mode's
reference occupation is already the top modal (absolute 1), so that mode's
own warm-start channel is exactly zero -- a correct VCC/physics fact, not a
limitation of this code (verified separately for CH2O at modals=4, where
the excited mode's channel is a genuine nonzero ~3.3e-3, matching the raw
MidasCpp solution-vector coefficient bit-for-bit).

Usage:
    python run_excited_state_demo.py
    python run_excited_state_demo.py --modals 4 --molecules ch2o
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import ch2fcl_forcefield as _ch2fcl
import ch2o_forcefield as _ch2o
import h2o_forcefield as _h2o

import vib_sqd.basis.modal_transform as mb
from vib_sqd.circuits.initial_states import build_vscf_reference_state
from vib_sqd.circuits.ucj_correlators import (
    build_vg_ucj,
    build_vim_ucj,
    gucj_init_vector,
    imucj_init_vector,
)
from vib_sqd.classical.mop_forcefield import CM2AU, mop_to_forcefield, parse_mop
from vib_sqd.hardware.shot_tiers import shots_for_qubit_count
from vib_sqd.run.dualbasis import _ordered, build_ansatz, ensure_clean_vcc, init_vector
from vib_sqd.run.excited_state import (
    build_excited_amps,
    excited_recovery_loop,
    excited_reference_energy,
    process_excited_hardware_result,
)

MOLECULES = {
    "h2o": dict(
        mod=_h2o,
        mop=_h2o.H2O_MOP,
        n_modes=_h2o.N_MODES_FULL,
        n_ho=_h2o.N_HO,
        lowest_mode=_h2o.LOWEST_FUNDAMENTAL_MODE,
        label_prefix="h2o_3x",
        runs_dir="h2o",
    ),
    "ch2o": dict(
        mod=_ch2o,
        mop=_ch2o.CH2O_MOP,
        n_modes=_ch2o.N_MODES_FULL,
        n_ho=_ch2o.N_HO,
        lowest_mode=_ch2o.LOWEST_FUNDAMENTAL_MODE,
        label_prefix="ch2o_6x",
        runs_dir="ch2o",
    ),
    "ch2fcl": dict(
        mod=_ch2fcl,
        mop=_ch2fcl.CH2FCL_MOP,
        n_modes=_ch2fcl.N_MODES_FULL,
        n_ho=_ch2fcl.N_HO,
        lowest_mode=_ch2fcl.LOWEST_FUNDAMENTAL_MODE,
        label_prefix="ch2fcl_9x",
        runs_dir="ch2fcl",
    ),
}

ANSATZE = ("UVCCSD", "VLUCJ", "VIm-uCJ", "Vg-uCJ", "CHC")
BASIS = "modal"
LAYERS, NPAIRS = 2, 8
REP_ROOT = _HERE / "report"


def _backend_suffix(backend_name):
    """Empty for ibm_kingston, "_<backend>" otherwise -- see run_ch2o.py's
    _backend_suffix docstring for why this exists (prevents a second
    backend's dry-run/submit output from silently overwriting the first
    backend's same-named file)."""
    return "" if backend_name == "ibm_kingston" else f"_{backend_name}"


def dry_run_cell(measured, backend_name, outdir=None, basis=BASIS):
    """Transpile against the REAL target backend (same interaction-aware,
    error-aware path as submit_excited_hardware_batch -- see
    transpile_interaction_aware's docstring in vib_sqd.hardware.
    transpilation for why this beats a forced straight-line layout) and
    record L3 CZ/depth per ansatz, so --dry-run previews what --submit will
    actually produce. Requires reachable backend/credentials but spends no
    QPU quota. If outdir is given, also serializes each transpiled ISA
    circuit to excited[_ho]_<ansatz>[_backend]_isa.qpy in that directory
    (basis suffix omitted for modal, the original/default basis, so
    existing filenames from before the HO-basis extension are unaffected;
    backend suffix omitted for ibm_kingston, the original/default
    backend, likewise)."""
    import os

    if not (
        os.environ.get("QISKIT_IBM_TOKEN") and os.environ.get("QISKIT_IBM_INSTANCE")
    ):
        raise SystemExit(
            "Set QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE before --dry-run "
            "(a real backend.target is needed for an accurate preview)."
        )
    from qiskit import qpy
    from qiskit_ibm_runtime import QiskitRuntimeService

    from vib_sqd.hardware.transpilation import transpile_interaction_aware

    svc = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["QISKIT_IBM_TOKEN"],
        instance=os.environ["QISKIT_IBM_INSTANCE"],
    )
    backend = svc.backend(backend_name)
    basis_suffix = "" if basis == "modal" else f"_{basis}"
    backend_suffix = _backend_suffix(backend_name)
    rows = []
    for name, m in measured:
        isa = transpile_interaction_aware(m, backend, optimization_level=3)
        rows.append(
            {
                "ansatz": name,
                "basis": basis,
                "n_qubits": m.num_qubits,
                "backend": backend_name,
                "l3_cz": int(isa.count_ops().get("cz", 0)),
                "l3_depth": int(isa.depth()),
            }
        )
        if outdir is not None:
            ansatz_key = name.lower().replace("-", "").replace("_", "")
            with open(
                outdir / f"excited{basis_suffix}_{ansatz_key}{backend_suffix}_isa.qpy",
                "wb",
            ) as fh:
                qpy.dump(isa, fh)
    return rows


def ideal_statevector_counts(circ_with_measure, n_shots, seed):
    from qiskit_aer import AerSimulator

    circ_no_measure = circ_with_measure.remove_final_measurements(inplace=False)
    nq = circ_no_measure.num_qubits
    circ_sv = circ_no_measure.copy()
    circ_sv.save_statevector()
    backend = AerSimulator(method="statevector")
    result = backend.run(circ_sv).result()
    sv = result.get_statevector()
    psi = np.asarray(sv.reverse_qargs().data)
    p = np.abs(psi) ** 2
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_shots, p)
    counts = {}
    for idx, c in enumerate(draws):
        if c == 0:
            continue
        counts[format(idx, f"0{nq}b")] = int(c)
    return counts


def submit_excited_hardware_batch(
    measured, backend_name, shots, outdir=None, basis=BASIS, backend=None
):
    """Submit the excited-state ansatz circuits to real hardware -- amplitude-
    agnostic, so this is identical to every run_*.py's own
    submit_hardware_batch (kept as one copy here rather than four, since
    unlike ground-state processing, nothing else about it is molecule- or
    excited/ground-specific). Uses transpile_interaction_aware (no forced
    initial_layout -- see that function's docstring). If outdir is given,
    also serializes each submitted ISA circuit to
    excited[_ho]_<ansatz>_submitted.qpy in that directory (basis suffix
    omitted for modal, same convention as dry_run_cell). `backend`: pass
    an already-fetched IBMBackend to reuse instead of calling
    svc.backend(backend_name) again -- see run_ch2o.py's
    submit_hardware_batch docstring for why this is required inside an
    active Session/Batch context."""
    import os

    if not (
        os.environ.get("QISKIT_IBM_TOKEN") and os.environ.get("QISKIT_IBM_INSTANCE")
    ):
        raise SystemExit(
            "Set QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE before --submit."
        )
    from qiskit import qpy
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    from vib_sqd.hardware.suppression import (
        RuntimeSuppressionConfig,
        apply_sampler_suppression_options,
    )
    from vib_sqd.hardware.transpilation import transpile_interaction_aware

    if backend is None:
        svc = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=os.environ["QISKIT_IBM_TOKEN"],
            instance=os.environ["QISKIT_IBM_INSTANCE"],
        )
        backend = svc.backend(backend_name)
    basis_suffix = "" if basis == "modal" else f"_{basis}"
    backend_suffix = _backend_suffix(backend_name)
    names = [n for n, _ in measured]
    isa = [
        transpile_interaction_aware(m, backend, optimization_level=3)
        for _, m in measured
    ]
    if outdir is not None:
        for name, circ in zip(names, isa):
            ansatz_key = name.lower().replace("-", "").replace("_", "")
            with open(
                outdir
                / f"excited{basis_suffix}_{ansatz_key}{backend_suffix}_submitted.qpy",
                "wb",
            ) as fh:
                qpy.dump(circ, fh)
    sampler = Sampler(mode=backend)
    apply_sampler_suppression_options(
        sampler,
        RuntimeSuppressionConfig(
            enable_dd=True,
            dd_sequence_type="XX",
            enable_pauli_twirling=True,
            num_randomizations=8,
        ),
    )
    job = sampler.run(isa, shots=shots)
    print(f"  job_id={job.job_id()} shots={shots}")
    return job


def build_measured_excited(ansatze, ref, n_modes, n_modals, a, scale=1.0):
    """Build + warm-start-bind + measure the requested excited-state ansatz
    circuits. Mirrors run_ch2o.py's build_measured dispatch (UVCCSD/VLUCJ/CHC
    via build_ansatz+init_vector, VIm-uCJ/Vg-uCJ via the extra n_pairs/t2
    builder signature) -- the only thing occupation-aware about any of this
    is `a` (the excited state's own VCCAmplitudes, from build_excited_amps),
    not the circuit-construction code itself. `scale` only affects
    VIm-uCJ/Vg-uCJ's arctan(scale*t2) warm-start angle, same as the
    ground-state build_measured."""
    builders = {
        "UVCCSD": lambda: (
            build_ansatz("uvccsd", ref, n_modes, n_modals, layers=1),
            init_vector("uvccsd", a, n_modes, n_modals, 1),
        ),
        "VLUCJ": lambda: (
            build_ansatz("vlucj", ref, n_modes, n_modals, layers=4),
            init_vector("vlucj", a, n_modes, n_modals, 4),
        ),
        "CHC": lambda: (
            build_ansatz("chc", ref, n_modes, n_modals, layers=4),
            init_vector("chc", a, n_modes, n_modals, 4),
        ),
        "VIm-uCJ": lambda: (
            build_vim_ucj(
                ref, n_modes, n_modals, layers=LAYERS, n_pairs=NPAIRS, t2=a.t2
            ),
            imucj_init_vector(a.t2, n_modes, n_modals, LAYERS, NPAIRS, scale=scale),
        ),
        "Vg-uCJ": lambda: (
            build_vg_ucj(
                ref, n_modes, n_modals, layers=LAYERS, n_pairs=NPAIRS, t2=a.t2
            ),
            gucj_init_vector(a.t2, n_modes, n_modals, LAYERS, NPAIRS, scale=scale),
        ),
    }
    measured = []
    for name in ansatze:
        circ, x0 = builders[name]()
        m = circ.assign_parameters(
            dict(zip(_ordered(circ), np.asarray(x0, float)))
        ).copy()
        m.measure_all()
        measured.append((name, m))
    return measured


def run_one(
    name,
    cfg,
    n_modals,
    shots=None,
    seed=20260722,
    submit=False,
    backend="ibm_kingston",
    dry_run=False,
    ansatze=ANSATZE,
    scale=1.0,
    basis=BASIS,
):
    n_modes, n_ho = cfg["n_modes"], cfg["n_ho"]
    label = f"{cfg['label_prefix']}{n_modals}"
    runs_dir = str(_HERE / "runs" / cfg["runs_dir"])
    nq = n_modes * n_modals
    if shots is None:
        shots = shots_for_qubit_count(nq)

    mout, ffmop = ensure_clean_vcc(label, cfg["mop"], None, n_modals, runs_dir)
    ff = mop_to_forcefield(parse_mop(ffmop))
    rot = mb.build_modal_rotation(mout, n_modes, n_ho, n_modals)
    # H_ho (the full compact HO Hamiltonian) is never built at all, in
    # either basis: excited_reference_energy uses the molecule module's
    # own ho_diagonal_element (one element, no matrix), and
    # excited_recovery_loop uses ho_submatrix (only the small submatrix
    # actually needed) via ho_submatrix_fn below -- both direct
    # element-by-element evaluations, the same OOM-avoiding fix already
    # applied to every run_*.py's ground-state recovery_loop this session.
    # Building the full matrix would be pure waste for modal basis and an
    # OOM risk in HO basis at 9 modes (CH2ClF).
    H_ho = None
    mod = cfg["mod"]

    def ho_diag(occ):
        return float(mod.ho_diagonal_element(ff, n_modals, tuple(occ)) / CM2AU)

    occupation, amps_exc, mout_exc = build_excited_amps(
        ffmop, n_modes, n_modals, cfg["lowest_mode"], rot, runs_dir, label
    )

    if basis == "modal":
        ground_E = excited_reference_energy(
            basis, [0] * n_modes, ff, rot, H_ho, n_modals, n_ho
        )
        exc_ref_E = excited_reference_energy(
            basis, occupation, ff, rot, H_ho, n_modals, n_ho
        )
    else:
        ground_E = ho_diag([0] * n_modes)
        exc_ref_E = ho_diag(occupation)

    ref = build_vscf_reference_state(n_modes, n_modals, 0)
    measured = build_measured_excited(
        ansatze, ref, n_modes, n_modals, amps_exc[basis], scale=scale
    )

    outdir = REP_ROOT / f"{name}_excited"
    outdir.mkdir(parents=True, exist_ok=True)
    basis_suffix = "" if basis == "modal" else f"_{basis}"
    backend_suffix = _backend_suffix(backend)

    if dry_run:
        rows = dry_run_cell(measured, backend, outdir=outdir, basis=basis)
        for r in rows:
            print(
                f"  {name} {r['ansatz']:8s} qubits={r['n_qubits']} "
                f"L3_CZ={r['l3_cz']} L3_depth={r['l3_depth']}"
            )
        (
            outdir
            / f"excited{basis_suffix}_modals{n_modals}{backend_suffix}_dryrun.json"
        ).write_text(json.dumps(rows, indent=2))
        return rows

    if submit:
        job = submit_excited_hardware_batch(
            measured, backend, shots, outdir=outdir, basis=basis
        )
        names = [n for n, _ in measured]
        out = process_excited_hardware_result(
            job,
            names,
            basis,
            occupation,
            ff,
            rot,
            H_ho,
            n_modes,
            n_modals,
            n_ho,
            outdir,
            shots,
            scale=scale,
            ho_submatrix_fn=mod.ho_submatrix if basis != "modal" else None,
        )
        for name_, v in out["variants"].items():
            print(
                f"  {name} {name_:8s} excited-state recovered E = "
                f"{v['recovered_energy_cm']:.2f} cm-1 (ref {exc_ref_E:.2f}, "
                f"ground {ground_E:.2f}, gap {exc_ref_E-ground_E:.2f})"
            )
        rows = [
            {
                "ansatz": n_,
                "recovered_energy_cm": v["recovered_energy_cm"],
                "iters": v["recovery_iters"],
            }
            for n_, v in out["variants"].items()
        ]
    else:
        rows = []
        for ansatz, m in measured:
            counts = ideal_statevector_counts(m, shots, seed)
            best, iters, rec, occ = excited_recovery_loop(
                basis,
                counts,
                occupation,
                ff,
                rot,
                H_ho,
                n_modes,
                n_modals,
                n_ho,
                seed=seed,
                ho_submatrix_fn=mod.ho_submatrix if basis != "modal" else None,
            )
            rows.append({"ansatz": ansatz, "recovered_energy_cm": best, "iters": iters})
            print(
                f"  {name} {ansatz:8s} excited-state recovered E = {best:.2f} cm-1 "
                f"(ref {exc_ref_E:.2f}, ground {ground_E:.2f}, gap {exc_ref_E-ground_E:.2f})"
            )

    result = {
        "molecule": name,
        "n_modals": n_modals,
        "n_modes": n_modes,
        "basis": basis,
        "excited_mode": cfg["lowest_mode"],
        "occupation": list(occupation),
        "ground_reference_energy_cm": ground_E,
        "excited_reference_energy_cm": exc_ref_E,
        "excitation_energy_cm": exc_ref_E - ground_E,
        "ansatze": rows,
    }
    (
        outdir / f"excited{basis_suffix}_modals{n_modals}{backend_suffix}.json"
    ).write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--modals", type=int, default=2)
    ap.add_argument(
        "--molecules", nargs="+", default=list(MOLECULES), choices=list(MOLECULES)
    )
    ap.add_argument(
        "--shots",
        type=int,
        default=None,
        help="overrides the qubit-count-tiered default "
        "(100k up to 24q, 500k 24-56q, 1M 56-72q)",
    )
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument(
        "--ansatze", nargs="+", default=list(ANSATZE), choices=list(ANSATZE)
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="VIm-uCJ/Vg-uCJ arctan(scale*t2) warm-start scale "
        "(see run_scale_tuning.py for the swept grid)",
    )
    ap.add_argument(
        "--basis",
        default=BASIS,
        choices=("modal", "ho"),
        help="modal (VSCF, default -- the original/only basis "
        "this runner supported until this flag was added) "
        "or ho (harmonic-oscillator basis, added to mirror "
        "the ground-state HO-vs-modal comparison)",
    )
    args = ap.parse_args()

    for name in args.molecules:
        print(f"=== {name} (modals={args.modals}, basis={args.basis}) ===")
        run_one(
            name,
            MOLECULES[name],
            args.modals,
            shots=args.shots,
            submit=args.submit,
            backend=args.backend,
            dry_run=args.dry_run,
            ansatze=tuple(args.ansatze),
            scale=args.scale,
            basis=args.basis,
        )


if __name__ == "__main__":
    main()
