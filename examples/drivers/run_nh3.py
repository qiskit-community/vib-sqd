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
NH3 (ammonia) vibrational-SQD sweep.

6 modes x {2,4,6} modals -> {12,24,36} qubits, HO basis vs modal (VSCF) basis,
5 ansatze (UVCCSD/VLUCJ/VIm-uCJ/Vg-uCJ/CHC) + a uniform-random classical
baseline, on a noiseless simulator (statevector-derived ideal sampling, where
tractable) and real ibm_kobe hardware (100,000 shots per ansatz).

NH3 has C3v symmetry: modes Q2/Q3 and Q4/Q5 are exactly degenerate
(nh3_forcefield.DEGENERATE_MODE_PAIRS). NO symmetry reduction is applied --
all 6 modes are treated uniformly, exactly like CH2O, matching the classical
VHCI reference calculation's own treatment.

One classical VCC run per modal count (cached); both bases reuse the SAME VCC
amplitudes -- only the Hamiltonian construction and the ansatz's warm-start
rotation differ:
  * modal basis: sparse build_subspace_hamiltonian with the real VSCF `rot`
    (scales to any qubit count, never forms a dense M^N matrix).
  * HO basis: sparse ho_hamiltonian_sparse(ff, n_modals) (nh3_forcefield.py) --
    the dense compact HO Hamiltonian (n_modals**6) is intractable to build/
    diagonalize densely at n_modals=6 (46656^2, ~17GB); the sparse builder
    reproduces it to machine precision at n_modals in {2,4} and is used
    uniformly at all three modal counts for consistency.

"Simulator" cells:
  * 12q/24q: exact Statevector -> multinomial-sampled ideal counts (true
    noiseless circuit execution).
  * 36q: NOT executable (2^36 statevector ~550GB) -- reports the classical
    reference-config energy only, clearly labeled, mirroring how
    run_glycine64.py / run_glycine100.py never attempt a noiseless statevector
    at 64q/100q either.

Usage:
  python run_nh3.py --modals 2 --basis modal --dry-run
  python run_nh3.py --modals 2 --basis modal --sim
  python run_nh3.py --modals 2 --basis modal --submit --backend ibm_kobe --shots 100000
  python run_nh3.py --modals 2 --basis modal --uniform-baseline
  python run_nh3.py --modals 2 --basis modal --fundamentals
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# The Loewdin-orthonormalization step in build_modal_rotation hits a harmless
# (verified: resulting H stays Hermitian to 1e-18) divide-by-zero/overflow in
# a near-singular eigenvalue during w**-0.5; silence it here.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import vib_sqd.basis.modal_transform as mb
from vib_sqd.classical.mop_forcefield import parse_mop, mop_to_forcefield, CM2AU
from vib_sqd.classical.vcc_extractor import extract_amplitudes
from vib_sqd.classical.vcc_solver import VCCAmplitudes
from vib_sqd.circuits.initial_states import build_vscf_reference_state
from vib_sqd.circuits.ucj_correlators import (
    build_vim_ucj,
    build_vg_ucj,
    imucj_init_vector,
    gucj_init_vector,
)
from vib_sqd.sqd.subspace_hamiltonian import build_subspace_hamiltonian
from vib_sqd.sqd.vibrational_recovery import (
    recover_direct_onehot_distribution,
    occupancies_from_ground_state,
    occupancies_from_raw_counts,
    initial_uniform_modal_occupancies,
    sample_basis_from_distribution,
    _modal_indices_from_onehot,
    _onehot_from_modal_indices,
)
from vib_sqd.run.dualbasis import _ordered, build_ansatz, init_vector
from vib_sqd.run.pipeline import run_midascpp
from vib_sqd.classical.mop_subset import write_subset_mop
from vib_sqd.classical.mop_forcefield import parse_mop as _parse_mop
from nh3_forcefield import (
    NH3_MOP,
    N_MODES_FULL,
    N_HO,
    ho_hamiltonian_sparse,
    compact_index,
)

N_MODES = N_MODES_FULL
MODAL_COUNTS = (2, 4, 6)
ANSATZE = ("UVCCSD", "VLUCJ", "VIm-uCJ", "Vg-uCJ", "CHC")
BASES = ("ho", "modal")
LAYERS, NPAIRS = 2, 8
BASIS_SIZE = 200
MAX_ITER, CONV_CM, SEED = 30, 0.1, 20260722

FUNDAMENTAL_CONFIGS = {
    "100000": (1, 0, 0, 0, 0, 0),
    "010000": (0, 1, 0, 0, 0, 0),
    "001000": (0, 0, 1, 0, 0, 0),
    "000100": (0, 0, 0, 1, 0, 0),
    "000010": (0, 0, 0, 0, 1, 0),
    "000001": (0, 0, 0, 0, 0, 1),
}

REP_ROOT = _HERE / "report"


def _rep_dir(n_modals):
    return REP_ROOT / f"nh3_{N_MODES * n_modals}q"


def ensure_clean_vcc_nh3(label, mop_path, n_modals, outroot):
    """NH3-specific VCC bootstrap: forces ground_state_only=True (unlike
    run_dualbasis_experiment.ensure_clean_vcc, which only does this at
    n_modes>=7). NH3's exactly-degenerate E-symmetry mode pairs (Q2/Q3,
    Q4/Q5) make the batched OccGroundState+OccAllFund VSCF fail: the excited-
    state VSCF for ONE mode of each degenerate pair oscillates indefinitely
    (confirmed: energy bounces between ~3.59e-2 and ~3.94e-2 a.u. for 100
    iterations with no decaying trend -- a genuine SCF instability at the
    exact degeneracy, not a slow-convergence issue MaxIter could fix).
    Confirmed separately: (1) the GROUND-STATE VSCF in that same batch run
    converges cleanly regardless (this function's amplitudes are therefore
    unaffected by the degenerate-mode failure); (2) each individual excited
    state, including the degenerate pair, converges cleanly in ~6 iterations
    when solved as an ISOLATED single-state VSCF (occup=[...]) rather than in
    the OccAllFund batch -- used by run_nh3.py's --fundamentals path instead
    of a batch call. This function requests ONLY the ground state, which is
    all the main sweep needs (VCC amplitudes for ansatz warm-starting)."""
    run_dir = Path(outroot) / f"vcc_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = _parse_mop(mop_path)
    src = Path(mop_path)
    ff_mop = run_dir / "midas" / src.name
    mout = run_dir / "midas" / "input.mout"
    if mout.exists() and "PROD-ALLHO" in mout.read_text():
        return mout, ff_mop
    n_modes = len(getattr(doc, "mode_names", []) or [])
    m = run_midascpp(
        src,
        run_dir / "midas",
        n_modals=n_modals,
        use_scaling_freqs=doc.scale_factors is not None,
        vcc_method="VCC[2]",
        n_modes=n_modes,
        ground_state_only=True,
    )
    return m, ff_mop


def build_amps_and_ho(n_modals):
    """One classical VCC run per modal count; returns ff, rot (modal-basis VSCF
    rotation), amps dict {"modal":.., "ho":..}, and H_ho (sparse compact HO
    Hamiltonian, dim n_modals**N_MODES)."""
    label = f"nh3_6x{n_modals}"
    mout, ffmop = ensure_clean_vcc_nh3(
        label, NH3_MOP, n_modals, str(_HERE / "runs" / "nh3")
    )
    ff = mop_to_forcefield(parse_mop(ffmop))
    rot = mb.build_modal_rotation(mout, N_MODES, N_HO, n_modals)
    t1_mod, t2_mod, meta = extract_amplitudes(Path(mout), N_MODES, n_modals)
    t1_ho, t2_ho = mb.ho_amplitudes_from_modal(t1_mod, t2_mod, rot)
    amps = {
        "modal": VCCAmplitudes(t1=t1_mod, t2=t2_mod, converged=True, metadata=meta),
        "ho": VCCAmplitudes(t1=t1_ho, t2=t2_ho, converged=True, metadata=meta),
    }
    H_ho = ho_hamiltonian_sparse(ff, n_modals)
    return ff, rot, amps, H_ho


def reference_config_energy(basis, ff, rot, H_ho, n_modals):
    """All-zero-config energy, basis-specific."""
    if basis == "modal":
        H = build_subspace_hamiltonian([tuple([0] * N_MODES)], ff, rot, N_HO)
        return float(H[0, 0].real / CM2AU)
    idx0 = compact_index(tuple([0] * N_MODES), n_modals)
    return float(H_ho[idx0, idx0].real / CM2AU)


def config_energy(basis, cfg, ff, rot, H_ho, n_modals):
    """Diagonal (uncoupled-config) energy for an arbitrary N_MODES-length
    occupation tuple, basis-specific. reference_config_energy(...) is the
    cfg=(0,)*N_MODES special case of this; kept as a separate function (not a
    delegating wrapper) so the already-working ground-state path is never
    touched. Used for NH3's --fundamentals flag, since (unlike CH2O) no
    literature reference exists for NH3's singly-excited fundamentals -- these
    are reported as-computed only."""
    if basis == "modal":
        H = build_subspace_hamiltonian([tuple(cfg)], ff, rot, N_HO)
        return float(H[0, 0].real / CM2AU)
    idx = compact_index(tuple(cfg), n_modals)
    return float(H_ho[idx, idx].real / CM2AU)


def build_measured(basis, amps, n_modals, ansatze=ANSATZE):
    """Build + warm-start-bind + measure the requested ansatz circuits for one
    basis. Returns list of (name, bound_circuit_with_measure)."""
    ref = build_vscf_reference_state(N_MODES, n_modals, 0)
    a = amps[basis]
    builders = {
        "UVCCSD": lambda: (
            build_ansatz("uvccsd", ref, N_MODES, n_modals, layers=1),
            init_vector("uvccsd", a, N_MODES, n_modals, 1),
        ),
        "VIm-uCJ": lambda: (
            build_vim_ucj(
                ref, N_MODES, n_modals, layers=LAYERS, n_pairs=NPAIRS, t2=a.t2
            ),
            imucj_init_vector(a.t2, N_MODES, n_modals, LAYERS, NPAIRS),
        ),
        "Vg-uCJ": lambda: (
            build_vg_ucj(
                ref, N_MODES, n_modals, layers=LAYERS, n_pairs=NPAIRS, t2=a.t2
            ),
            gucj_init_vector(a.t2, N_MODES, n_modals, LAYERS, NPAIRS),
        ),
        "VLUCJ": lambda: (
            build_ansatz("vlucj", ref, N_MODES, n_modals, layers=4),
            init_vector("vlucj", a, N_MODES, n_modals, 4),
        ),
        "CHC": lambda: (
            build_ansatz("chc", ref, N_MODES, n_modals, layers=4),
            init_vector("chc", a, N_MODES, n_modals, 4),
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


def recovery_loop(
    basis, raw_counts, ff, rot, H_ho, n_modals, basis_size=BASIS_SIZE, seed=SEED
):
    """Self-consistent SQD recovery, basis-dispatched Hamiltonian lookup.
    Returns (best_energy_cm, iterations, after_distribution, after_occupancies)."""
    vmax = n_modals - 1
    occ = initial_uniform_modal_occupancies(N_MODES, n_modals, vmax)
    rng = np.random.default_rng(seed)
    best = float("inf")
    prev = None
    iters = 0
    rec = None
    for _ in range(MAX_ITER):
        iters += 1
        rec = recover_direct_onehot_distribution(
            raw_counts,
            avg_modal_occupancies=occ,
            n_modes=N_MODES,
            n_modals=n_modals,
            vmax=vmax,
            seed=int(rng.integers(0, 2**31 - 1)),
            randomize=True,
        )
        bb = sample_basis_from_distribution(
            rec,
            basis_size,
            rng,
            always_include=sorted(rec, key=rec.get, reverse=True)[:basis_size],
        )
        cfgs = [tuple(_modal_indices_from_onehot(b, N_MODES, n_modals)) for b in bb]
        if basis == "modal":
            H = build_subspace_hamiltonian(cfgs, ff, rot, N_HO)
        else:
            idxs = [compact_index(c, n_modals) for c in cfgs]
            H = H_ho[np.ix_(idxs, idxs)].toarray()
        w, V = np.linalg.eigh(H)
        e = w[0].real / CM2AU
        best = min(best, e)
        occ = occupancies_from_ground_state(bb, V[:, 0], N_MODES, n_modals, vmax)
        if prev is not None and abs(e - prev) < CONV_CM:
            break
        prev = e
    return best, iters, rec, occ


def bits_big(k, nq):
    return k.replace(" ", "")[-nq:].zfill(nq)[::-1]


def onehot_decode(bb, n_modes, n_modals):
    occ = []
    for m in range(n_modes):
        blk = bb[m * n_modals : (m + 1) * n_modals]
        if blk.count("1") != 1:
            return None
        occ.append(blk.index("1"))
    return tuple(occ)


def dry_run_cell(basis, n_modals, measured):
    """Transpile + record L3 CZ/depth per ansatz (generic Heron basis gates)."""
    from qiskit import transpile

    rows = []
    for name, m in measured:
        isa = transpile(m, basis_gates=["rz", "sx", "x", "cz"], optimization_level=3)
        rows.append(
            {
                "ansatz": name,
                "basis": basis,
                "n_modals": n_modals,
                "n_qubits": m.num_qubits,
                "l3_cz": int(isa.count_ops().get("cz", 0)),
                "l3_depth": int(isa.depth()),
            }
        )
    return rows


def ideal_statevector_counts(circ_with_measure, n_shots, seed):
    """Exact noiseless statevector -> multinomial-sampled counts dict. Feasible
    up to ~24 qubits (2^24 complex128 ~ 256MB); caller must not use above that.

    Uses Aer's AerSimulator(method="statevector") rather than Qiskit's default
    Statevector(circuit) constructor -- the default does naive gate-by-gate
    evolution and is impractically slow (>90s, did not complete) for a deep
    ~11,000-gate 24-qubit circuit; Aer's tensor-contraction backend does the
    same computation in ~1.3s. reverse_qargs() is still applied (this
    pipeline's universal convention, see run_pipeline.py:_energy) so amplitude
    index i's binary representation has qubit 0 as the most-significant bit --
    i.e. bitstring position 0 is mode 0's block, matching
    _modal_indices_from_onehot's expectation."""
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
        bb = format(idx, f"0{nq}b")
        counts[bb] = int(c)
    return counts


def submit_hardware_batch(basis, n_modals, measured, backend_name, shots):
    """One job covering all requested ansatz circuits for this (basis, modal
    count) cell -- mirrors run_glycine64.py's submit branch."""
    if not (
        os.environ.get("QISKIT_IBM_TOKEN") and os.environ.get("QISKIT_IBM_INSTANCE")
    ):
        raise SystemExit(
            "Set QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE before --submit."
        )
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from vib_sqd.hardware.transpilation import choose_best_connected_line
    from vib_sqd.hardware.suppression import (
        RuntimeSuppressionConfig,
        apply_sampler_suppression_options,
    )

    nq = n_modals * N_MODES
    svc = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["QISKIT_IBM_TOKEN"],
        instance=os.environ["QISKIT_IBM_INSTANCE"],
    )
    backend = svc.backend(backend_name)
    line = choose_best_connected_line(backend, logical_width=nq)
    print(f"  best {nq}q line={line}")
    pm = generate_preset_pass_manager(
        target=backend.target, optimization_level=3, initial_layout=line
    )
    isa = [pm.run(m) for _, m in measured]
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


def process_hardware_result(job, names, basis, ff, rot, H_ho, n_modals, rep_dir, shots):
    """Retrieve job.result(), decode + recover per ansatz, write per-cell JSON
    artifacts (results/recovered/beforeafter/dryrun already written elsewhere)."""
    nq = n_modals * N_MODES
    result = job.result()
    out = {
        "backend": job.backend().name if hasattr(job, "backend") else None,
        "job_id": job.job_id(),
        "basis": basis,
        "n_modes": N_MODES,
        "n_modals": n_modals,
        "shots": shots,
        "variants": {},
    }
    for name, pub in zip(names, result):
        d = pub.data
        reg = [
            x
            for x in dir(d)
            if not x.startswith("_") and hasattr(getattr(d, x), "get_counts")
        ][0]
        counts = getattr(d, reg).get_counts()
        raw = {}
        kept = 0
        total = sum(counts.values())
        for k, c in counts.items():
            bb = bits_big(k, nq)
            raw[bb] = raw.get(bb, 0.0) + float(c)
            if onehot_decode(bb, N_MODES, n_modals) is not None:
                kept += c
        before = dict(sorted(raw.items(), key=lambda kv: -kv[1])[:50])
        before_total = sum(before.values())
        before = (
            {k: v / before_total for k, v in before.items()} if before_total else {}
        )
        rec_e, iters, after, after_occ = recovery_loop(
            basis, raw, ff, rot, H_ho, n_modals
        )
        vmax = n_modals - 1
        before_occ = occupancies_from_raw_counts(raw, N_MODES, n_modals, vmax)
        out["variants"][name] = {
            "onehot_retained_frac": kept / total if total else 0.0,
            "recovered_energy_cm": float(rec_e),
            "recovery_iters": iters,
        }
        ansatz_key = name.lower().replace("-", "").replace("_", "")
        (rep_dir / f"{basis}_{ansatz_key}_beforeafter.json").write_text(
            json.dumps(
                {
                    "before": before,
                    "after": after,
                    "converged_cm": float(rec_e),
                    "kept_frac": kept / total if total else 0.0,
                    "before_occ": before_occ.tolist(),
                    "after_occ": after_occ.tolist(),
                },
                indent=2,
            )
        )
        print(
            f"  {name:8s} one-hot retained={kept/total*100:6.3f}%  "
            f"recovered E={rec_e:9.2f} cm^-1",
            flush=True,
        )
    (rep_dir / f"{basis}_results.json").write_text(
        json.dumps(out, indent=2, default=float)
    )
    print(f"wrote {rep_dir}/{basis}_results.json")
    return out


def uniform_baseline(n_modals, n_shots, seed=SEED):
    """One raw-counts draw per modal count (basis-independent sampling)."""
    from quantum_vs_uniform import uniform_counts

    return uniform_counts(n_shots, N_MODES, n_modals, N_MODES * n_modals, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modals", type=int, choices=MODAL_COUNTS, required=True)
    ap.add_argument("--basis", choices=BASES, required=True)
    ap.add_argument("--backend", default="ibm_kobe")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--uniform-baseline", action="store_true")
    ap.add_argument("--fundamentals", action="store_true")
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--basis-size", type=int, default=BASIS_SIZE)
    ap.add_argument("--ansatze", nargs="+", default=list(ANSATZE), choices=ANSATZE)
    a = ap.parse_args()

    rep_dir = _rep_dir(a.modals)
    rep_dir.mkdir(parents=True, exist_ok=True)
    nq = a.modals * N_MODES

    print(f"NH3 {N_MODES} modes x {a.modals} modal = {nq} qubits, basis={a.basis}")
    ff, rot, amps, H_ho = build_amps_and_ho(a.modals)
    ref_cm = reference_config_energy(a.basis, ff, rot, H_ho, a.modals)
    print(f"  reference-config energy ({a.basis} basis) = {ref_cm:.2f} cm^-1")

    if a.fundamentals:
        rows = {"basis": a.basis, "n_modals": a.modals, "000000": ref_cm}
        for label, cfg in FUNDAMENTAL_CONFIGS.items():
            rows[label] = config_energy(a.basis, cfg, ff, rot, H_ho, a.modals)
        (rep_dir / f"{a.basis}_fundamentals.json").write_text(
            json.dumps(rows, indent=2)
        )
        print(f"  fundamentals ({a.basis} basis, n_modals={a.modals}):")
        for label in ["000000"] + list(FUNDAMENTAL_CONFIGS):
            print(f"    v={label}: {rows[label]:.2f} cm^-1")
        return

    if a.uniform_baseline:
        raw = uniform_baseline(a.modals, a.shots)
        rec_e, iters, after, after_occ = recovery_loop(
            a.basis, raw, ff, rot, H_ho, a.modals, basis_size=a.basis_size
        )
        vmax = a.modals - 1
        before_occ = occupancies_from_raw_counts(raw, N_MODES, a.modals, vmax)
        out = {
            "basis": a.basis,
            "n_modes": N_MODES,
            "n_modals": a.modals,
            "shots": a.shots,
            "reference_config_energy_cm": ref_cm,
            "recovered_energy_cm": float(rec_e),
            "recovery_iters": iters,
            "before_occ": before_occ.tolist(),
            "after_occ": after_occ.tolist(),
        }
        (rep_dir / f"{a.basis}_uniform_recovered.json").write_text(
            json.dumps(out, indent=2)
        )
        print(f"  uniform-random recovered E={rec_e:.2f} cm^-1 (ref {ref_cm:.2f})")
        return

    measured = build_measured(a.basis, amps, a.modals, a.ansatze)

    if a.dry_run:
        rows = dry_run_cell(a.basis, a.modals, measured)
        for r in rows:
            print(
                f"  {r['ansatz']:8s} qubits={r['n_qubits']} L3_CZ={r['l3_cz']} L3_depth={r['l3_depth']}"
            )
        for r in rows:
            ansatz_key = r["ansatz"].lower().replace("-", "").replace("_", "")
            (rep_dir / f"{a.basis}_{ansatz_key}_dryrun.json").write_text(
                json.dumps(r, indent=2)
            )
        return

    if a.sim:
        if nq > 24:
            out = {
                "basis": a.basis,
                "n_modes": N_MODES,
                "n_modals": a.modals,
                "note": "not executable: 2^36 statevector infeasible",
                "reference_config_energy_cm": ref_cm,
            }
            (rep_dir / f"{a.basis}_classical_reference.json").write_text(
                json.dumps(out, indent=2)
            )
            print(
                f"  36q: no synthetic circuit executed; classical reference-config "
                f"energy = {ref_cm:.2f} cm^-1 written to {a.basis}_classical_reference.json"
            )
            return
        vmax = a.modals - 1
        for name, m in measured:
            raw = ideal_statevector_counts(m, a.shots, SEED)
            rec_e, iters, after, after_occ = recovery_loop(
                a.basis, raw, ff, rot, H_ho, a.modals, basis_size=a.basis_size
            )
            before_occ = occupancies_from_raw_counts(raw, N_MODES, a.modals, vmax)
            ansatz_key = name.lower().replace("-", "").replace("_", "")
            out = {
                "ansatz": name,
                "basis": a.basis,
                "n_modes": N_MODES,
                "n_modals": a.modals,
                "shots": a.shots,
                "reference_config_energy_cm": ref_cm,
                "recovered_energy_cm": float(rec_e),
                "recovery_iters": iters,
                "before_occ": before_occ.tolist(),
                "after_occ": after_occ.tolist(),
            }
            (rep_dir / f"{a.basis}_{ansatz_key}_results.json").write_text(
                json.dumps(out, indent=2)
            )
            print(f"  {name:8s} sim recovered E={rec_e:9.2f} cm^-1 (ref {ref_cm:.2f})")
        return

    if a.submit:
        job = submit_hardware_batch(a.basis, a.modals, measured, a.backend, a.shots)
        names = [n for n, _ in measured]
        process_hardware_result(
            job, names, a.basis, ff, rot, H_ho, a.modals, rep_dir, a.shots
        )
        return

    print(
        "no action requested; pass one of --dry-run / --sim / --submit / --uniform-baseline / --fundamentals"
    )


if __name__ == "__main__":
    main()
