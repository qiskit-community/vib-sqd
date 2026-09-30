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
CH2ClF (fluorochloromethane) vibrational-SQD sweep.

9 modes x {2,4,6,8} modals -> {18,36,54,72} qubits, HO basis vs modal (VSCF)
basis, 5 ansatze (UVCCSD/VLUCJ/VIm-uCJ/Vg-uCJ/CHC) + a uniform-random
classical baseline, on a noiseless simulator (statevector-derived ideal
sampling, where tractable) and real ibm_kingston hardware.

Replaces C2H2/CNH3 in the manuscript's molecule set (both molecules' VCI
energies fail to converge with increasing HO-basis size -- see
ch2fcl_forcefield.py's module docstring).

Shots are qubit-count-tiered (shots_for_qubit_count: 100k up to 24q, 500k
24-56q, 1M 56-72q) rather than a single flat value -- pass --shots to
override with an explicit count (e.g. for a pinned reproducibility re-run).

One classical VCC run per modal count (cached); both bases reuse the SAME VCC
amplitudes -- only the Hamiltonian construction and the ansatz's warm-start
rotation differ:
  * modal basis: sparse build_subspace_hamiltonian with the real VSCF `rot`
    (scales to any qubit count, never forms a dense M^N matrix).
  * HO basis: sparse ho_hamiltonian_sparse(ff, n_modals) (ch2fcl_forcefield.py) --
    the dense compact HO Hamiltonian (n_modals**9) is intractable to build/
    diagonalize densely above n_modals=2; the sparse builder reproduces it to
    machine precision and is used uniformly at all four modal counts.

"Simulator" cells:
  * 18q: exact Statevector -> multinomial-sampled ideal counts (true
    noiseless circuit execution).
  * 36q/54q/72q: NOT executable (2^36+ statevector infeasible) -- reports the
    classical reference-config energy only, clearly labeled, mirroring
    run_ch2o.py/run_cnh3.py's >24q branch.

Usage:
  python run_ch2fcl.py --modals 2 --basis modal --dry-run
  python run_ch2fcl.py --modals 2 --basis modal --sim
  python run_ch2fcl.py --modals 2 --basis modal --submit --backend ibm_kingston
  python run_ch2fcl.py --modals 2 --basis modal --uniform-baseline
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
from vib_sqd.hardware.shot_tiers import shots_for_qubit_count
from vib_sqd.run.dualbasis import ensure_clean_vcc, _ordered, build_ansatz, init_vector
from ch2fcl_forcefield import (
    CH2FCL_MOP,
    N_MODES_FULL,
    N_HO,
    ho_hamiltonian_sparse,
    ho_diagonal_element,
    ho_submatrix,
    compact_index,
)

N_MODES = N_MODES_FULL
MODAL_COUNTS = (2, 4, 6, 8)
ANSATZE = ("UVCCSD", "VLUCJ", "VIm-uCJ", "Vg-uCJ", "CHC")
BASES = ("ho", "modal")
NPAIRS = 8
BASIS_SIZE = 200
MAX_ITER, CONV_CM, SEED = 30, 0.1, 20260722

REP_ROOT = _HERE / "report"


def _rep_dir(n_modals):
    return REP_ROOT / f"ch2fcl_{N_MODES * n_modals}q"


def build_amps_and_ho(n_modals, need_h_ho=True):
    """One classical VCC run per modal count; returns ff, rot (modal-basis VSCF
    rotation), amps dict {"modal":.., "ho":..}, and H_ho -- ALWAYS None now.

    The full sparse H_ho matrix is no longer built at all: reference_config_energy
    uses ho_diagonal_element (a single element, <0.1s) and recovery_loop uses
    ho_submatrix (only the basis_size x basis_size submatrix actually needed,
    <1s), both direct element-by-element evaluations that never form the full
    dim=n_modals**N_MODES sparse matrix. Building that full matrix via
    ho_hamiltonian_sparse's scipy.sparse.kron chain caused two independent OOM
    kills for CH2ClF at n_modals=6 (dim ~1.0e7) and was already known to take
    minutes even when it didn't crash -- confirmed 2026-09-06. `need_h_ho` is
    kept as a parameter for call-site compatibility but no longer does
    anything; H_ho is always None."""
    label = f"ch2fcl_9x{n_modals}"
    mout, ffmop = ensure_clean_vcc(
        label, CH2FCL_MOP, None, n_modals, str(_HERE / "runs" / "ch2fcl")
    )
    ff = mop_to_forcefield(parse_mop(ffmop))
    rot = mb.build_modal_rotation(mout, N_MODES, N_HO, n_modals)
    t1_mod, t2_mod, meta = extract_amplitudes(Path(mout), N_MODES, n_modals)
    t1_ho, t2_ho = mb.ho_amplitudes_from_modal(t1_mod, t2_mod, rot)
    amps = {
        "modal": VCCAmplitudes(t1=t1_mod, t2=t2_mod, converged=True, metadata=meta),
        "ho": VCCAmplitudes(t1=t1_ho, t2=t2_ho, converged=True, metadata=meta),
    }
    return ff, rot, amps, None


def reference_config_energy(basis, ff, rot, H_ho, n_modals):
    """All-zero-config energy, basis-specific. HO basis uses the direct
    diagonal-element formula (ho_diagonal_element) rather than indexing into
    H_ho, so this works even when H_ho is None (the --dry-run/lazy path).
    NOTE: the HO-basis compact Hamiltonian here is truncated to n_modals
    levels per mode (matching the qubit budget), NOT the full N_HO=11 -- same
    convention as the original ho_hamiltonian_sparse(ff, n_modals) call."""
    if basis == "modal":
        H = build_subspace_hamiltonian([tuple([0] * N_MODES)], ff, rot, N_HO)
        return float(H[0, 0].real / CM2AU)
    return float(ho_diagonal_element(ff, n_modals, tuple([0] * N_MODES)) / CM2AU)


def build_measured(basis, amps, n_modals, ansatze=ANSATZE, scale=1.0, layers=4):
    """Build + warm-start-bind + measure the requested ansatz circuits for one
    basis. Returns list of (name, bound_circuit_with_measure). `scale` only
    affects VIm-uCJ/Vg-uCJ's arctan(scale*t2) warm-start angle
    (t2_to_angle in ucj_correlators.py) -- meaningless for UVCCSD/VLUCJ/CHC,
    whose builders ignore it. `layers` uniformly controls VIm-uCJ/Vg-uCJ/
    VLUCJ/CHC together (default 4) -- all four have a genuine repeated-layer
    structure, so a fair small-to-large NISQ sweep (1, then 2, then 4) means
    running them at the SAME layer count. UVCCSD has no comparable
    stacked-layer structure and always uses layers=1, unaffected by this
    parameter."""
    ref = build_vscf_reference_state(N_MODES, n_modals, 0)
    a = amps[basis]
    L = layers
    builders = {
        "UVCCSD": lambda: (
            build_ansatz("uvccsd", ref, N_MODES, n_modals, layers=1),
            init_vector("uvccsd", a, N_MODES, n_modals, 1),
        ),
        "VIm-uCJ": lambda: (
            build_vim_ucj(ref, N_MODES, n_modals, layers=L, n_pairs=NPAIRS, t2=a.t2),
            imucj_init_vector(a.t2, N_MODES, n_modals, L, NPAIRS, scale=scale),
        ),
        "Vg-uCJ": lambda: (
            build_vg_ucj(ref, N_MODES, n_modals, layers=L, n_pairs=NPAIRS, t2=a.t2),
            gucj_init_vector(a.t2, N_MODES, n_modals, L, NPAIRS, scale=scale),
        ),
        "VLUCJ": lambda: (
            build_ansatz("vlucj", ref, N_MODES, n_modals, layers=L),
            init_vector("vlucj", a, N_MODES, n_modals, L),
        ),
        "CHC": lambda: (
            build_ansatz("chc", ref, N_MODES, n_modals, layers=L),
            init_vector("chc", a, N_MODES, n_modals, L),
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
    Returns (best_energy_cm, iterations, after_distribution, after_occupancies).

    HO basis uses ho_submatrix (direct element-by-element evaluation of the
    basis_size x basis_size submatrix actually needed) instead of slicing a
    pre-built full H_ho -- building the full sparse dim=n_modals**N_MODES
    matrix caused two independent OOM kills for CH2ClF at n_modals=6
    (dim ~1.0e7), confirmed 2026-09-06; H_ho is now unused for the HO basis
    and may be passed as None.

    The all-zero reference/HF-like configuration -- the state every ansatz
    is warm-started from, normally carrying the dominant weight on a clean
    circuit -- is forced into the repaired distribution and into the
    diagonalization basis at EVERY iteration, unconditionally. Without this,
    at zero one-hot retention (raw hardware sample contains no valid
    one-hot bitstrings at all -- confirmed real case: CH2ClF modals=6 HO,
    VLUCJ/CHC, 500k/500k shots, 54 qubits), the loop starts from a flat
    uniform occupancy prior with no real signal to anchor on, and converges
    to an arbitrary seed-dependent WRONG fixed point -- confirmed by
    re-running with 5 different seeds on the same raw data: 5 different
    wrong energies (8724-13245 cm^-1), none near the true ~6708 cm^-1.
    Forcing in the one configuration that's always analytically known in
    advance (regardless of what hardware sampling produced) gives the
    self-consistent loop a real anchor even in a total-noise cell."""
    vmax = n_modals - 1
    occ = initial_uniform_modal_occupancies(N_MODES, n_modals, vmax)
    rng = np.random.default_rng(seed)
    ref_bitstring = _onehot_from_modal_indices([0] * N_MODES, n_modals)
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
        rec[ref_bitstring] = max(rec.get(ref_bitstring, 0.0), 1.0 / basis_size)
        top = sorted(rec, key=rec.get, reverse=True)[:basis_size]
        if ref_bitstring not in top:
            top = top[:-1] + [ref_bitstring]
        bb = sample_basis_from_distribution(rec, basis_size, rng, always_include=top)
        cfgs = [tuple(_modal_indices_from_onehot(b, N_MODES, n_modals)) for b in bb]
        if basis == "modal":
            H = build_subspace_hamiltonian(cfgs, ff, rot, N_HO)
        else:
            H = ho_submatrix(ff, n_modals, cfgs)
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


def _backend_suffix(backend_name):
    """Empty for ibm_kingston (the original/default backend -- every
    existing filename on disk implicitly means "kingston", so this keeps
    that convention unchanged), "_<backend>" for anything else. Without
    this, dry-run/submit output for a second backend (e.g. ibm_phoenix)
    silently overwrites the first backend's same-named file (same fix as
    run_ch2o.py's _backend_suffix -- see that module's docstring for the
    confirmed data-loss incident this prevents)."""
    return "" if backend_name == "ibm_kingston" else f"_{backend_name}"


def _shot_suffix(shots, nq):
    """Empty when `shots` matches the qubit-count-tiered default for this
    cell (the original/default convention -- every existing filename on
    disk implicitly means "standard shot tier"), "_shotsNM"/"_shotsNk"
    otherwise. Without this, a --shots override (e.g. a higher-shot
    empirical test of whether more raw shots closes the CH2ClF
    VCC(8)/VHCI gap) would silently overwrite the standard-tier cell's
    raw_counts/after_full/results files at the same name -- the exact
    data an earlier basis-size-convergence investigation this session
    was built from."""
    default = shots_for_qubit_count(nq)
    if shots == default:
        return ""
    if shots % 1_000_000 == 0:
        return f"_shots{shots // 1_000_000}M"
    if shots % 1_000 == 0:
        return f"_shots{shots // 1_000}k"
    return f"_shots{shots}"


def dry_run_cell(basis, n_modals, measured, backend_name, rep_dir=None, layers=4):
    """Transpile against the REAL target backend (same interaction-aware,
    error-aware path as submit_hardware_batch -- generate_preset_pass_manager
    with no forced initial_layout, see transpile_interaction_aware's
    docstring) and record L3 CZ/depth per ansatz, so --dry-run is an actual
    preview of what --submit will produce rather than a backend-agnostic
    approximation. Requires reachable backend/credentials but spends no QPU
    quota (this is a local transpile against cached target/calibration data,
    not a job submission). If rep_dir is given, also serializes each
    transpiled ISA circuit to <basis>_<ansatz>_isa.qpy in that directory
    (round-trippable via qiskit.qpy.load) -- the filename gets an _L<layers>
    suffix for VIm-uCJ/Vg-uCJ/VLUCJ/CHC when layers != 4, so a layer sweep
    doesn't collide with the default-4 baseline (this parameter is ONLY used
    for filename-suffixing here; the circuits in `measured` were already
    built with the right layer count upstream in build_measured). UVCCSD's
    filename is never suffixed, since it always uses layers=1 regardless.
    A non-kingston backend gets its own _<backend> filename suffix too
    (see _backend_suffix)."""
    if not (
        os.environ.get("QISKIT_IBM_TOKEN") and os.environ.get("QISKIT_IBM_INSTANCE")
    ):
        raise SystemExit(
            "Set QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE before --dry-run "
            "(a real backend.target is needed for an accurate preview)."
        )
    from qiskit_ibm_runtime import QiskitRuntimeService
    from vib_sqd.hardware.transpilation import transpile_interaction_aware
    import qiskit.qpy as qpy

    svc = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["QISKIT_IBM_TOKEN"],
        instance=os.environ["QISKIT_IBM_INSTANCE"],
    )
    backend = svc.backend(backend_name)
    layer_suffix = "" if layers == 4 else f"_L{layers}"
    backend_suffix = _backend_suffix(backend_name)
    rows = []
    for name, m in measured:
        isa = transpile_interaction_aware(m, backend, optimization_level=3)
        rows.append(
            {
                "ansatz": name,
                "basis": basis,
                "n_modals": n_modals,
                "n_qubits": m.num_qubits,
                "backend": backend_name,
                "l3_cz": int(isa.count_ops().get("cz", 0)),
                "l3_depth": int(isa.depth()),
            }
        )
        if rep_dir is not None:
            ansatz_key = name.lower().replace("-", "").replace("_", "")
            suffix = (
                layer_suffix if name in ("VIm-uCJ", "Vg-uCJ", "VLUCJ", "CHC") else ""
            )
            with open(
                rep_dir / f"{basis}_{ansatz_key}{suffix}{backend_suffix}_isa.qpy", "wb"
            ) as fh:
                qpy.dump(isa, fh)
    return rows


def ideal_statevector_counts(circ_with_measure, n_shots, seed):
    """Exact noiseless statevector -> multinomial-sampled counts dict. Feasible
    up to ~24 qubits (2^24 complex128 ~ 256MB); caller must not use above that.

    Uses Aer's AerSimulator(method="statevector") rather than Qiskit's default
    Statevector(circuit) constructor -- the default does naive gate-by-gate
    evolution and is impractically slow for a deep many-qubit circuit; Aer's
    tensor-contraction backend is far faster. reverse_qargs() is still applied
    (this pipeline's universal convention, see run_pipeline.py:_energy) so
    amplitude index i's binary representation has qubit 0 as the
    most-significant bit -- i.e. bitstring position 0 is mode 0's block,
    matching _modal_indices_from_onehot's expectation."""
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


def submit_hardware_batch(
    basis, n_modals, measured, backend_name, shots, rep_dir=None, layers=4, backend=None
):
    """One job covering all requested ansatz circuits for this (basis, modal
    count) cell -- mirrors run_ch2o.py's submit branch. Uses
    transpile_interaction_aware (no forced initial_layout -- see that
    function's docstring for why this beats choose_best_connected_line's
    straight line). If rep_dir is given, also serializes each submitted ISA
    circuit to <basis>_<ansatz>_submitted.qpy in that directory (VIm-uCJ/
    Vg-uCJ/VLUCJ/CHC filenames get an _L<layers> suffix when != 4, same
    convention as dry_run_cell -- filename-suffixing only, doesn't affect
    what's built). `backend`: pass an already-fetched IBMBackend to reuse
    instead of calling svc.backend(backend_name) again -- see run_ch2o.py's
    submit_hardware_batch docstring for why this is required inside an
    active Session/Batch context."""
    if not (
        os.environ.get("QISKIT_IBM_TOKEN") and os.environ.get("QISKIT_IBM_INSTANCE")
    ):
        raise SystemExit(
            "Set QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE before --submit."
        )
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from vib_sqd.hardware.transpilation import transpile_interaction_aware
    from vib_sqd.hardware.suppression import (
        RuntimeSuppressionConfig,
        apply_sampler_suppression_options,
    )
    import qiskit.qpy as qpy

    if backend is None:
        svc = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=os.environ["QISKIT_IBM_TOKEN"],
            instance=os.environ["QISKIT_IBM_INSTANCE"],
        )
        backend = svc.backend(backend_name)
    names = [n for n, _ in measured]
    isa = [
        transpile_interaction_aware(m, backend, optimization_level=3)
        for _, m in measured
    ]
    if rep_dir is not None:
        layer_suffix = "" if layers == 4 else f"_L{layers}"
        backend_suffix = _backend_suffix(backend_name)
        shot_suffix = _shot_suffix(shots, n_modals * N_MODES)
        for name, circ in zip(names, isa):
            ansatz_key = name.lower().replace("-", "").replace("_", "")
            suffix = (
                layer_suffix if name in ("VIm-uCJ", "Vg-uCJ", "VLUCJ", "CHC") else ""
            )
            with open(
                rep_dir
                / f"{basis}_{ansatz_key}{suffix}{backend_suffix}{shot_suffix}_submitted.qpy",
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


def process_hardware_result(
    job, names, basis, ff, rot, H_ho, n_modals, rep_dir, shots, scale=1.0, layers=4
):
    """Retrieve job.result(), decode + recover per ansatz, write per-cell JSON
    artifacts (results/recovered/beforeafter/dryrun already written elsewhere).
    `scale` != 1.0 tags VIm-uCJ/Vg-uCJ's variant key with "@scaleN" (meaningless
    for UVCCSD/VLUCJ/CHC, see run_scale_tuning.py); `layers` != 4 tags
    VIm-uCJ/Vg-uCJ/VLUCJ/CHC's variant key with "@LN" (meaningless for
    UVCCSD, which always uses layers=1). Both tags can apply at once for
    VIm-uCJ/Vg-uCJ (e.g. "VIm-uCJ@scale3.0@L2"); either way the untagged key
    (just `name`) is reserved for the scale=1.0/layers=4 baseline so a sweep
    never overwrites it.

    Also writes the COMPLETE raw pre-recovery bitstring counts (every unique
    measured bitstring, not just the top 50 kept in *_beforeafter.json's
    "before" field) to <basis>_<ansatz>_raw_counts.json, and the complete
    post-recovery distribution (already untruncated) to
    <basis>_<ansatz>_after_full.json -- so both the actual hardware output and
    the actual recovered distribution can be fully visualized later, not just
    a truncated summary."""
    import time as _time

    nq = n_modals * N_MODES
    result = job.result()
    job_backend_name = job.backend().name if hasattr(job, "backend") else None
    backend_suffix = _backend_suffix(job_backend_name) if job_backend_name else ""
    shot_suffix = _shot_suffix(shots, nq)
    out_path = rep_dir / f"{basis}{backend_suffix}{shot_suffix}_results.json"
    # Merge with any existing result file rather than overwrite it: a job
    # that submits only a subset of ansatze (e.g. a scale-tuning run
    # targeting just VIm-uCJ/Vg-uCJ) must not silently drop the other
    # ansatze's already-recorded variants (same fix as excited_state.py's
    # process_excited_hardware_result). The backend_suffix above also keeps
    # a second backend's results from overwriting the first backend's
    # same-cell file (see _backend_suffix).
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out.update(
        {
            "backend": job_backend_name,
            "job_id": job.job_id(),
            "basis": basis,
            "n_modes": N_MODES,
            "n_modals": n_modals,
            "shots": shots,
        }
    )
    out.setdefault("variants", {})
    recovery_and_diag_s_total = out.get("recovery_and_diagonalization_s", 0.0) or 0.0
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
        _t0 = _time.time()
        rec_e, iters, after, after_occ = recovery_loop(
            basis, raw, ff, rot, H_ho, n_modals
        )
        recovery_and_diag_s_total += _time.time() - _t0
        vmax = n_modals - 1
        before_occ = occupancies_from_raw_counts(raw, N_MODES, n_modals, vmax)
        tags = ""
        if name in ("VIm-uCJ", "Vg-uCJ") and scale != 1.0:
            tags += f"@scale{scale}"
        if name in ("VIm-uCJ", "Vg-uCJ", "VLUCJ", "CHC") and layers != 4:
            tags += f"@L{layers}"
        variant_key = f"{name}{tags}" if tags else name
        out["variants"][variant_key] = {
            "onehot_retained_frac": kept / total if total else 0.0,
            "recovered_energy_cm": float(rec_e),
            "recovery_iters": iters,
            "scale": scale,
            "layers": layers,
        }
        ansatz_key = (
            variant_key.lower()
            .replace("-", "")
            .replace("_", "")
            .replace("@", "_")
            .replace(".", "p")
        )
        (
            rep_dir
            / f"{basis}_{ansatz_key}{backend_suffix}{shot_suffix}_raw_counts.json"
        ).write_text(
            json.dumps(
                {
                    "job_id": job.job_id(),
                    "n_qubits": nq,
                    "shots_total": total,
                    "onehot_retained_frac": kept / total if total else 0.0,
                    "raw_counts": {k: int(v) for k, v in raw.items()},
                },
                indent=2,
            )
        )
        (
            rep_dir
            / f"{basis}_{ansatz_key}{backend_suffix}{shot_suffix}_after_full.json"
        ).write_text(
            json.dumps(
                {
                    "job_id": job.job_id(),
                    "converged_cm": float(rec_e),
                    "after_distribution": after,
                    "after_occ": after_occ.tolist(),
                },
                indent=2,
            )
        )
        (
            rep_dir
            / f"{basis}_{ansatz_key}{backend_suffix}{shot_suffix}_beforeafter.json"
        ).write_text(
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
            f"  {variant_key:16s} one-hot retained={kept/total*100:6.3f}%  "
            f"recovered E={rec_e:9.2f} cm^-1",
            flush=True,
        )
    out["recovery_and_diagonalization_s"] = recovery_and_diag_s_total
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {out_path}")
    return out


def uniform_baseline(n_modals, n_shots, seed=SEED):
    """One raw-counts draw per modal count (basis-independent sampling)."""
    from quantum_vs_uniform import uniform_counts

    return uniform_counts(n_shots, N_MODES, n_modals, N_MODES * n_modals, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modals", type=int, choices=MODAL_COUNTS, required=True)
    ap.add_argument("--basis", choices=BASES, required=True)
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--uniform-baseline", action="store_true")
    ap.add_argument(
        "--shots",
        type=int,
        default=None,
        help="overrides the qubit-count-tiered default "
        "(100k up to 24q, 500k 24-56q, 1M 56-72q)",
    )
    ap.add_argument("--basis-size", type=int, default=BASIS_SIZE)
    ap.add_argument("--ansatze", nargs="+", default=list(ANSATZE), choices=ANSATZE)
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="VIm-uCJ/Vg-uCJ arctan(scale*t2) warm-start scale "
        "(see run_scale_tuning.py for the swept grid)",
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=4,
        choices=(1, 2, 4),
        help="Layer count for VIm-uCJ/Vg-uCJ/VLUCJ/CHC together "
        "(default 4, matching the original hardcoded VLUCJ/"
        "CHC value; VIm-uCJ/Vg-uCJ were previously fixed at "
        "LAYERS=2, now also swept). UVCCSD always uses 1 "
        "layer (no comparable stacked-layer structure), "
        "unaffected by this flag. Go small-to-large "
        "(1, then 2, then 4) on real hardware rather than "
        "assuming the deepest circuit is required.",
    )
    a = ap.parse_args()

    import time

    timing = {}

    rep_dir = _rep_dir(a.modals)
    rep_dir.mkdir(parents=True, exist_ok=True)
    nq = a.modals * N_MODES
    shots = a.shots if a.shots is not None else shots_for_qubit_count(nq)

    print(
        f"CH2ClF {N_MODES} modes x {a.modals} modal = {nq} qubits, basis={a.basis}, shots={shots}"
    )
    # --dry-run only prints the reference-config energy (a cheap direct
    # diagonal-element computation, see reference_config_energy) and never
    # touches H_ho's off-diagonal structure -- skip building the full sparse
    # HO Hamiltonian entirely in that case (can take minutes at large
    # n_modals for CH2ClF's 9 modes otherwise).
    t0 = time.time()
    ff, rot, amps, H_ho = build_amps_and_ho(a.modals, need_h_ho=not a.dry_run)
    timing["classical_vcc_and_hamiltonian_s"] = time.time() - t0
    t0 = time.time()
    ref_cm = reference_config_energy(a.basis, ff, rot, H_ho, a.modals)
    timing["reference_config_energy_s"] = time.time() - t0
    print(f"  reference-config energy ({a.basis} basis) = {ref_cm:.2f} cm^-1")

    if a.uniform_baseline:
        raw = uniform_baseline(a.modals, shots)
        rec_e, iters, after, after_occ = recovery_loop(
            a.basis, raw, ff, rot, H_ho, a.modals, basis_size=a.basis_size
        )
        vmax = a.modals - 1
        before_occ = occupancies_from_raw_counts(raw, N_MODES, a.modals, vmax)
        out = {
            "basis": a.basis,
            "n_modes": N_MODES,
            "n_modals": a.modals,
            "shots": shots,
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

    t0 = time.time()
    measured = build_measured(
        a.basis, amps, a.modals, a.ansatze, scale=a.scale, layers=a.layers
    )
    timing["circuit_construction_s"] = time.time() - t0
    # VIm-uCJ/Vg-uCJ/VLUCJ/CHC's filenames vary with --layers (a sweep at
    # layers != 4 must not collide with the default-4 baseline already on
    # disk); UVCCSD is fixed at layers=1 and unaffected by this flag.
    layer_suffix = "" if a.layers == 4 else f"_L{a.layers}"
    backend_suffix = _backend_suffix(a.backend)
    shot_suffix = _shot_suffix(shots, nq)
    LAYERED_ANSATZE = ("VIm-uCJ", "Vg-uCJ", "VLUCJ", "CHC")

    def _write_timing(extra=None):
        payload = dict(timing)
        if extra:
            payload.update(extra)
        (
            rep_dir
            / f"{a.basis}_timing{layer_suffix}{backend_suffix}{shot_suffix}.json"
        ).write_text(json.dumps(payload, indent=2))

    if a.dry_run:
        t0 = time.time()
        rows = dry_run_cell(
            a.basis, a.modals, measured, a.backend, rep_dir=rep_dir, layers=a.layers
        )
        timing["transpile_dryrun_s"] = time.time() - t0
        for r in rows:
            print(
                f"  {r['ansatz']:8s} qubits={r['n_qubits']} L3_CZ={r['l3_cz']} L3_depth={r['l3_depth']}"
            )
        for r in rows:
            ansatz_key = r["ansatz"].lower().replace("-", "").replace("_", "")
            suffix = layer_suffix if r["ansatz"] in LAYERED_ANSATZE else ""
            (
                rep_dir / f"{a.basis}_{ansatz_key}{suffix}{backend_suffix}_dryrun.json"
            ).write_text(json.dumps(r, indent=2))
        _write_timing()
        return

    if a.sim:
        if nq > 24:
            out = {
                "basis": a.basis,
                "n_modes": N_MODES,
                "n_modals": a.modals,
                "note": f"not executable: 2^{nq} statevector infeasible",
                "reference_config_energy_cm": ref_cm,
            }
            (rep_dir / f"{a.basis}_classical_reference.json").write_text(
                json.dumps(out, indent=2)
            )
            print(
                f"  {nq}q: no synthetic circuit executed; classical reference-config "
                f"energy = {ref_cm:.2f} cm^-1 written to {a.basis}_classical_reference.json"
            )
            _write_timing()
            return
        vmax = a.modals - 1
        per_ansatz_recovery_s = {}
        for name, m in measured:
            t0 = time.time()
            raw = ideal_statevector_counts(m, shots, SEED)
            statevector_s = time.time() - t0
            t0 = time.time()
            rec_e, iters, after, after_occ = recovery_loop(
                a.basis, raw, ff, rot, H_ho, a.modals, basis_size=a.basis_size
            )
            per_ansatz_recovery_s[name] = {
                "statevector_sampling_s": statevector_s,
                "recovery_loop_s": time.time() - t0,
                "recovery_iters": iters,
            }
            before_occ = occupancies_from_raw_counts(raw, N_MODES, a.modals, vmax)
            ansatz_key = name.lower().replace("-", "").replace("_", "")
            suffix = layer_suffix if name in LAYERED_ANSATZE else ""
            out = {
                "ansatz": name,
                "basis": a.basis,
                "n_modes": N_MODES,
                "n_modals": a.modals,
                "shots": shots,
                "reference_config_energy_cm": ref_cm,
                "recovered_energy_cm": float(rec_e),
                "recovery_iters": iters,
                "before_occ": before_occ.tolist(),
                "after_occ": after_occ.tolist(),
            }
            (rep_dir / f"{a.basis}_{ansatz_key}{suffix}_results.json").write_text(
                json.dumps(out, indent=2)
            )
            print(f"  {name:8s} sim recovered E={rec_e:9.2f} cm^-1 (ref {ref_cm:.2f})")
        _write_timing({"per_ansatz_s": per_ansatz_recovery_s})
        return

    if a.submit:
        t0 = time.time()
        job = submit_hardware_batch(
            a.basis,
            a.modals,
            measured,
            a.backend,
            shots,
            rep_dir=rep_dir,
            layers=a.layers,
        )
        timing["transpile_and_submit_s"] = time.time() - t0
        names = [n for n, _ in measured]
        t0 = time.time()
        process_hardware_result(
            job,
            names,
            a.basis,
            ff,
            rot,
            H_ho,
            a.modals,
            rep_dir,
            shots,
            scale=a.scale,
            layers=a.layers,
        )
        timing["queue_wait_plus_result_processing_s"] = time.time() - t0
        _write_timing()
        return

    print(
        "no action requested; pass one of --dry-run / --sim / --submit / --uniform-baseline"
    )


if __name__ == "__main__":
    main()
