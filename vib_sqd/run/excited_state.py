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

"""Molecule-agnostic excited-state helpers shared by the run_*.py sweep
scripts: run a MidasCpp ``#3 Occup`` VCC calculation for a single excited
mode, extract that state's own t1/t2 amplitudes, and run the self-consistent
SQD recovery loop against it.

The excited-state VSCF reference is always built with
``build_vscf_reference_state(n_modes, n_modals, 0)`` -- the SAME circuit call
as the ground state -- because every mode's reference occupation is encoded
at logical qubit position 0 by convention (see
``vib_sqd.basis.modal_occupation``). Only the classical bookkeeping
here (amplitude extraction, decoding recovered configurations for
Hamiltonian lookups) needs to know which absolute modal each mode's logical
position 0 actually represents.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import vib_sqd.basis.modal_transform as mb
from vib_sqd.basis.modal_occupation import logical_to_absolute_occupation
from vib_sqd.classical.mop_forcefield import CM2AU
from vib_sqd.classical.vcc_extractor import extract_amplitudes_for_occupation
from vib_sqd.classical.vcc_solver import VCCAmplitudes
from vib_sqd.run.pipeline import run_midascpp
from vib_sqd.sqd.subspace_hamiltonian import build_subspace_hamiltonian
from vib_sqd.sqd.vibrational_recovery import (
    _modal_indices_from_onehot,
    _onehot_from_modal_indices,
    initial_uniform_modal_occupancies,
    occupancies_from_ground_state,
    recover_direct_onehot_distribution,
    sample_basis_from_distribution,
)


def _compact_index(cfg, n_modals):
    """Row-major index into the implicit product(range(n_modals),
    repeat=n_modes) ordering (matches every *_forcefield.py's identical
    helper of the same name)."""
    idx = 0
    for c in cfg:
        idx = idx * n_modals + c
    return idx


def ensure_clean_vcc_excited(
    label, mop_path, n_modals, outroot, occupation, n_modes, use_scaling_freqs
):
    """Occup-based VCC-only MidasCpp run for an excited-state reference
    (mirrors run_dualbasis_experiment.ensure_clean_vcc, but for an explicit
    per-mode occupation instead of the all-zero ground state)."""
    run_dir = Path(outroot) / f"vcc_excited_{label}" / "midas"
    mout = run_dir / "input.mout"
    if mout.exists() and "PROD-ALLHO" in mout.read_text():
        return mout
    return run_midascpp(
        mop_path,
        run_dir,
        n_modals=n_modals,
        use_scaling_freqs=use_scaling_freqs,
        vcc_method="VCC[2]",
        occup=list(occupation),
        n_modes=n_modes,
    )


def build_excited_amps(
    mop_path,
    n_modes,
    n_modals,
    mode_idx,
    rot,
    outroot,
    label,
    use_scaling_freqs=True,
    target_modal=1,
):
    """Run a #3 Occup excited-state VCC calc (mode `mode_idx` excited to
    absolute modal `target_modal`, default 1 -- the lowest nontrivial
    excitation) and extract that state's own t1/t2 amplitudes in both modal
    and HO bases.

    Returns (occupation, amps, mout_path) where `occupation` is the absolute
    per-mode modal tuple and `amps` is a {"modal": .., "ho": ..}
    VCCAmplitudes dict, same shape as build_amps_and_ho's ground-state amps.
    """
    occupation = [0] * n_modes
    occupation[mode_idx] = target_modal
    mout = ensure_clean_vcc_excited(
        label, mop_path, n_modals, outroot, occupation, n_modes, use_scaling_freqs
    )
    t1_mod, t2_mod, meta = extract_amplitudes_for_occupation(
        mout, n_modes, n_modals, occupation
    )
    t1_ho, t2_ho = mb.ho_amplitudes_from_modal(t1_mod, t2_mod, rot)
    amps = {
        "modal": VCCAmplitudes(t1=t1_mod, t2=t2_mod, converged=True, metadata=meta),
        "ho": VCCAmplitudes(t1=t1_ho, t2=t2_ho, converged=True, metadata=meta),
    }
    return tuple(occupation), amps, mout


def excited_reference_energy(basis, occupation, ff, rot, H_ho, n_modals, N_HO):
    """Absolute energy of the excited reference configuration (analogous to
    each run_*.py's reference_config_energy, generalized to an arbitrary
    occupation instead of the all-zero ground state)."""
    if basis == "modal":
        H = build_subspace_hamiltonian([tuple(occupation)], ff, rot, N_HO)
        return float(H[0, 0].real / CM2AU)
    idx0 = _compact_index(tuple(occupation), n_modals)
    return float(H_ho[idx0, idx0].real / CM2AU)


def excited_recovery_loop(
    basis,
    raw_counts,
    occupation,
    ff,
    rot,
    H_ho,
    n_modes,
    n_modals,
    N_HO,
    basis_size=200,
    seed=20260722,
    max_iter=30,
    conv_cm=0.1,
    ho_submatrix_fn=None,
):
    """Self-consistent SQD recovery for an excited-state reference -- same
    algorithm as each run_*.py's recovery_loop, generalized so that (a) the
    excited reference configuration is guaranteed to be in the sampled
    subspace even if it was not actually measured (the manuscript's
    "guaranteed inclusion of the main configuration" methodology note,
    applied here to the excited-state reference rather than the ground
    state), (b) decoded logical modal indices are converted to absolute
    modal indices (via `occupation`) before every Hamiltonian lookup, and
    (c) the reported energy/eigenvector at each iteration is the one with
    MAXIMUM OVERLAP onto the guaranteed-included excited reference
    configuration, not simply the lowest eigenvalue of the sampled
    subspace.

    (c) is not optional: taking the lowest eigenvalue (the right choice
    for the ground state, by the variational principle) silently collapses
    to the GLOBAL ground state's energy for an excited-state target as soon
    as the sampled subspace contains any lower-energy configuration --
    which repair starting from a uniform occupancy prior makes near-certain
    once real (or even synthetic 1%) sampling noise is present. Verified:
    with 99% of raw counts on the exact correct excited-reference
    bitstring and 1% uniform-random noise, taking w[0] recovers the GROUND
    reference energy (wrong by >1000 cm^-1 for CH2O); taking the
    max-overlap eigenvector recovers the excited reference energy exactly.

    Returns (energy_cm, iterations, recovered_distribution,
    after_occupancies) -- same shape as recovery_loop's return value.

    `ho_submatrix_fn` (HO basis only): a molecule module's own
    `ho_submatrix(ff, n_modals, cfgs)` -- direct element-by-element
    evaluation of just the needed submatrix, the same fix applied to
    every run_*.py's ground-state recovery_loop this session (building
    the full sparse H_ho and slicing it, the previous behavior here,
    caused two independent OOM crashes for CH2ClF's 9-mode ground-state
    HO-basis loop at dim~1e7; the excited-state loop was never run in
    HO basis before, so this same crash was latent, not yet triggered).
    If not given, falls back to the old H_ho-slicing behavior (only
    safe for small systems, e.g. H2O's 3 modes).
    """
    vmax = n_modals - 1
    occ = initial_uniform_modal_occupancies(n_modes, n_modals, vmax)
    rng = np.random.default_rng(seed)
    # By construction (see vib_sqd.basis.modal_occupation), the
    # excited reference's LOGICAL one-hot is always all-zero: logical
    # position 0 means "this mode's own reference occupation" for every
    # mode, regardless of which absolute modal that is.
    main_config = _onehot_from_modal_indices([0] * n_modes, n_modals)

    e = None
    prev = None
    iters = 0
    rec = None
    for _ in range(max_iter):
        iters += 1
        rec = recover_direct_onehot_distribution(
            raw_counts,
            avg_modal_occupancies=occ,
            n_modes=n_modes,
            n_modals=n_modals,
            vmax=vmax,
            seed=int(rng.integers(0, 2**31 - 1)),
            randomize=True,
        )
        if main_config not in rec:
            rec = dict(rec)
            rec[main_config] = max(rec.values())
        top = sorted(rec, key=rec.get, reverse=True)[:basis_size]
        if main_config not in top:
            top = top[:-1] + [main_config]
        bb = sample_basis_from_distribution(rec, basis_size, rng, always_include=top)
        if main_config not in bb:
            bb = bb + [main_config]
        main_idx = bb.index(main_config)
        logical_cfgs = [
            tuple(_modal_indices_from_onehot(b, n_modes, n_modals)) for b in bb
        ]
        cfgs = [
            logical_to_absolute_occupation(lc, occupation, n_modals)
            for lc in logical_cfgs
        ]
        if basis == "modal":
            H = build_subspace_hamiltonian(cfgs, ff, rot, N_HO)
        elif ho_submatrix_fn is not None:
            H = ho_submatrix_fn(ff, n_modals, cfgs)
        else:
            idxs = [_compact_index(c, n_modals) for c in cfgs]
            H = H_ho[np.ix_(idxs, idxs)].toarray()
        w, V = np.linalg.eigh(H)
        overlaps = np.abs(V[main_idx, :]) ** 2
        best_idx = int(np.argmax(overlaps))
        e = w[best_idx].real / CM2AU
        occ = occupancies_from_ground_state(bb, V[:, best_idx], n_modes, n_modals, vmax)
        if prev is not None and abs(e - prev) < conv_cm:
            break
        prev = e
    return e, iters, rec, occ


def _bits_big(k, nq):
    return k.replace(" ", "")[-nq:].zfill(nq)[::-1]


def _onehot_decode(bb, n_modes, n_modals):
    occ = []
    for m in range(n_modes):
        blk = bb[m * n_modals : (m + 1) * n_modals]
        if blk.count("1") != 1:
            return None
        occ.append(blk.index("1"))
    return tuple(occ)


def process_excited_hardware_result(
    job,
    names,
    basis,
    occupation,
    ff,
    rot,
    H_ho,
    n_modes,
    n_modals,
    N_HO,
    rep_dir,
    shots,
    scale=1.0,
    ho_submatrix_fn=None,
):
    """Excited-state analogue of each run_*.py's process_hardware_result:
    retrieve job.result(), decode + recover per ansatz via
    excited_recovery_loop (occupation-aware) rather than the ground-state
    recovery_loop, write per-cell JSON artifacts under
    rep_dir/excited_<basis>_*.json. `scale` != 1.0 writes VIm-uCJ/Vg-uCJ under
    a scale-suffixed variant key (e.g. "VIm-uCJ@scale3.0") rather than
    overwriting the scale=1.0 baseline key -- see run_scale_tuning_excited.py.
    `ho_submatrix_fn`: passed straight through to excited_recovery_loop
    (HO basis only) -- see that function's docstring. A non-kingston
    backend (from job.backend().name) gets its own _<backend> filename
    suffix on every output file (same fix, and same reasoning, as
    run_ch2o.py's/run_ch2fcl.py's _backend_suffix -- without it, a second
    backend's excited-state result silently overwrites the first's)."""
    from vib_sqd.sqd.vibrational_recovery import occupancies_from_raw_counts

    nq = n_modes * n_modals
    result = job.result()
    job_backend_name = job.backend().name if hasattr(job, "backend") else None
    backend_suffix = (
        ""
        if not job_backend_name or job_backend_name == "ibm_kingston"
        else f"_{job_backend_name}"
    )
    # Filename includes n_modals: excited_modal_results.json (no suffix) was
    # the original, ambiguous scheme -- every modal count at a given basis
    # clobbered the same file, silently losing e.g. CH2O's 36q UVCCSD/VLUCJ
    # results the moment the 48q cell was submitted. A one-time migration
    # below moves that legacy file to its correct modals-suffixed name (using
    # the n_modals value already recorded inside it) before this cell writes
    # its own. Legacy files are always kingston-era (predate the backend
    # suffix), so this migration is intentionally unsuffixed.
    legacy_path = rep_dir / f"excited_{basis}_results.json"
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text())
        legacy_modals = legacy.get("n_modals")
        if legacy_modals is not None:
            migrated_path = (
                rep_dir / f"excited_{basis}_results_modals{legacy_modals}.json"
            )
            if not migrated_path.exists():
                migrated_path.write_text(json.dumps(legacy, indent=2, default=float))
                print(f"  migrated legacy {legacy_path.name} -> {migrated_path.name}")
            legacy_path.unlink()

    out_path = (
        rep_dir / f"excited_{basis}_results_modals{n_modals}{backend_suffix}.json"
    )
    # Merge with any existing result file at this modal count rather than
    # overwrite it: a job that submits only a subset of ansatze (e.g. adding
    # VIm-uCJ/Vg-uCJ/CHC to an existing UVCCSD/VLUCJ result at the same cell)
    # must not silently drop the other ansatze's already-recorded variants.
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out.update(
        {
            "backend": job_backend_name,
            "job_id": job.job_id(),
            "basis": basis,
            "n_modes": n_modes,
            "n_modals": n_modals,
            "occupation": list(occupation),
            "shots": shots,
        }
    )
    out.setdefault("variants", {})
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
            bb = _bits_big(k, nq)
            raw[bb] = raw.get(bb, 0.0) + float(c)
            if _onehot_decode(bb, n_modes, n_modals) is not None:
                kept += c
        before = dict(sorted(raw.items(), key=lambda kv: -kv[1])[:50])
        before_total = sum(before.values())
        before = (
            {k: v / before_total for k, v in before.items()} if before_total else {}
        )
        rec_e, iters, after, after_occ = excited_recovery_loop(
            basis,
            raw,
            occupation,
            ff,
            rot,
            H_ho,
            n_modes,
            n_modals,
            N_HO,
            ho_submatrix_fn=ho_submatrix_fn,
        )
        vmax = n_modals - 1
        before_occ = occupancies_from_raw_counts(raw, n_modes, n_modals, vmax)
        variant_key = (
            f"{name}@scale{scale}"
            if (name in ("VIm-uCJ", "Vg-uCJ") and scale != 1.0)
            else name
        )
        out["variants"][variant_key] = {
            "onehot_retained_frac": kept / total if total else 0.0,
            "recovered_energy_cm": float(rec_e),
            "recovery_iters": iters,
            "scale": scale,
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
            / f"excited_{basis}_{ansatz_key}{backend_suffix}_modals{n_modals}_beforeafter.json"
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
            f"excited-state recovered E={rec_e:9.2f} cm^-1",
            flush=True,
        )
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {out_path}")
    return out
