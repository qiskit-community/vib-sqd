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
Noisy-hardware benchmark of the locked-in ansatze (VIm-uCJ, Vg-uCJ, CHC, VLUCJ) on
glycine-4 modal, with a CALIBRATED fake-IBM-backend noise model, one-hot
post-selection, and SQD projection + recovery -- plus the probability
distributions.

Pipeline per ansatz:
  1. build circuit + few-step warm start (arctan-t2 for Im/Vg-uCJ, VCC-init for
     CHC/VLUCJ), optimize FEWSTEPS steps (SQD operating point).
  2. transpile to a fake IBM backend and sample on AerSimulator.from_backend
     (real calibrated device noise: gate + readout errors, connectivity).
  3. POST-PROCESS the noisy counts:
       * one-hot post-selection: keep only shots that are valid one-hot-per-mode
         bitstrings (the encoding's physical subspace) -> report retained frac;
       * map surviving configs -> SQD parents -> iterative subspace growth
         recovery (qsci_augment.iterative_subspace_growth, bounded cap).
  4. distributions: exact |c|^2, noiseless circuit |c|^2, noisy RAW counts,
     noisy POST-SELECTED counts -- over the compact modal configs.

Figures:
  * noisy_dist_imucj.png  -- exact/noiseless/noisy-raw/noisy-postsel overlay for
    the locked-in VIm-uCJ.
  * noisy_dist_compare.png -- noisy post-selected |c|^2 for VIm-uCJ vs CHC vs
    VLUCJ side by side (robustness under noise).
Console: retained fraction + recovered gap (noiseless vs noisy+postsel+SQD).
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import product
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
import numpy as np

matplotlib.use("Agg")
from qiskit import transpile
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import vib_sqd.basis.modal_transform as mb
import vib_sqd.sqd.qsci_augment as qa
from vib_sqd.circuits.initial_states import build_vscf_reference_state
from vib_sqd.circuits.ucj_correlators import (
    build_vg_ucj,
    build_vim_ucj,
    gucj_init_vector,
    imucj_init_vector,
)
from vib_sqd.classical.mop_forcefield import CM2AU, mop_to_forcefield, parse_mop
from vib_sqd.classical.vcc_extractor import extract_amplitudes
from vib_sqd.classical.vcc_solver import VCCAmplitudes
from vib_sqd.run.dualbasis import (
    _assign,
    build_ansatz,
    ensure_clean_vcc,
    init_vector,
)

FIG = _HERE / "report" / "figures"
OUT = _HERE / "runs" / "noisy_benchmark"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

FEWSTEPS = 2  # SQD operating point (few-step)
SHOTS = 8192
MAXSUB = 25  # bounded SQD diagonalization subspace (discriminating)
SEED = 20260714
LAYERS, NPAIRS = 2, 4


def onehot_index_big(occ, n_modals):
    bits = []
    for v in occ:
        blk = [0] * n_modals
        blk[v] = 1
        bits.extend(blk)
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


def decode_onehot_bitstring(bitstr_bigendian, n_modes, n_modals):
    """Map a big-endian bitstring (mode0 modal0..M-1 | mode1 ...) to an occupation
    tuple IF it is valid one-hot per mode, else None (post-selection)."""
    occ = []
    for m in range(n_modes):
        block = bitstr_bigendian[m * n_modals : (m + 1) * n_modals]
        if block.count("1") != 1:
            return None
        occ.append(block.index("1"))
    return tuple(occ)


def counts_to_config_probs(counts, n_modes, n_modals, occs):
    """Post-select one-hot shots from noisy counts, return (config_prob_vector
    aligned with occs, retained_fraction). Qiskit bitstrings are little-endian
    over qubits -> reverse to big-endian to match our encoding."""
    n_qubits = n_modes * n_modals
    idx_of = {o: i for i, o in enumerate(occs)}
    p = np.zeros(len(occs))
    total = sum(counts.values())
    kept = 0
    for bstr, c in counts.items():
        b = bstr.replace(" ", "")[-n_qubits:].zfill(n_qubits)
        big = b[::-1]  # little-endian -> big-endian
        occ = decode_onehot_bitstring(big, n_modes, n_modals)
        if occ is not None and occ in idx_of:
            p[idx_of[occ]] += c
            kept += c
    frac = kept / total if total else 0.0
    if kept:
        p = p / kept
    return p, frac


def build_variants(ref, amps, n_modes, n_modals):
    out = []
    im = build_vim_ucj(
        ref, n_modes, n_modals, layers=LAYERS, n_pairs=NPAIRS, t2=amps.t2
    )
    out.append(
        ("VIm-uCJ", im, imucj_init_vector(amps.t2, n_modes, n_modals, LAYERS, NPAIRS))
    )
    g = build_vg_ucj(ref, n_modes, n_modals, layers=LAYERS, n_pairs=NPAIRS, t2=amps.t2)
    out.append(
        ("Vg-uCJ", g, gucj_init_vector(amps.t2, n_modes, n_modals, LAYERS, NPAIRS))
    )
    chc = build_ansatz("chc", ref, n_modes, n_modals, layers=4)
    out.append(("CHC", chc, init_vector("chc", amps, n_modes, n_modals, 4)))
    vl = build_ansatz("vlucj", ref, n_modes, n_modals, layers=4)
    out.append(("VLUCJ", vl, init_vector("vlucj", amps, n_modes, n_modals, 4)))
    return out


def main():
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    backend = FakeSherbrooke()
    noisy_sim = AerSimulator.from_backend(backend)
    print(
        f"noise: AerSimulator.from_backend(FakeSherbrooke)  "
        f"({backend.num_qubits} qubits, calibrated Eagle noise)"
    )

    n_modes, n_modals, n_ho = 4, 3, 9
    mout, ffmop = ensure_clean_vcc(
        "glycine4",
        _HERE.parents[1] / "data" / "mop_files" / "glycinehplus.mop",
        (24, 25, 26, 27),
        n_modals,
        str(OUT),
    )
    ff = mop_to_forcefield(parse_mop(ffmop))
    rot = mb.build_modal_rotation(mout, n_modes, n_ho, n_modals)
    t1m, t2m, _ = extract_amplitudes(Path(mout), n_modes, n_modals)
    amps = VCCAmplitudes(t1=t1m, t2=t2m, converged=True)
    Hc = mb.modal_hamiltonian_compact(mb.ho_hamiltonian_full(ff, n_ho), rot)
    w, V = np.linalg.eigh(Hc)
    exact_cm = w[0].real / CM2AU
    c_exact = V[:, 0] / np.linalg.norm(V[:, 0])
    p_exact = np.abs(c_exact) ** 2

    occs = list(product(range(n_modals), repeat=n_modes))
    prod = np.array([onehot_index_big(o, n_modals) for o in occs])
    ref = build_vscf_reference_state(n_modes, n_modals, 0)

    def recovered_gap(config_probs, thresh_count=0.0):
        # parents = sampled configs above threshold; weights = their (noisy)
        # probability, so the subspace cap keeps the MOST PROBABLE configs
        # (SQD ranks by sampled weight, then diagonalizes H exactly in that set).
        parents, weights = [], {}
        for i in range(len(occs)):
            if config_probs[i] > thresh_count:
                ci = qa.config_to_index(occs[i], n_modals)
                parents.append(ci)
                weights[ci] = float(config_probs[i])
        if not parents:
            return float("nan"), 0
        tr = qa.iterative_subspace_growth(
            Hc,
            parents,
            n_modes,
            n_modals,
            max_rounds=5,
            tol_cm=1e-2,
            cm2au=CM2AU,
            max_size=MAXSUB,
            weights=weights,
        )
        return tr[-1][1] / CM2AU - exact_cm, tr[-1][0]

    results = {"exact_cm": exact_cm, "variants": {}}
    dists = {}
    for name, circ, x0 in build_variants(ref, amps, n_modes, n_modals):

        def E(x):
            a = np.asarray(Statevector(_assign(circ, x)).reverse_qargs().data)[prod]
            a = a / np.linalg.norm(a)
            return float(np.real(np.vdot(a, Hc @ a))) / CM2AU

        # few-step optimize
        if FEWSTEPS == 0:
            xs = np.asarray(x0, float)
        else:
            xs = minimize(
                E, x0, method="COBYLA", options={"maxiter": FEWSTEPS, "rhobeg": 0.2}
            ).x
        bound = _assign(circ, xs)
        # noiseless distribution (statevector)
        a = np.asarray(Statevector(bound).reverse_qargs().data)[prod]
        p_noiseless = np.abs(a) ** 2
        p_noiseless /= p_noiseless.sum()
        # noisy sampling on the fake backend
        meas = bound.copy()
        meas.measure_all()
        tqc = transpile(
            meas, backend=noisy_sim, optimization_level=1, seed_transpiler=SEED
        )
        counts = (
            noisy_sim.run(tqc, shots=SHOTS, seed_simulator=SEED).result().get_counts()
        )
        p_noisy_raw = np.zeros(len(occs))
        # raw (no post-selection): still project onto our config labels for plot
        p_ps, frac = counts_to_config_probs(counts, n_modes, n_modals, occs)
        # raw distribution over configs = fraction of shots landing on each valid
        # one-hot label WITHOUT renormalizing away the discarded mass
        total = sum(counts.values())
        praw, _ = counts_to_config_probs(counts, n_modes, n_modals, occs)
        p_noisy_raw = praw * frac  # un-renormalized (keeps the leak visible)

        # recovery
        g_noiseless, nb_nl = recovered_gap(p_noiseless, thresh_count=1e-3)
        g_ps, nb_ps = recovered_gap(p_ps, thresh_count=0.5 / SHOTS)  # >=1 shot

        dists[name] = {
            "noiseless": p_noiseless,
            "noisy_raw": p_noisy_raw,
            "noisy_ps": p_ps,
        }
        results["variants"][name] = {
            "onehot_retained_frac": frac,
            "gap_noiseless_rec": float(g_noiseless),
            "nbasis_noiseless": nb_nl,
            "gap_noisy_ps_rec": float(g_ps),
            "nbasis_noisy_ps": nb_ps,
            "distinct_valid_configs": int((p_ps > 0).sum()),
        }
        print(
            f"  {name:7s} one-hot retained={frac:5.1%}  "
            f"valid configs={int((p_ps>0).sum()):3d}  "
            f"rec_gap noiseless={g_noiseless:7.2f}  noisy+PS+SQD={g_ps:7.2f} cm-1",
            flush=True,
        )

    (OUT / "noisy_benchmark.json").write_text(
        json.dumps(
            {
                k: (v if k != "variants" else {n: vv for n, vv in v.items()})
                for k, v in results.items()
            },
            indent=2,
            default=float,
        )
    )
    from vib_sqd.analysis.plots import plot_noisy_compare, plot_noisy_overlay

    plot_noisy_overlay(
        occs,
        p_exact,
        dists["VIm-uCJ"],
        "VIm-uCJ",
        results["variants"]["VIm-uCJ"]["onehot_retained_frac"],
        FEWSTEPS,
        FIG / "noisy_dist_imucj.png",
    )
    plot_noisy_compare(
        occs,
        p_exact,
        dists,
        results,
        ["VIm-uCJ", "CHC", "VLUCJ"],
        FIG / "noisy_dist_compare.png",
    )
    print(f"\nwrote {OUT}/noisy_benchmark.json and 2 figures")


if __name__ == "__main__":
    main()
