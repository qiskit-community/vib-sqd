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
Dual-basis (HO vs modal) circuit-initialization and VQE-convergence comparison.

The circuit encoding is basis-agnostic (one-hot per level); only three things
change between bases:
  * the Hamiltonian:   HO product H     vs   modal-rotated H = U H U^dagger
  * the reference:     bare-oscillator |0..0>  vs  VSCF ground |0..0> (same
                       circuit, different physical meaning)
  * the amplitudes:    modal t1/t2 are MidasCpp-native; HO t1/t2 rotated from them.

``build_ansatz``/``ensure_clean_vcc`` are the two functions every
``run_<molecule>.py`` driver imports from here: the ansatz-name dispatcher
(UVCCSD/CHC/VLUCJ) and the cached, VCI-free MidasCpp VCC run used for every
molecule's classical bootstrap.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

import vib_sqd.basis.modal_transform as mb
from vib_sqd.circuits.ansatz_enhanced import EnhancedVUCJAnsatz, UVCCSDAnsatz
from vib_sqd.circuits.chc import build_chc_ansatz
from vib_sqd.circuits.initial_states import build_vscf_reference_state
from vib_sqd.classical.mop_forcefield import CM2AU, mop_to_forcefield, parse_mop
from vib_sqd.classical.mop_subset import write_subset_mop
from vib_sqd.classical.vcc_extractor import extract_amplitudes
from vib_sqd.classical.vcc_solver import VCCAmplitudes
from vib_sqd.initialization.vcc_initialization import (
    map_vccsd_to_chc_params,
    map_vccsd_to_uvccsd_initial_point,
    map_vccsd_to_vlucj_params,
)
from vib_sqd.run.pipeline import run_midascpp

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _ordered(c):
    return sorted(
        c.parameters, key=lambda p: int(re.search(r"\[(\d+)\]", p.name).group(1))
    )


def _assign(c, v):
    return c.assign_parameters(dict(zip(_ordered(c), v)))


def two_qubit_gates(c):
    o = c.count_ops()
    return int(o.get("cz", 0) + o.get("cx", 0) + o.get("cp", 0))


class Ev:
    def __init__(self, h_full, idx, gs):
        self.H = h_full
        self.idx = np.asarray(idx, int)
        self.gs = gs / np.linalg.norm(gs)

    def _psi(self, c, x):
        psi = np.asarray(Statevector(_assign(c, x)).reverse_qargs().data)
        p = np.zeros_like(psi)
        p[self.idx] = psi[self.idx]
        p /= np.linalg.norm(p)
        return p

    def energy(self, c, x):
        p = self._psi(c, x)
        return float(np.real(np.vdot(p, self.H @ p))) / CM2AU

    def overlap(self, c, x):
        return float(abs(np.vdot(self.gs, self._psi(c, x)[self.idx])) ** 2)


def run_vqe(ev, name, circ, x0, maxiter, seed=None):
    n = len(_ordered(circ))
    if x0 is None:
        x0 = np.random.default_rng(seed).uniform(-0.05, 0.05, n)
    hist = []

    def obj(x):
        e = ev.energy(circ, x)
        hist.append(e)
        return e

    e0 = ev.energy(circ, x0)
    t0 = time.time()
    res = minimize(
        obj,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.1, "tol": 1e-8},
    )
    return {
        "name": name,
        "n_params": n,
        "two_qubit_gates": two_qubit_gates(circ),
        "depth": int(circ.depth()),
        "initial_cm": e0,
        "final_cm": float(res.fun),
        "overlap_final": ev.overlap(circ, res.x),
        "n_evals": len(hist),
        "history_cm": hist,
    }


def build_ansatz(name, ref, n_modes, n_modals, layers):
    dummy = SparsePauliOp(["I" * (n_modes * n_modals)], np.array([0.0]))
    name = name.lower()
    if name == "uvccsd":
        return UVCCSDAnsatz("direct").build(
            dummy,
            ref,
            {
                "n_modes": n_modes,
                "n_modals": n_modals,
                "parameterized": True,
                "measure": False,
            },
        )[0]
    if name == "chc":
        return build_chc_ansatz(ref, n_modes, n_modals, layers=layers)
    if name == "vlucj":
        return EnhancedVUCJAnsatz("direct").build(
            dummy,
            ref,
            {
                "n_modes": n_modes,
                "n_modals": n_modals,
                "layers": layers,
                "measure": False,
                "include_rz": True,
                "adjacent_jastrow": True,
            },
        )[0]
    raise ValueError(name)


def init_vector(name, amps, n_modes, n_modals, layers):
    name = name.lower()
    if name == "uvccsd":
        return map_vccsd_to_uvccsd_initial_point(
            amps, n_modes=n_modes, n_modals=n_modals, scaling_strategy="direct"
        )
    if name == "chc":
        return map_vccsd_to_chc_params(
            amps,
            n_modes=n_modes,
            n_modals=n_modals,
            layers=layers,
            scaling_strategy="direct",
            clip=np.pi,
            distribute_over_layers=True,
        )
    if name == "vlucj":
        return map_vccsd_to_vlucj_params(
            amps,
            n_modes=n_modes,
            n_modals=n_modals,
            layers=layers,
            scaling_strategy="direct",
            clip=np.pi,
            distribute_over_layers=True,
        )
    raise ValueError(name)


def ensure_clean_vcc(label, mop_path, selected_modes, n_modals, outroot):
    """Run a clean VCC-only (no VCI) MidasCpp job in the HO basis and return the
    .mout path. Provides both the VSCF modal matrices and uncontaminated
    ground-state VCC amplitudes (the CI runs mix in VCI vectors, which confuse
    the amplitude extractor)."""
    run_dir = Path(outroot) / f"vcc_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = parse_mop(mop_path)
    # Resolve the (possibly subset) source mop, then the copy MidasCpp reads.
    if selected_modes is not None:
        src = run_dir / f"{label}.mop"
        write_subset_mop(doc, selected_modes, src)
    else:
        src = Path(mop_path)
    ff_mop = run_dir / "midas" / src.name  # run_midascpp copies here
    mout = run_dir / "midas" / "input.mout"
    if mout.exists() and "PROD-ALLHO" in mout.read_text():
        return mout, ff_mop
    # Pass n_modes so run_midascpp can auto-avoid the OccAllFund crash at >=7
    # modes (upstream MidasCpp bug; see MIDASCPP_BUG_OccAllFund.md). For a mode
    # subset this is len(selected_modes); otherwise the full mode count.
    n_modes = (
        len(selected_modes)
        if selected_modes is not None
        else len(getattr(doc, "mode_names", []) or [])
    )
    m = run_midascpp(
        src,
        run_dir / "midas",
        n_modals=n_modals,
        use_scaling_freqs=doc.scale_factors is not None,
        vcc_method="VCC[2]",
        n_modes=n_modes,
    )
    return m, ff_mop


def run_system(
    label,
    mop_path,
    mout_path,
    n_modes,
    n_modals,
    n_ho,
    layers,
    maxiter,
    ansatze,
    outdir,
):
    ff = mop_to_forcefield(parse_mop(mop_path))
    H_ho_full = mb.ho_hamiltonian_full(ff, n_ho)
    rot = mb.build_modal_rotation(mout_path, n_modes, n_ho, n_modals)

    # Native MODAL amplitudes from MidasCpp; HO amplitudes by rotation.
    t1_mod, t2_mod, meta = extract_amplitudes(Path(mout_path), n_modes, n_modals)
    t1_ho, t2_ho = mb.ho_amplitudes_from_modal(t1_mod, t2_mod, rot)
    amps = {
        "modal": VCCAmplitudes(t1=t1_mod, t2=t2_mod, converged=True),
        "ho": VCCAmplitudes(t1=t1_ho, t2=t2_ho, converged=True),
    }

    # Basis Hamiltonians truncated to n_modals, one-hot embedded.
    #   modal: U H U^dagger ; HO: top-left n_modals block product.
    H_mod_c = mb.modal_hamiltonian_compact(H_ho_full, rot)
    H_ho_c = mb.ho_hamiltonian_full(ff, n_modals)  # HO truncated to n_modals
    results = {
        "label": label,
        "n_modes": n_modes,
        "n_modals": n_modals,
        "n_qubits": n_modes * n_modals,
        "bases": {},
    }

    for basis, Hc in [("modal", H_mod_c), ("ho", H_ho_c)]:
        Hfull, phys = mb.embed_onehot(Hc, n_modes, n_modals)
        w, V = np.linalg.eigh(Hc)
        exact = float(w[0].real) / CM2AU
        # exact ground embedded onto physical indices, reversed-qarg convention:
        gs_full = np.zeros(2 ** (n_modes * n_modals), dtype=complex)
        gs_full[phys] = V[:, 0]
        ev = Ev(Hfull, phys, gs_full[phys])
        ref = build_vscf_reference_state(n_modes, n_modals, 0)
        bres = {"exact_ground_cm": exact, "runs": []}
        print(
            f"\n[{label}/{basis}] exact ground = {exact:.2f} cm-1 "
            f"(ref |0..0> = {Hc[0,0].real/CM2AU:.2f})"
        )
        for aname in ansatze:
            circ = build_ansatz(aname, ref, n_modes, n_modals, layers)
            n = len(_ordered(circ))
            x_vcc = init_vector(aname, amps[basis], n_modes, n_modals, layers)
            if len(x_vcc) != n:
                print(f"  ! {aname}: init len {len(x_vcc)} != params {n}; skip")
                continue
            r0 = run_vqe(ev, f"{aname}_zero", circ, np.zeros(n), maxiter)
            rv = run_vqe(ev, f"{aname}_vcc", circ, x_vcc, maxiter)
            for r in (r0, rv):
                r["ansatz"] = aname
                bres["runs"].append(r)
                print(
                    f"  {r['name']:14s} init={r['initial_cm']:8.2f} "
                    f"final={r['final_cm']:8.2f} (Δ{r['final_cm']-exact:+7.2f}) "
                    f"ov={r['overlap_final']:.3f}"
                )
        results["bases"][basis] = bres

    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f"{label}_dualbasis.json").write_text(
        json.dumps(
            results,
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o,
        )
    )
    return results


def modal_count_sweep(label, mop_path, mout_path, n_modes, n_ho, outdir):
    """Exact ground error vs modals M, HO vs modal basis (compact diag)."""
    ff = mop_to_forcefield(parse_mop(mop_path))
    H_ho_full = mb.ho_hamiltonian_full(ff, n_ho)
    Efull = mb.exact_ground_cm(H_ho_full)
    rows = []
    for M in range(1, min(n_ho, 6) + 1):
        rot = mb.build_modal_rotation(mout_path, n_modes, n_ho, M)
        Emod = mb.exact_ground_cm(mb.modal_hamiltonian_compact(H_ho_full, rot))
        Eho = mb.exact_ground_cm(mb.ho_hamiltonian_full(ff, M))
        rows.append(
            {
                "M": M,
                "qubits": n_modes * M,
                "ho_err": Eho - Efull,
                "modal_err": Emod - Efull,
            }
        )
        print(
            f"  M={M} qubits={n_modes*M}: HO err {Eho-Efull:+.2f}, "
            f"modal err {Emod-Efull:+.2f} cm-1"
        )
    out = {"label": label, "n_modes": n_modes, "exact_full_cm": Efull, "sweep": rows}
    (Path(outdir) / f"{label}_modalsweep.json").write_text(json.dumps(out, indent=2))
    return out
