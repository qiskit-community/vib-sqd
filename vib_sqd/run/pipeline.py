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

"""MidasCpp subprocess wrapper and small ansatz/energy helpers shared by
every ``run_<molecule>.py`` driver script.

``run_midascpp`` writes the corrected HO-basis ``.inp`` (see
``classical.inp_template``), runs ``midascpp.x``, and returns the
``.mout`` path. It auto-scales ``ItEqMaxIt`` with mode count and defaults
to ground-state-only VSCF at >= 7 modes to avoid the upstream
``OccAllFund`` crash (see ``MIDASCPP_BUG_OccAllFund.md`` in
``workflows/midascpp_discovery/pipeline/``). Do not "simplify" that
workaround away -- it is load-bearing at large mode counts.

``build_ansatz``/``run_vqe``/``_energy`` are the pure-statevector VQE
helpers used by the single-molecule end-to-end driver
(``run_pipeline.py``) and by ``glycine_sweep.py``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from vib_sqd.classical.inp_template import build_midas_input

_HERE = Path(__file__).resolve().parent
# Repo layout: src/vib_sqd/run/pipeline.py -> repo root is 3 up.
_REPO_ROOT = _HERE.parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# The two placeholder paths below are never expected to exist -- they exist
# only so a MIDASCPP_BIN-unset run fails with a clear, actionable error
# (below) rather than an obscure one. There is no shared/vendored MidasCpp
# install; every user builds and points at their own. See
# examples/*/README.md ("0. Prerequisites") and docs/midascpp_installation.md
# for how to set MIDASCPP_BIN / MIDAS_INSTALL_PREFIX / MIDAS_INSTALL_DATADIR
# for your own build.
_MIDASCPP_BIN_PLACEHOLDER = "/path/to/your/midascpp/release/bin/midascpp.x"
_MIDASCPP_EXTLIB_PLACEHOLDERS = [
    "/path/to/your/midascpp/extlibs/cutee-build/midascpp_install/lib",
    "/path/to/your/midascpp/extlibs/stackd-build/midascpp_install/lib",
]


# --------------------------------------------------------------------------- #
# MidasCpp run
# --------------------------------------------------------------------------- #
def _midas_bin() -> str:
    from_env = os.environ.get("MIDASCPP_BIN")
    bin_path = from_env or _MIDASCPP_BIN_PLACEHOLDER
    if not Path(bin_path).exists():
        raise FileNotFoundError(
            f"midascpp.x not found at {bin_path!r}. "
            "Set MIDASCPP_BIN to your own build's binary (and "
            "MIDAS_INSTALL_PREFIX / MIDAS_INSTALL_DATADIR to your install's "
            f"prefix/data dirs). See {_EXAMPLES_DIR}/ground_state_tutorial/"
            "README.md ('0. Prerequisites') and docs/midascpp_installation.md "
            "for the full walkthrough."
        )
    return bin_path


def _midas_extlib_paths() -> str:
    """Derive the cutee/stackd/gsl (etc.) extlib dirs from the midascpp.x
    location, with a hardcoded fallback. MidasCpp needs these on
    DYLD_LIBRARY_PATH (macOS)/LD_LIBRARY_PATH (Linux) to load
    libcutee/libstackd/libgsl; a bare shell without them fails with
    'Library not loaded: @rpath/libcutee.1.dylib' (macOS) or 'error while
    loading shared libraries: libgsl.so.0: cannot open shared object file'
    (Linux, when the build had to self-build GSL from source because no
    system GSL was found -- that extlib directory is named e.g. 'gsl-1.16',
    without the '-build' suffix cutee/stackd's self-built dirs have)."""
    paths: list[str] = []
    try:
        # midascpp.x lives at <root>/release/bin/midascpp.x; extlibs at <root>/extlibs.
        root = Path(_midas_bin()).resolve().parents[2]
        extlibs = root / "extlibs"
        if extlibs.is_dir():
            for pattern in ("*-build/midascpp_install/lib", "*/midascpp_install/lib"):
                for sub in sorted(extlibs.glob(pattern)):
                    if str(sub) not in paths:
                        paths.append(str(sub))
    except Exception:  # noqa: BLE001
        pass
    if not paths:
        # Last-resort fallback -- placeholders, not a real shared install
        # (see this module's top-of-file comment).
        paths = list(_MIDASCPP_EXTLIB_PLACEHOLDERS)
    paths.append("/opt/homebrew/lib")
    return ":".join(paths)


def run_midascpp(
    mop_path: Path,
    run_dir: Path,
    *,
    n_modals: int,
    use_scaling_freqs: bool,
    vcc_method: str,
    it_eq_max_it: int | None = None,
    occup: list[int] | None = None,
    ground_state_only: bool | None = None,
    n_modes: int | None = None,
) -> Path:
    """Write the .inp, run MidasCpp in run_dir, return the .mout path.

    ``it_eq_max_it`` defaults to ``max(100, 8*n_modes)`` (VCC needs more
    iterations as the number of coupled modes grows); pass an int to override,
    or rely on the auto value when ``n_modes`` is given. ``occup`` selects an
    explicit occupation (per-fundamental path)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    local_mop = run_dir / mop_path.name
    shutil.copy(mop_path, local_mop)

    # Auto-scale VCC iterations with mode count unless the caller overrides.
    if it_eq_max_it is None and n_modes is not None:
        it_eq_max_it = max(100, 8 * int(n_modes))

    # OccAllFund crashes MidasCpp at >= ~7 modes (upstream bug; see
    # MIDASCPP_BUG_OccAllFund.md). Default to ground-state-only there so the
    # main pipeline does not hit it; the caller can force either way. Only
    # print the ">= 7 modes" rationale when THAT is actually why
    # ground_state_only ended up True -- a caller-forced override (e.g. for
    # a molecule with its own separate VSCF-convergence issue at < 7 modes)
    # has nothing to do with the OccAllFund mode-count threshold, and
    # printing this message for it would misreport the reason.
    auto_triggered = (
        ground_state_only is None
        and occup is None
        and n_modes is not None
        and n_modes >= 7
    )
    if ground_state_only is None:
        ground_state_only = auto_triggered
    if auto_triggered:
        print(
            f"  (n_modes={n_modes} >= 7: OccGroundState only, "
            "avoiding the OccAllFund crash)"
        )

    inp_text = build_midas_input(
        mop_filename=mop_path.name,
        n_modals=n_modals,
        use_scaling_freqs=use_scaling_freqs,
        vcc_method=vcc_method,
        it_eq_max_it=it_eq_max_it,
        occup=occup,
        ground_state_only=ground_state_only,
        n_modes=n_modes,
    )
    inp_path = run_dir / "input.inp"
    inp_path.write_text(inp_text)

    env = dict(os.environ)
    extra = _midas_extlib_paths()
    for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        existing = env.get(var, "")
        env[var] = f"{extra}:{existing}" if existing else extra
    # Soft defaults, not hard requirements -- MidasCpp itself only needs
    # these set if ITS OWN install actually uses them; unlike
    # MIDASCPP_BIN (checked for existence above), there is no local check
    # here because a wrong value only matters if/when MidasCpp reads it.
    # Same placeholder caveat as _MIDASCPP_BIN_PLACEHOLDER above: set
    # MIDAS_INSTALL_PREFIX / MIDAS_INSTALL_DATADIR yourself for your own
    # install (see examples/ground_state_tutorial/README.md).
    env.setdefault("MIDAS_INSTALL_PREFIX", "/path/to/your/midascpp")
    env.setdefault("MIDAS_INSTALL_DATADIR", "/path/to/your/midascpp/data")

    print(f"  Running MidasCpp ({vcc_method}) in {run_dir} ...")
    with open(run_dir / "run.stdout", "w") as out:
        proc = subprocess.run(
            [_midas_bin(), "input.inp"],
            cwd=run_dir,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    mout = run_dir / "input.mout"
    if proc.returncode != 0 or not mout.exists():
        raise RuntimeError(
            f"MidasCpp failed (exit {proc.returncode}); see {run_dir/'run.stdout'}"
        )

    text = mout.read_text()
    if "PROD-ALLHO" not in text:
        raise RuntimeError(
            "MidasCpp did not use the HO basis (no 'PROD-ALLHO' in .mout). "
            "The #3 BasisType HO fix is not taking effect."
        )
    print("  ✓ MidasCpp ran; HO basis confirmed (PROD-ALLHO).")
    return mout


# --------------------------------------------------------------------------- #
# Circuit / VQE
# --------------------------------------------------------------------------- #
def _ordered_params(circuit):
    """Parameters sorted by their ParameterVector index (theta[3] etc.)."""

    def key(p):
        m = re.search(r"\[(\d+)\]", p.name)
        if m is None:
            raise ValueError(f"cannot order parameter {p.name}")
        return int(m.group(1))

    return sorted(circuit.parameters, key=key)


def _assign(circuit, values):
    params = _ordered_params(circuit)
    if len(values) != len(params):
        raise ValueError(f"param count mismatch: {len(values)} values vs {len(params)}")
    return circuit.assign_parameters(dict(zip(params, values)))


def _energy(circuit, h_full, physical_indices):
    """Energy expectation on the FULL-space Hamiltonian.

    Qubit-ordering note: ``vib_sqd.encoding.direct`` builds the dense
    Hamiltonian with qubit 0 as the most-significant bit (big-endian), while
    Qiskit's Statevector is little-endian. ``reverse_qargs()`` reconciles the
    two so the circuit's amplitudes line up with H's basis ordering. We then
    renormalize onto the physical (one-hot) subspace and take <psi|H|psi>.
    """
    from qiskit.quantum_info import Statevector

    psi_full = np.asarray(Statevector(circuit).reverse_qargs().data)
    # Restrict to the physical subspace (zero any leakage), renormalize.
    psi = np.zeros_like(psi_full)
    psi[physical_indices] = psi_full[physical_indices]
    norm = float(np.vdot(psi, psi).real)
    if norm < 1e-12:
        raise ValueError("state has ~zero norm in physical subspace")
    psi = psi / np.sqrt(norm)
    return float(np.real(np.vdot(psi, h_full @ psi)))


def build_ansatz(name, reference_state, n_modes, n_modals, layers):
    """Return a parameterized ansatz circuit for the given name."""
    from qiskit.quantum_info import SparsePauliOp

    from vib_sqd.circuits.ansatz_enhanced import (
        EnhancedVUCJAnsatz,
        UVCCSDAnsatz,
    )
    from vib_sqd.circuits.chc import build_chc_ansatz

    dummy = SparsePauliOp(["I" * (n_modes * n_modals)], np.array([0.0]))
    name = name.lower()
    if name == "uvccsd":
        circ = UVCCSDAnsatz(encoding="direct").build(
            hamiltonian=dummy,
            initial_state=reference_state,
            cfg={
                "n_modes": n_modes,
                "n_modals": n_modals,
                "parameterized": True,
                "measure": False,
            },
        )[0]
    elif name == "chc":
        circ = build_chc_ansatz(reference_state, n_modes, n_modals, layers=layers)
    elif name == "vlucj":
        circ = EnhancedVUCJAnsatz(encoding="direct").build(
            hamiltonian=dummy,
            initial_state=reference_state,
            cfg={
                "n_modes": n_modes,
                "n_modals": n_modals,
                "layers": layers,
                "measure": False,
                "include_rz": True,
                "adjacent_jastrow": True,
            },
        )[0]
    else:
        raise ValueError(f"unknown ansatz {name}")
    return circ


def run_vqe(name, circuit, x0, h_matrix, physical_indices, maxiter, seed=None):
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    if x0 is None:  # random
        x0 = rng.uniform(-0.05, 0.05, size=len(_ordered_params(circuit)))
    hist = []

    def obj(x):
        e = _energy(_assign(circuit, x), h_matrix, physical_indices)
        hist.append(e)
        return e

    e0 = _energy(_assign(circuit, x0), h_matrix, physical_indices)
    t0 = time.time()
    res = minimize(
        obj,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.1, "tol": 1e-8, "disp": False},
    )
    return {
        "name": name,
        "initial_energy": e0,
        "final_energy": float(res.fun),
        "n_evals": len(hist),
        "wall_sec": time.time() - t0,
        "energy_history": hist,
    }
