# Tutorial: reproducing a ground-state ZPE (H$_2$O, modal basis, $M=4$)

This walks through reproducing one concrete ground-state cell: water
(H$_2$O), 4 modals per mode (12 qubits), modal (VSCF) basis, all five
ansätze (UVCCSD, VLUCJ, VIm-uCJ, Vg-uCJ, CHC). It uses the real driver
script, `run_h2o_sweep.py` — nothing here is a separate reimplementation.
`run_ch2o.py` and `run_ch2fcl.py` are structured identically for two other
molecules (6 and 9 modes).

You'll go through three stages: a classical-only preview (no qubits
touched at all), a noiseless simulator run (exact statevector sampling, no
hardware noise), and — as an optional final step — a real submission to
IBM Quantum hardware.

## 0. Prerequisites

**Python environment**, from the repo root:

```bash
pip install -e ".[hardware,dev]"
```

`hardware` pulls in `qiskit-ibm-runtime` (needed even for `--dry-run`/`--sim`,
since those import from `vib_sqd.hardware.transpilation`); `dev` pulls in
`pytest` if you want to run the test suite as a sanity check.

**MidasCpp** — a separate C++ program, not a Python package. This pipeline
uses it to solve the classical vibrational coupled-cluster (VCC) equations
that give each ansatz its initial amplitudes; without it, nothing here runs
at all (there is no pure-Python fallback). Build it by following
[`docs/midascpp_installation.md`](../../docs/midascpp_installation.md), then
point this pipeline at YOUR OWN install with three environment variables:

```bash
export MIDASCPP_BIN=/path/to/your/midascpp/release/bin/midascpp.x
export MIDAS_INSTALL_PREFIX=/path/to/your/midascpp/install
export MIDAS_INSTALL_DATADIR=/path/to/your/midascpp/install/data
```

**(Optional, only needed for the real-hardware step in §4)** IBM Quantum
credentials — set `QISKIT_IBM_TOKEN` and `QISKIT_IBM_INSTANCE` in your
environment (e.g. via a local, gitignored `.env` file and `export $(grep -v
'^#' .env | xargs)`, or direnv/python-dotenv). **Never commit these.**

`--dry-run` and `--sim` also need `QISKIT_IBM_TOKEN`/`QISKIT_IBM_INSTANCE`
set, even though they never submit a job — `--dry-run` transpiles against
the real backend's calibration data (to give an accurate CZ-count/depth
preview), and both stages import the same `hardware` module that expects
those two variables to exist. Only `--submit` actually spends QPU time.

All commands below assume you're in `examples/drivers/`:

```bash
cd examples/drivers
```

## 1. What MidasCpp is actually computing

For the ground state, MidasCpp runs a `#3 OccGroundState` VCC[2] calculation:
it solves for the vibrational coupled-cluster amplitudes ($t_1$, $t_2$) that
describe the correlated ground-state wavefunction on top of the VSCF
reference (all modes in their lowest modal). This repo's
`vib_sqd.classical.vcc_extractor` parses those amplitudes straight out of
MidasCpp's `.mout` file and maps them into each ansatz's initial circuit
parameters — this is the "warm start" that makes VQE unnecessary in this
pipeline (see [`docs/architecture.md`](../../docs/architecture.md) for the
full layer map and `inp_template.py` for the exact MidasCpp input block).

The first time you run any modal count, you'll see MidasCpp actually
execute (a few seconds to ~1 minute depending on mode count); every
subsequent run at that same modal count reuses the cached `.mout` file
(checked via a `PROD-ALLHO` marker in `ensure_clean_vcc`), so re-running the
same `--modals` value is fast.

## 2. Classical-only preview: `--dry-run`

Transpiles all five ansätze against the real `ibm_kingston` backend target
and reports circuit cost — no qubits, no QPU time, no job submitted:

```bash
python run_h2o_sweep.py --modals 4 --basis modal --dry-run
```

Expected output (one line per ansatz):

```
H2O 3 modes x 4 modal = 12 qubits, basis=modal, shots=100000
  reference-config energy (modal basis) = 4630.42 cm^-1
  UVCCSD   qubits=12 L3_CZ=798  L3_depth=1902
  VLUCJ    qubits=12 L3_CZ=196  L3_depth=436
  VIm-uCJ  qubits=12 L3_CZ=1302 L3_depth=3050
  Vg-uCJ   qubits=12 L3_CZ=1429 L3_depth=3217
  CHC      qubits=12 L3_CZ=496  L3_depth=793
```

These are the real transpiled numbers from this session's own run of this
exact command (`report/h2o_12q/modal_<ansatz>_dryrun.json`, default
`--layers 4`). The exact CZ-count/depth you get will drift slightly from
these — `ibm_kingston`'s calibration data (and hence the transpiler's
routing/optimization choices) changes over time — but should be in the
same ballpark; VLUCJ is by far the shallowest of the five. This also writes
`report/h2o_12q/modal_<ansatz>_dryrun.json` and a transpiled `.qpy` circuit
per ansatz for inspection.

## 3. Noiseless simulator: `--sim`

Draws exact statevector-derived samples (no hardware noise at all) and runs
the same self-consistent SQD recovery loop the real hardware path uses:

```bash
python run_h2o_sweep.py --modals 4 --basis modal --sim
```

Expected output, one line per ansatz:

```
  UVCCSD   sim recovered E=  4614.52 cm^-1 (ref 4630.42)
  VLUCJ    sim recovered E=  4630.42 cm^-1 (ref 4630.42)
  VIm-uCJ  sim recovered E=  4614.75 cm^-1 (ref 4630.42)
  Vg-uCJ   sim recovered E=  4614.75 cm^-1 (ref 4630.42)
  CHC      sim recovered E=  4630.42 cm^-1 (ref 4630.42)
```

These are the **real, verified numbers** for this exact command. UVCCSD/
VIm-uCJ/Vg-uCJ recover to within a fraction of a cm$^{-1}$ of the classical
VCC($M{=}4$) reference for water, 4614.43 cm$^{-1}$ — on a noiseless
simulator, SQD recovery should match the classical VCC answer almost
exactly for these three, since there's no hardware noise to recover from.

**VLUCJ and CHC land exactly on the un-recovered reference energy here — this
is expected, not a bug.** VLUCJ's diagonal Jastrow correlator commutes with
(is "inert" on) a one-hot computational-basis reference state — it changes
the state's global phase, never its measurement probabilities — so **no
warm-start angle, however large, changes VLUCJ's `--sim` sampling
distribution at all**, for any molecule (confirmed here for both H2O and
CH2O). See [`examples/ansatz_physics_tutorial/`](../ansatz_physics_tutorial/)
for the generator-matrix derivation. CHC's warm-start does move population — it has
real Givens-rotation gates, not just diagonal phases (verified for water:
$|\langle\text{ref}|\text{bound}\rangle|^2 \approx 0.99996$, a genuinely
nonzero shift) — but the shift scales with the classical VCC amplitudes
themselves, and at $M=4$ for both H2O and CH2O it is too small to survive
100,000-shot statistical sampling, so recovery still finds nothing to work
with beyond the reference configuration.

On **real hardware** (`--submit`, §4), physical noise itself provides the
bitstring spread recovery needs, independent of how large the warm-start
rotation was — which is why the real-hardware ledger for this exact cell
(`runs/recovery_table.json`) shows all five ansätze, including VLUCJ/CHC,
recovering to ≈4614.43 cm$^{-1}$: hardware noise, not warm-start rotation,
supplies the recoverable signal there. VLUCJ's inertness is structural and
holds regardless of system size or hardware noise; CHC's `--sim` recovery
may still work at other modal counts where the VCC amplitudes are larger,
but this has not been verified across the full sweep — treat CHC's
noiseless-simulator behavior as cell-dependent rather than assuming it
behaves like UVCCSD/VIm-uCJ/Vg-uCJ.

This writes `report/h2o_12q/modal_<ansatz>_results.json` per ansatz.

## 4. Real hardware (optional): `--submit`

**This spends real QPU quota on your own IBM Quantum account.** Only run
this once your credentials are set (§0) and you're ready to use hardware
time:

```bash
python run_h2o_sweep.py --modals 4 --basis modal --submit --backend ibm_kingston
```

This prints a job ID immediately:

```
  job_id=<some-job-id> shots=100000
```

then blocks waiting for the job to finish on the queue (this can take
anywhere from seconds to hours depending on queue congestion — real IBM
hardware queues, not a bug in this script). Once it completes, you'll see
per-ansatz recovered energies exactly like §3, but now derived from real
noisy hardware samples:

```
  UVCCSD   one-hot retained= 41.23%  recovered E=  ... cm^-1
  ...
```

**Your numbers will differ from any other run** — real hardware noise,
calibration drift, and queue-dependent conditions all vary run to run. This
is expected, not a failure. What should hold up qualitatively:
`onehot_retained_frac` (the fraction of shots that decoded to a physically
valid one-hot bitstring) should be well above the fully-random baseline,
and the recovered energy should land noticeably closer to the VCC($M{=}4$)
reference (4614.43 cm$^{-1}$) than the raw pre-recovery energy did — that's
the "before vs. after recovery" effect this pipeline's recovery loop
exists to produce.

## 5. Where results end up

Everything for this cell lives under `report/h2o_12q/`:

| file | contents |
|---|---|
| `modal_results.json` | consolidated summary: job ID, per-ansatz recovered energy, retention |
| `modal_<ansatz>_raw_counts.json` | every unique bitstring the hardware actually returned, unprocessed |
| `modal_<ansatz>_beforeafter.json` | top-50 bitstring distribution before vs. after recovery |
| `modal_<ansatz>_after_full.json` | complete (untruncated) post-recovery configuration distribution |
| `modal_<ansatz>_dryrun.json` | transpiled circuit cost (CZ count, depth) from §2 |
| `modal_<ansatz>_isa.qpy` / `_submitted.qpy` | the actual transpiled circuit, loadable via `qiskit.qpy.load` |

Swap `--modals 4` for `2`/`6`/`8` and `--basis modal` for `--basis ho` to
reproduce any other ground-state cell; swap `run_h2o_sweep.py` for
`run_ch2o.py` (6 modes) or `run_ch2fcl.py` (9 modes) for the other two
molecules — both take the exact same flags.

## 6. Troubleshooting

**`FileNotFoundError: midascpp.x not found at ...`** — `MIDASCPP_BIN` isn't
set, or points to a nonexistent file. Re-check §0; the error message names
the exact path it tried.

**MidasCpp crashes at 7+ modes with an `OccAllFund`-related error** — a
known upstream MidasCpp bug when it tries to enumerate excited-state
occupations for many modes at once. This pipeline works around it
automatically for the ground state (`ground_state_only` defaults to `True`
at `n_modes >= 7`, see `vib_sqd.run.pipeline.run_midascpp`), so you
shouldn't hit this from `run_h2o_sweep.py`/`run_ch2o.py`/`run_ch2fcl.py`
directly — it's only relevant if you call `run_midascpp` yourself with
`occup` set and `n_modes >= 7`. See
[`MIDASCPP_BUG_OccAllFund.md`](../drivers/MIDASCPP_BUG_OccAllFund.md) in
`examples/drivers/` for the full writeup.

**Memory blows up in HO basis at large qubit counts** — the HO-basis
recovery loop uses `ho_submatrix` (each molecule's own force-field module),
which evaluates only the small submatrix actually needed for each recovery
iteration, never the full $M^{N_\text{modes}}$ Hamiltonian. If you're
calling lower-level functions directly rather than going through
`run_h2o_sweep.py`/`run_ch2o.py`/`run_ch2fcl.py`, make sure you're using
that submatrix path — building the full sparse Hamiltonian is what caused
out-of-memory crashes for CH$_2$ClF's 9-mode HO basis during this
pipeline's own development.
