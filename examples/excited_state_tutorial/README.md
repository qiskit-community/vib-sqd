# Tutorial: reproducing an excited-state gap ($\Delta E_1$, H$_2$O, modal basis, $M=6$)

This reproduces one concrete excited-state cell: water's lowest vibrational
excitation ($\Delta E_1$), 6 modals per mode (18 qubits), modal (VSCF)
basis, VLUCJ ansatz. It uses the real driver script,
`run_excited_state_demo.py`.

**Read the [ground-state tutorial](../ground_state_tutorial/README.md)
first** if you haven't already — this one assumes MidasCpp, the Python
environment, and (for the hardware step) IBM Quantum credentials are
already set up exactly as described there. This tutorial only covers what's
different about the excited state.

## 0. Prerequisites

Same as the ground-state tutorial's §0 — Python environment, MidasCpp, and
(optional) IBM Quantum credentials. Same working directory:

```bash
cd examples/drivers
```

## 1. What's physically different about the excited state

The ground state's VSCF reference has every mode in its own lowest modal
(occupation `[0, 0, 0]` for water's 3 modes). The excited state used here
promotes water's lowest-frequency fundamental mode to its first excited
modal — occupation `[0, 1, 0]` (mode index 1, absolute modal 1) — and runs a
**separate** MidasCpp `#3 Occup` VCC calculation for that specific
occupation, producing that excited state's own $t_1$/$t_2$ amplitudes. This
is not the ground-state amplitudes reused with a different label — it's a
genuinely different classical calculation. See
`vib_sqd.run.excited_state`'s module docstring for the exact mechanism, and
`build_excited_amps` for where the `#3 Occup` MidasCpp run happens.

The quantum circuit construction itself (ansatz layout, warm-start binding)
is identical to the ground state — only the classical amplitudes and the
Hamiltonian-lookup step in the recovery loop are occupation-aware.

One consequence worth knowing about: vibrational coupled-cluster excitation
operators only ever act *upward* from the reference. At low modal counts
($M=2$), the excited mode's own reference occupation is already the highest
available modal, so that particular channel's warm-start amplitude is
exactly zero — this is a correct physics fact about the VCC ansatz, not a
bug, and you'll see it if you try `--modals 2`.

## 2. Classical-only preview: `--dry-run`

```bash
python run_excited_state_demo.py --modals 6 --molecules h2o --ansatze VLUCJ --basis modal --dry-run
```

This transpiles the excited-state VLUCJ circuit against the real
`ibm_kingston` backend and reports its cost — same zero-QPU-time preview as
the ground-state tutorial's `--dry-run`. Needs `QISKIT_IBM_TOKEN`/
`QISKIT_IBM_INSTANCE` set for the same reason (accurate calibration-aware
transpilation), but submits nothing.

## 3. No noiseless-simulator mode here

Unlike `run_h2o_sweep.py`, `run_excited_state_demo.py` only has
`--dry-run` and `--submit` — there is no `--sim` stage for the excited
state. If you want to sanity-check the recovery math without spending QPU
time, the lowest-level building blocks (`excited_recovery_loop`,
`build_excited_amps`) are also directly callable from a Python shell; the
CLI itself just doesn't wire up a `--sim` flag for this script.

## 4. Real hardware (optional): `--submit`

**This spends real QPU quota.** Same credential requirements as the
ground-state tutorial:

```bash
python run_excited_state_demo.py --modals 6 --molecules h2o --ansatze VLUCJ --basis modal --submit
```

Expected output:

```
=== h2o (modals=6, basis=modal) ===
  job_id=<some-job-id> shots=100000
  VLUCJ            one-hot retained= 4.545%  excited-state recovered E=  6166.54 cm^-1
```

The `6166.54 cm^-1` figure above is the **real result from this session's
own run of this exact cell**
(`report/h2o_excited/excited_modal_results_modals6.json`). Your own run will
differ (real hardware noise varies run to run — see the ground-state
tutorial's §4 for why that's expected), but should land in the same
ballpark.

To turn this into a $\Delta E_1$ number, you need the ground state's own
SQD-recovered energy for the **same** modal count and basis (not the raw
reference-configuration energy) — run the ground-state tutorial's
`--submit` step at `--modals 6 --basis modal` first, then:

$$\Delta E_1^{\rm SQD} = E_1^{\rm SQD} - \mathrm{ZPE}^{\rm SQD}$$

For one real pair of numbers this way: $E_1^{\rm SQD} = 6166.54$,
$\mathrm{ZPE}^{\rm SQD} = 4614.25$ (water, modals=6, modal basis, VLUCJ),
giving $\Delta E_1^{\rm SQD} = 1552.29$ cm$^{-1}$ — within 0.07 cm$^{-1}$ of
the classical VCC($M{=}6$) reference, 1552.22 cm$^{-1}$. It's the
*recovered* ZPE, not the reference-config energy, that belongs in this
formula — using the wrong one silently gives the wrong $\Delta E_1$.

## 5. Both bases

The `--basis` flag (added to this script alongside the ground-state HO/modal
comparison) works the same way here:

```bash
python run_excited_state_demo.py --modals 6 --molecules h2o --ansatze VLUCJ --basis ho --submit
```

## 6. Where results end up

Everything lives under `report/h2o_excited/`:

| file | contents |
|---|---|
| `excited_<basis>_results_modals<M>.json` | consolidated summary: job ID, occupation, per-ansatz recovered energy |
| `excited_<basis>_<ansatz>_beforeafter.json` | top-50 bitstring distribution before vs. after recovery |
| `excited<basis_suffix>_<ansatz>_isa.qpy` / `_submitted.qpy` | the transpiled circuit |

(`basis_suffix` is empty for modal basis, `_ho` for HO basis — matches the
ground-state convention, except the consolidated results filename, which
always names the basis explicitly.)

Swap `--modals 6` for `2`/`4`/`8` to reproduce other excited-state cells, or
`--molecules ch2o`/`ch2fcl` for the other two molecules (their lowest
fundamental mode is chosen automatically via each molecule's
`LOWEST_FUNDAMENTAL_MODE`) — `--ansatze` also accepts any of UVCCSD, VLUCJ,
VIm-uCJ, Vg-uCJ, CHC (or several at once), and defaults to all five if
omitted.

## 7. Troubleshooting

Same MidasCpp path issues as the ground-state tutorial's §6 apply here —
`ensure_clean_vcc_excited` calls the same underlying
`vib_sqd.run.pipeline.run_midascpp`. The
`ho_submatrix_fn` parameter on `excited_recovery_loop` (used automatically
when `--basis ho`) avoids the same out-of-memory failure mode described
there, for the excited state's own recovery loop.
