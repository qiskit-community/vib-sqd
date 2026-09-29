# Tutorial: a second molecule end to end (NH$_3$), and a real MidasCpp convergence gotcha

This walks through the ammonia (NH$_3$) sweep: the same MidasCpp → VCC →
warm-start → SQD-recovery pipeline as the [ground-state
tutorial](../ground_state_tutorial/README.md), applied to a molecule with
an exact symmetry degeneracy that genuinely breaks MidasCpp's default
classical bootstrap — and the fix for it. It uses the real driver script,
`run_nh3.py`.

**Read the [ground-state tutorial](../ground_state_tutorial/README.md)
first** if you haven't already — this one assumes MidasCpp, the Python
environment, and (for the hardware step) IBM Quantum credentials are
already set up exactly as described there. Same working directory:

```bash
cd examples/drivers
```

## 1. NH$_3$'s force field and its exact C3v degeneracy

NH$_3$ has 6 vibrational modes in `NH3_MOP` (a MidasCpp `.mop` quartic
force field), truncated with `N_HO=11` harmonic-oscillator primitives per
mode. Ammonia's C3v molecular symmetry means two pairs of modes are
**exactly** degenerate: the asymmetric N-H stretch (modes 2/3) and the
asymmetric bend (modes 4/5), recorded in `nh3_forcefield.py`'s
`DEGENERATE_MODE_PAIRS`.

**No symmetry reduction is applied anywhere in this pipeline** — all 6
modes are treated uniformly. The degeneracy is real physics, not an
approximation artifact: it's exact down to the force field's own stored
`#1 ScaleFactors` coefficients, not just the resulting frequencies (you can
check this yourself — `doc.scale_factors[2] == doc.scale_factors[3]` and
`doc.scale_factors[4] == doc.scale_factors[5]` are bit-identical after
parsing `NH3_MOP`).

## 2. The classical VSCF blocker (and its fix)

When this sweep was first run for NH3, the classical VCC bootstrap
**failed** at every modal count with the same root cause: MidasCpp's
batched `OccGroundState + OccAllFund` VSCF calculation genuinely
**oscillates** rather than converges for the excited state of one mode in
each degenerate pair. This is confirmed in the MidasCpp log: the
per-iteration energy bounces between roughly `3.59e-2` and `3.94e-2` a.u.
for all 100 allowed iterations, with no decaying trend — a real numerical
instability at the exact degeneracy (the self-consistent field cannot pick
a stable rotation within the degenerate subspace), not a slow-convergence
issue that more iterations would fix.

Two facts made the fix straightforward:
1. The **ground-state** VSCF in that same batch converges cleanly
   regardless — the failure is specific to the excited-state calculation.
2. Each individual excited state, **including the degenerate pair**,
   converges cleanly (in about 6 iterations) when solved as an **isolated**
   single-state VSCF (`occup=[...]`) rather than in the `OccAllFund` batch.

The fix, implemented in `run_nh3.py`'s `ensure_clean_vcc_nh3`: the main
sweep requests **ground-state-only** VSCF/VCC (all it needs for ansatz
warm-start amplitudes), and the `--fundamentals` CLI flag solves each of
the 6 singly-excited configurations as an isolated single-state VSCF
instead of a batch. Read `ensure_clean_vcc_nh3`'s own docstring for the
authoritative explanation — it's the same fix this repo's
`run_midascpp`/`ground_state_only` mechanism provides generically (see the
ground-state tutorial's §6 troubleshooting entry), just forced on for NH3
regardless of mode count, since NH3's own blocker is a genuine VSCF
instability, not the (unrelated, mode-count-triggered) `OccAllFund` crash
that mechanism was originally built for.

## 3. Classical-only preview: `--dry-run`

```bash
python run_nh3.py --modals 2 --basis modal --dry-run
```

Expected output:

```
NH3 6 modes x 2 modal = 12 qubits, basis=modal
  reference-config energy (modal basis) = 7389.17 cm^-1
  UVCCSD   qubits=12 L3_CZ=140 L3_depth=291
  VLUCJ    qubits=12 L3_CZ=33  L3_depth=87
  VIm-uCJ  qubits=12 L3_CZ=260 L3_depth=559
  Vg-uCJ   qubits=12 L3_CZ=260 L3_depth=559
  CHC      qubits=12 L3_CZ=64  L3_depth=148
```

These are the real transpiled numbers from this session's own run of this
exact command. VLUCJ is again by far the shallowest, same as for water —
see the ground-state tutorial's §2 for why.

## 4. Noiseless simulator: `--sim`

```bash
python run_nh3.py --modals 2 --basis modal --sim
```

Expected output:

```
  UVCCSD   sim recovered E=  7387.18 cm^-1 (ref 7389.17)
  VLUCJ    sim recovered E=  7389.17 cm^-1 (ref 7389.17)
  VIm-uCJ  sim recovered E=  7387.18 cm^-1 (ref 7389.17)
  Vg-uCJ   sim recovered E=  7387.18 cm^-1 (ref 7389.17)
  CHC      sim recovered E=  7389.17 cm^-1 (ref 7389.17)
```

Same pattern as the ground-state tutorial's H$_2$O/CH$_2$O result: VLUCJ
and CHC land exactly on the un-recovered reference energy on this
noiseless simulator, for the same reason (VLUCJ's diagonal Jastrow is
provably inert on a one-hot reference; CHC's warm-start is real but too
small at this small a system to survive shot noise). See the ground-state
tutorial's §3 for the full explanation, and
[`examples/ansatz_physics_tutorial/`](../ansatz_physics_tutorial/) for the
generator-matrix derivation.

## 5. Real hardware (optional): `--submit`

**This spends real QPU quota.** Same credential requirements as the
ground-state tutorial:

```bash
python run_nh3.py --modals 2 --basis modal --submit --backend ibm_kobe
```

The real hardware job for this exact cell (`ibm_kobe`, job
`d9gdc5qneu4c739qmfk0`, 100,000 shots, all 5 ansätze) recovered to
$7385.99\,\mathrm{cm}^{-1}$ for every ansatz except VLUCJ ($7385.996$) — a
tight cluster, all within a few thousandths of a cm$^{-1}$ of each other.
As on real hardware for water/CH2O, this is expected: physical noise
itself supplies the bitstring spread the noiseless simulator's `--sim`
path lacks for VLUCJ/CHC at small system sizes.

## 6. Other flags

| flag | effect |
|---|---|
| `--uniform-baseline` | classical control: draw uniform-random one-hot configs instead of a quantum circuit, run through the *identical* recovery loop |
| `--fundamentals` | compute the 6 singly-excited diagonal config energies via the isolated single-state VSCF fix from §2 — as-computed only, no independent literature reference exists for NH3's fundamentals in this repo |
| `--shots N` | shots per ansatz (default 100,000) |
| `--ansatze ...` | restrict to a subset of the 5 ansätze (default: all) |

## 7. Where results end up

Everything for this cell lives under `report/nh3_12q/` (swap `--modals 2`
for `4`/`6` for the 24q/36q cells, which write to `nh3_24q`/`nh3_36q`):

| file | contents |
|---|---|
| `modal_results.json` | consolidated summary: job ID, per-ansatz recovered energy, retention |
| `modal_<ansatz>_dryrun.json` | transpiled circuit cost from §3 |
| `modal_<ansatz>_results.json` | per-ansatz `--sim` result from §4 |
| `modal_fundamentals.json` | the 6 singly-excited diagonal energies from `--fundamentals` |

## Takeaways

- NH3's C3v degeneracy (modes 2/3 and 4/5) is exact and genuinely
  destabilizes MidasCpp's batched excited-state VSCF (a real SCF
  oscillation, not a slow-convergence issue). Ground-state-only VSCF/VCC
  is the fix the main sweep uses; isolated single-state VSCF is the fix
  `--fundamentals` uses.
- The same 5-ansatz, self-consistent-recovery pipeline generalizes to a
  new molecule with zero changes beyond a molecule-specific force-field
  module (`nh3_forcefield.py`) and, here, one molecule-specific VCC
  bootstrap override (`ensure_clean_vcc_nh3`).
- `run_nh3.py`'s CLI flags map directly onto this repo's library functions
  (`vib_sqd.run.pipeline`, `vib_sqd.run.dualbasis`) — `--dry-run`/`--sim`/
  `--uniform-baseline`/`--fundamentals` are all classical/simulator-only
  and safe to re-run any time; `--submit` costs real hardware quota.
