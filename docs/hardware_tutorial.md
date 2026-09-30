# Running on IBM Quantum hardware

The runnable, verified path from a `.mop` force field to a real-hardware
submission is the tutorials in [`examples/`](../examples/) — start with
[`examples/ground_state_tutorial/`](../examples/ground_state_tutorial/README.md).
This page documents the underlying `vib_sqd.hardware` API those tutorials'
driver scripts are built on, for anyone writing a new driver rather than
using an existing one.

## The three stages every driver in `examples/drivers/` implements

```
force field → encoded qubit Hamiltonian → measured circuit → sample → SQD recovery
```

Only the **sampling** step changes between the three stages:

| Stage | What it does | Spends QPU time? |
|---|---|---|
| `--dry-run` | build + transpile the circuits, report resource cost (2q-gate count, depth) against the real backend's calibration | No |
| `--sim` | exact statevector-derived sampling (`vib_sqd.hardware.aer`), no hardware noise | No |
| `--submit` | real submission via `vib_sqd.hardware.runtime` (`SamplerV2`) | **Yes** |

Always run `--dry-run` before `--submit` on a new cell — it uses the real
backend's live calibration data (via `generate_preset_pass_manager`) to give
an accurate cost preview with zero QPU spend.

## Credentials

IBM Runtime reads credentials from the environment — never put a token in a
file or commit one:

```bash
export QISKIT_IBM_TOKEN="<your IBM Quantum token>"
export QISKIT_IBM_INSTANCE="<your instance / CRN>"
```

`vib_sqd.hardware.runtime.run_on_ibm_runtime_with_exact` (the real-submission
entry point) reads these two variables directly; both `--dry-run` and
`--submit` need them, since `--dry-run` also transpiles against the real
backend's live `Target` for an accurate preview.

## What the real-device path does

1. Compute exact reference probabilities from the **logical** circuit (before
   ISA transpilation) — `vib_sqd.hardware.aer.exact_probabilities_from_circuit`.
2. Optionally choose the best-connected physical qubit line
   (`vib_sqd.hardware.transpilation.choose_best_connected_line`).
3. Transpile to the backend ISA (`generate_preset_pass_manager`).
4. Apply error suppression (dynamical decoupling + Pauli twirling) via
   `vib_sqd.hardware.suppression.RuntimeSuppressionConfig`/
   `apply_sampler_suppression_options`.
5. Submit with `SamplerV2` (job mode), collect counts.
6. Decode one-hot bitstrings, feed the self-consistent SQD recovery loop
   (`vib_sqd.sqd.vibrational_recovery`) to get the recovered energy.

## Safety notes

- `--dry-run` never submits. `--submit` spends your IBM Quantum quota.
- Credentials are read from env vars only; keep them out of any config file
  and out of git.
- Start small (few qubits / low shots) and inspect the `--dry-run` resource
  report before scaling up — see the ground-state tutorial's §2 for what
  that report actually looks like for a real cell.
