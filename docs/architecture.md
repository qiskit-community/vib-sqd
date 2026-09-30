# Architecture

`vib_sqd` is a layered package: low-level primitives at the bottom,
orchestration at the top. Dependencies point **downward only** (a module never
imports from a layer above it), which keeps the low levels reusable and the
high levels easy to change.

```
                        ┌──────────────────────────────────────────────┐
  top / orchestration   │  run/pipeline, run/dualbasis, run/excited_state │
                        │  examples/drivers/ (CLI scripts on top of them) │
                        └───────────────────────┬──────────────────────┘
                                                │
        ┌───────────────────────────────────────┼───────────────────────────────┐
        │                                       │                               │
  ┌─────▼──────┐                    ┌───────────▼─────────┐         ┌──────────▼────────┐
  │ hardware/  │                    │ sqd/                │         │ initialization/   │
  │ aer,       │                    │ subspace_hamiltonian,│         │ vcc_initialization,│
  │ runtime,   │                    │ vibrational_recovery,│         │ sparse_uvccsd,     │
  │ transpile, │                    │ qsci_augment, ...    │         │ natural_modal      │
  │ suppression│                    └───────────┬──────────┘         └──────────┬────────┘
  └─────┬──────┘                                │                               │
        │                              ┌─────────▼─────────┐          ┌─────────▼────────┐
        └─────────────────────────────►│ circuits/         │          │ classical/       │
                                       │ initial_states,    │◄─────────┤ mop_forcefield,  │
                                       │ ansatz_enhanced,    │          │ vcc_extractor,   │
                                       │ chc, ucj_correlators│          │ vcc_solver,      │
                                       └─────────┬───────────┘          │ inp_template     │
                                                 │                     └─────────┬────────┘
                                       ┌─────────▼─────────┐                     │
                                       │ encoding/          │                     │
                                       │ direct (one-hot)   │                     │
                                       └─────────┬───────────┘                     │
                                                 │                               │
                                    ┌─────────────▼──────────────┐                │
                                    │ basis/ (harmonic,          │◄───────────────┘
                                    │  ho_hamiltonian, modal,    │
                                    │  modal_transform)          │   operators/ (pauli_utils, penalty)
                                    └─────────────────────────────┘
                                              primitives / math
```

## Layers (bottom → top)

| Layer | Package | Responsibility |
|---|---|---|
| primitives / math | `basis/`, `operators/` | HO matrix elements, ladder ops, the shared HO-Hamiltonian builder, Pauli helpers, penalties |
| classical | `classical/` | `.mop` force-field parsing (`mop_forcefield`), MidasCpp `.inp` generation (`inp_template`), VCC amplitude extraction/container (`vcc_extractor`, `vcc_solver`) |
| encoding | `encoding/` | force field → qubit Hamiltonian (`direct`, one-hot) |
| circuits | `circuits/` | initial states, ansätze (UVCCSD/CHC/VLUCJ/VIm-uCJ/Vg-uCJ) |
| initialization | `initialization/` | map classical VCC amplitudes → circuit parameters |
| sqd | `sqd/` | project onto sampled subspace, diagonalize, and the self-consistent recovery loop (ground + excited state) |
| hardware | `hardware/` | sampling: local Aer + IBM Runtime; transpilation; error suppression |
| orchestration | `run/` | the MidasCpp subprocess wrapper, the ansatz dispatcher, and the excited-state helpers every driver script in `examples/drivers/` is built on |

`analysis/` (plotting) and `io/` (Matlab interop) sit alongside this stack as
optional, non-load-bearing utilities.

## Design rules

- **Downward-only imports.** e.g. `encoding` may use `basis`/`operators`; `sqd`
  may use `encoding`; nothing in `basis`/`operators` imports upward.
- **Optional heavy deps are lazy.** `qiskit-aer` and `qiskit-ibm-runtime` are
  imported inside the functions that need them, so the package (and its tests)
  import without them. Install the `hardware` extra for the real-device path.
- **Credentials via environment only.** IBM Runtime reads `QISKIT_IBM_TOKEN` and
  `QISKIT_IBM_INSTANCE`; tokens are never stored in code or config.
- **Generated artifacts are not tracked.** Anything under `outputs/`, `report/`,
  or `**/runs/` (figures, circuit `.qpy`, samples, logs) is git-ignored.

## Where the force field comes from

The quantum side consumes a **force field**: harmonic frequencies plus
anharmonic coupling terms (`classical.forcefield_types.VibrationalForceField`).
That force field comes from an external classical tool — MidasCpp, or anything
that can produce a MidasCpp `.mop` operator file (the repo root's
`vib_sqd.classical.gaussian_freq_anharm_to_midas_mop` converts Gaussian
`freq=anharmonic` output; an analogous converter for Molpro/ORCA would use the
same `.mop` grammar). `classical.mop_forcefield.mop_to_forcefield` parses a
`.mop` into that same representation, so everything downstream — encoding,
circuits, hardware, SQD — is identical regardless of where the force field
came from:

```
  MidasCpp / Gaussian / ORCA / Molpro  ──►  .mop  ──►  mop_to_forcefield
                                                              │
                                                              ▼
                                       encoding → circuits → hardware → sqd → result
```

See [`examples/`](../examples/) for five worked, verified examples of this
whole path end to end (H2O, CH2O, NH3), and
[`docs/midascpp_installation.md`](midascpp_installation.md) for building
MidasCpp itself.
