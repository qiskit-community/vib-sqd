<div align="left">

  [![License](https://img.shields.io/github/license/qiskit-community/vib-sqd.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

# vib-sqd

> [!NOTE]
> This repository is under active development and the code here should not be considered stable. This is not an officially supported IBM Quantum software.

A Qiskit-native workflow for vibrational-state simulation with sample-based
quantum diagonalization (SQD). It takes a molecular force field (from
MidasCpp, or anything that can produce a MidasCpp `.mop` operator file —
Gaussian, ORCA, Molpro), extracts classical vibrational coupled-cluster (VCC)
amplitudes, uses them to warm-start a family of one-hot ansätze (UVCCSD,
VLUCJ, VIm-uCJ, Vg-uCJ, CHC), samples the resulting circuits — on a noiseless
simulator or real IBM Quantum hardware — and recovers the ground- or
excited-state energy with a self-consistent subspace-diagonalization
recovery loop.

## Overview

The workflow mirrors the structure of other Qiskit SQD examples:

1. build a Hamiltonian from a classical force field
2. prepare a warm-started circuit family
3. sample bitstrings (simulator or real hardware)
4. recover physically valid one-hot configurations and grow the sampled subspace
5. project the Hamiltonian into that subspace and diagonalize classically
6. compare against the classical VCC reference

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — the package's layer map
- [`docs/hardware_tutorial.md`](docs/hardware_tutorial.md) — the real-hardware submission path
- [`docs/midascpp_installation.md`](docs/midascpp_installation.md) — building MidasCpp

## Installation

```bash
# Minimal install (simulator path only)
pip install -e .

# With the real-hardware path (qiskit-ibm-runtime)
pip install -e ".[hardware]"

# Full developer install
pip install -e ".[dev,hardware]"
```

## Getting started

Five worked, verified tutorials in [`examples/`](examples/README.md) — start
with [`examples/ground_state_tutorial/`](examples/ground_state_tutorial/README.md):

```bash
pip install -e ".[hardware,dev]"
cd examples/drivers
python run_h2o_sweep.py --modals 4 --basis modal --sim       # noiseless simulator, no QPU time
python run_h2o_sweep.py --modals 4 --basis modal --dry-run   # circuit cost preview, no QPU time
                                                                # (needs QISKIT_IBM_TOKEN/QISKIT_IBM_INSTANCE
                                                                #  to transpile against real calibration data)
```

MidasCpp is a separate C++ program this pipeline calls to solve the classical
VCC equations — without it, nothing here runs (there is no pure-Python
fallback). See [`docs/midascpp_installation.md`](docs/midascpp_installation.md)
to build it, then point this package at your own install:

```bash
export MIDASCPP_BIN=/path/to/your/midascpp/release/bin/midascpp.x
export MIDAS_INSTALL_PREFIX=/path/to/your/midascpp/install
export MIDAS_INSTALL_DATADIR=/path/to/your/midascpp/install/data
```

## Data availability

`.mop` force-field inputs for seven systems (water, formaldehyde, CH$_2$ClF,
acetylene, ammonia, and a glycine cation active space) ship in
[`data/mop_files/`](data/mop_files/). A separate dataset — real-hardware job
ledgers, per-cell recovered energies, physical-qubit layouts, and timing for
every (molecule, modal count, basis, ansatz) setting studied — is archived on
Zenodo: **[10.5281/zenodo.23011464](https://doi.org/10.5281/zenodo.23011464)**.
This repository's own `--sim`/`--submit` runs reproduce the *method*; they do
not replay the archived numbers verbatim (hardware noise varies run to run).

## Ansätze

| Ansatz | Structure |
|---|---|
| **UVCCSD** | full unitary VCC singles-and-doubles |
| **VLUCJ** | diagonal-Jastrow-correlated, one-hot; shallowest of the five, but its correlator is provably inert on a one-hot reference (see [`examples/ansatz_physics_tutorial/`](examples/ansatz_physics_tutorial/README.md)) |
| **VIm-uCJ** | imaginary (anti-Hermitian) correlator — genuinely rotates amplitude onto the correlation tail |
| **Vg-uCJ** | fully complex correlator — adds independent relative-phase control, useful when multiple doubles interfere coherently |
| **CHC** | compact heuristic circuit: Givens singles blocks + a compact pair-correlation phase block |

## Testing

```bash
pytest test/ -q
```

One test requires a local MidasCpp install (`MIDASCPP_BIN`) and skips itself
cleanly if one isn't found.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup
and development instructions. This project adheres to the
[Qiskit Code of Conduct](CODE_OF_CONDUCT.md).

## Citing this package

See [`CITATION.cff`](CITATION.cff) — GitHub renders a "Cite this repository"
button from it. The Zenodo record above is the citation for the dataset,
not the code.

## License

This project is licensed under the [Apache License 2.0](LICENSE.txt).
