# Reproducing the vibrational-SQD results

This folder has two step-by-step tutorials, starting from nothing but this
repo, a MidasCpp build, and (optionally) an IBM Quantum account.

- [`ground_state_tutorial/`](ground_state_tutorial/README.md) — reproduce a
  ground-state ZPE (H$_2$O, 4 modals/mode, modal basis) on a noiseless
  simulator, then optionally on real hardware.
- [`excited_state_tutorial/`](excited_state_tutorial/README.md) — reproduce
  the lowest vibrational excitation energy ($\Delta E_1$) for the same
  molecule, using the same underlying pipeline extended with an explicit
  excited-mode occupation.
- [`ansatz_physics_tutorial/`](ansatz_physics_tutorial/README.md) — why the
  VIm-uCJ/Vg-uCJ ansätze move population on a one-hot reference while
  VLUCJ's diagonal Jastrow correlator provably does not, derived from the
  actual gate generator matrices.
- [`sqd_workflow_tutorial/`](sqd_workflow_tutorial/README.md) — the SQD
  operating model as a mechanism (warm start → sample → subspace growth →
  recover), plus a noisy-simulator benchmark with one-hot post-selection.
- [`nh3_tutorial/`](nh3_tutorial/README.md) — a second molecule end to end,
  including a real MidasCpp convergence gotcha (NH3's C3v mode degeneracy)
  and its fix.

Both `ground_state_tutorial` and `excited_state_tutorial` use the real
driver scripts in [`drivers/`](drivers/) (`run_h2o_sweep.py` /
`run_excited_state_demo.py`) — nothing there is a separate
reimplementation. Read the ground-state tutorial first even if you only
care about excited states: it covers the shared setup (MidasCpp install,
Python environment, IBM Quantum credentials) that the excited-state
tutorial assumes you've already done.

## What you need

| | Ground state | Excited state |
|---|---|---|
| Python + this package installed (`pip install -e ".[hardware,dev]"`) | required | required |
| MidasCpp built and on `MIDASCPP_BIN` | required | required |
| Noiseless simulator (`--sim`) | yes | no such mode — see tutorial |
| Real IBM Quantum hardware (`--submit`) | optional | optional |
