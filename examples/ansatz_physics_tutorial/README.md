# Tutorial: ansatz & physics — why VIm-uCJ and Vg-uCJ move population, and VLUCJ's Jastrow doesn't

This derives, from the actual gate generator matrices, *why* VLUCJ's
diagonal Jastrow correlator is inert on a one-hot reference state while
the imaginary correlator (VIm-uCJ) and the fully complex correlator
(Vg-uCJ) genuinely move amplitude onto a doubly-excited configuration.
It's the physics behind the recurring "VLUCJ/CHC land on the un-recovered
reference energy" observation in the ground-state and NH3 tutorials —
read this if you want the actual math, not just the empirical fact.

**Key result:** a *real diagonal* correlator can only add phases (inert);
an *imaginary* (anti-Hermitian) correlator rotates amplitude onto the
doubly-excited configuration — genuine pair correlation.

## 0. Prerequisites

Same Python environment as the other tutorials (`pip install -e
".[hardware,dev]"`). This tutorial does **not** need MidasCpp — everything
here is pure linear algebra on small, explicit matrices (a minimal
2-mode/2-modal toy system), not a real molecule's amplitudes.

```bash
cd examples/ansatz_physics_tutorial
python reproduce_ansatz_physics.py
```

This calls `analyze_imucj_math.py` and `analyze_gucj_math.py` in
`examples/drivers/` exactly as documented below — nothing here is a
separate reimplementation.

## 1. The generators: why the diagonal Jastrow is inert

On a minimal 2-mode/2-modal system (4-dimensional one-hot physical
subspace, basis `{00, 01, 10, 11}` — `00` is the reference, `11` is the
target double), we extract each correlator's generator $G$ (with
$U = e^{\theta G}$) restricted to that physical subspace:

- **VIm-uCJ generator** is *real, antisymmetric, off-diagonal*
  (reference↔double) → an SO(2) rotation that **moves amplitude**.
- **VLUCJ CP generator** is *imaginary, diagonal on the reference* → a
  **phase**, and its commutator with any diagonal $H$ is zero, so it can
  never lower the energy — *inert*.

Running `reproduce_ansatz_physics.py` prints both generators explicitly:

```
VIm-uCJ 2-level generator (SO(2) rotation -> transfers amplitude):
[[ 0.  1.]
 [-1.  0.]]
VLUCJ CP 2-level generator (diagonal phase -> no amplitude transfer):
[[0.+1.j 0.+0.j]
 [0.+0.j 0.+0.j]]
```

and the **transfer map** — the punchline — is the maximum weight each
correlator can move from the reference `00` onto the double `11`, swept
over the correlator's own parameter:

```
VIm-uCJ  max |<11|U|00>|^2 = 1.0000
VLUCJ    max |<11|U|00>|^2 = 0.0000e+00  (inert)
Givens   max |<11|U|00>|^2 = 0.0000e+00  (single-mode: cannot reach a double)
```

One VIm-uCJ correlator can rotate the reference *entirely* onto the
double ($|c|^2$ reaches 1.0); the VLUCJ CP moves **exactly nothing** (not
approximately — this is a provable, structural zero, not a small number);
and a single-mode Givens rotation can reach the single-excitation `10`
but **never** the two-mode double `11` — it simply doesn't have the
two-qubit structure to do so.

This is the exact, general mechanism behind the ground-state tutorial's
observation that VLUCJ's `--sim` recovery lands precisely on the
un-recovered reference energy for water and CH2O: it isn't a bug, a
small-amplitude effect, or specific to those two molecules — VLUCJ's
diagonal Jastrow simply cannot change measurement probabilities on a
one-hot reference, for *any* input amplitudes, on *any* molecule.

## 2. What Vg-uCJ adds: complex amplitude on the double

Vg-uCJ factorizes as $U = e^{i\phi G_{\rm ph}} \cdot e^{i\theta G_{\rm rot}}$:
$G_{\rm rot}$ is the VIm-uCJ generator (transfers amplitude), $G_{\rm ph}$
is an *off-diagonal CHC-style* phase acting on the **double** (not the
inert same-modal CP acting on the *reference*). VIm-uCJ reaches only
*real* double-amplitudes; Vg-uCJ reaches the **full complex disk**:

```
VIm-uCJ: c_dbl is REAL, range [-1.000, +0.000]
Vg-uCJ : c_dbl fills a COMPLEX disk, |c_dbl|max=1.000, arg spans [-3.14, +3.14] rad
```

This only matters when several doubles must interfere with *opposite
sign* — a two-double toy model shows VIm-uCJ caps at fidelity 0.667 to the
target state, while Vg-uCJ (with its extra phase handle) reaches 1.0:

```
best fidelity  VIm-uCJ (phase forced 0) = 0.6667
best fidelity  Vg-uCJ  (phase free)      = 1.0000
```

On systems with one dominant double (e.g. small active spaces at low
modal counts), VIm-uCJ and Vg-uCJ tie — the extra phase handle only pays
off once multiple doubles need to interfere coherently, which matters more
at larger/deeper systems.

## Takeaways

- The diagonal Jastrow is **provably inert** on the one-hot reference
  ($[G_{\rm cp}, H] = 0$) — not approximately small, exactly zero.
- **VIm-uCJ** (imaginary correlator) rotates amplitude onto the
  correlation tail — genuine, robust pair correlation, at the fewest
  parameters.
- **Vg-uCJ** adds relative-phase control; it matters only for multi-double
  coherent interference (larger/deeper systems).

See [`examples/sqd_workflow_tutorial/`](../sqd_workflow_tutorial/) for how
these ansätze feed the SQD sample→recover loop, and how they behave under
realistic hardware noise.
