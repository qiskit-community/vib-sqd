#!/usr/bin/env python3

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

r"""
Thin convenience driver for README.md's tutorial: derives, from the actual
gate generator matrices, why VLUCJ's diagonal Jastrow correlator is inert
on the one-hot reference while VIm-uCJ (imaginary correlator) and Vg-uCJ
(fully complex correlator) move amplitude onto the doubly-excited
configuration -- exactly the two analysis scripts this tutorial is built
on, run and reported here for convenience.

This is NOT a reimplementation -- it imports and calls
analyze_imucj_math.py / analyze_gucj_math.py exactly as documented, from
this repo's examples/drivers/ directory.

Usage:
    python reproduce_ansatz_physics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DRIVERS_DIR = _HERE.parent / "drivers"
sys.path.insert(0, str(_DRIVERS_DIR))

import numpy as np  # noqa: E402


def main() -> None:
    import analyze_imucj_math as im_math

    print("=== Part 1: generator matrices (analyze_imucj_math.py) ===\n")
    Gi, Gc = im_math.analyze_generators()

    blk_i, blk_c, _ = im_math.two_level_blocks(Gi, Gc)
    print("\nVIm-uCJ 2-level generator (SO(2) rotation -> transfers amplitude):")
    print(np.real(blk_i))
    print("VLUCJ CP 2-level generator (diagonal phase -> no amplitude transfer):")
    print(blk_c)

    thetas = np.linspace(-np.pi, np.pi, 121)
    w_im = im_math.transfer_map(im_math.imucj_correlator_circuit, thetas)
    w_cp = im_math.transfer_map(im_math.vlucj_cp_circuit, thetas)
    w_gv = im_math.transfer_map(im_math.givens_circuit, thetas)
    dbl = im_math.PHYS_LABELS.index("11")
    print("\nMax weight transferred from reference '00' onto double '11':")
    print(f"  VIm-uCJ  max |<11|U|00>|^2 = {w_im[:, dbl].max():.4f}")
    print(f"  VLUCJ    max |<11|U|00>|^2 = {w_cp[:, dbl].max():.4e}  (inert)")
    print(
        f"  Givens   max |<11|U|00>|^2 = {w_gv[:, dbl].max():.4e}  "
        f"(single-mode: cannot reach a double)"
    )

    print("\n=== Part 2: complex-amplitude reach (analyze_gucj_math.py) ===\n")
    import analyze_gucj_math as g_math

    g_math.main()


if __name__ == "__main__":
    main()
