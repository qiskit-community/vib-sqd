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

"""
Generate a corrected MidasCpp ``.inp`` that forces the **harmonic-oscillator**
primitive basis (not the default B-splines).

The critical fix (found by reading MidasCpp ``src/input/BasDef.h`` + CHANGELOG):
``#3 HoBasis N`` only sets the number of HO quantum numbers/modals; the *basis
type* is the separate ``#3 BasisType`` keyword whose default is ``BSPLINE``.
Without ``#3 BasisType HO`` the output reports ``PROD-ALLBSPLINES``. With it,
``PROD-ALLHO``.

If the ``.mop`` carries a ``#1 ScaleFactors`` block (force field in dimensionless
coordinates), ``#3 UseScalingFreqs`` is also required.

Note (learned by running): ``HoBasis`` must be >= the number of modals actually
requested by VSCF (OccAllFund). We set it to ``max(ho_basis, n_modals + slack)``.
"""

from __future__ import annotations


def build_midas_input(
    *,
    mop_filename: str,
    n_modals: int,
    use_scaling_freqs: bool,
    vcc_method: str = "VCC[2]",
    ho_basis: int | None = None,
    it_eq_resid_thr: str = "1.0e-8",
    it_eq_max_it: int | None = None,
    occup: list[int] | None = None,
    ground_state_only: bool = False,
    io_level: int = 5,
    n_modes: int | None = None,
    vcc_print_elements: int | None = None,
    include_vci: bool = False,
    vci_eigenvals: int = 10,
) -> str:
    """Return the text of a MidasCpp ``.inp`` file.

    Parameters
    ----------
    mop_filename
        Operator file name (relative to the run directory).
    n_modals
        Number of modals per mode requested for VSCF/VCC.
    use_scaling_freqs
        Emit ``#3 UseScalingFreqs`` (needed when the ``.mop`` has ScaleFactors).
    vcc_method
        e.g. ``"VCC[2]"`` or ``"VCC[3]"``.
    ho_basis
        Number of HO primitive functions. Defaults to ``max(10, n_modals + 4)``
        so VSCF has enough primitives above the requested modals.
    it_eq_max_it
        If set, emit ``#3 ItEqMaxIt <n>`` in the Vcc block. VCC needs more
        iterations as the number of coupled modes grows.
    occup
        If given, emit an explicit ``#3 Occup`` vector (space-separated, one
        entry per mode) INSTEAD of ``OccGroundState``/``OccAllFund``. Used by
        the per-fundamental path (``fundamentals.py``) to target one excited
        configuration and avoid the ``OccAllFund`` batch-setup crash at large N.
    n_modes
        Number of modes, used only to size the default ``vcc_print_elements``
        (ignored if ``vcc_print_elements`` is given explicitly).
    vcc_print_elements
        How many solution-vector components MidasCpp's own ``#3 IoLevel``
        setting requests it print (n_out = max(5, IoLevel) in
        MidasCpp's src/vcc/Vcc.cc) -- our extractor needs every active-space
        t1/t2 amplitude to appear in the .mout. Defaults to the exact
        singles+doubles count for (n_modes, n_modals) plus a small margin,
        e.g. ~1800 for 9 modes/8 modals. IMPORTANT: requesting far more than
        needed (e.g. a flat 100000 regardless of system size) is not just
        wasteful -- for a 9-mode molecule at modals>=6 it made MidasCpp's own
        sort/print step consume 100s of GB and never finish, even though the
        actual VCC[2] calculation itself takes under a second and 100MB once
        the requested count matches what's actually needed. If ``n_modes``
        is not given, falls back to the old flat default (100000).
    include_vci
        Also emit a VCI block (for validation / excited states).
    """
    if vcc_print_elements is None:
        if n_modes is not None:
            n_exc = max(n_modals - 1, 0)
            needed = n_modes * n_exc + (n_modes * (n_modes - 1) // 2) * n_exc * n_exc
            vcc_print_elements = max(needed + 200, 500)
        else:
            vcc_print_elements = 100000
    if ho_basis is None:
        ho_basis = max(10, n_modals + 4)
    if ho_basis < n_modals:
        raise ValueError(f"ho_basis ({ho_basis}) must be >= n_modals ({n_modals})")

    lines: list[str] = []
    a = lines.append

    a("#0 MIDAS Input")
    a("")
    a("#1 General")
    a("   #2 IoLevel")
    a(f"      {io_level}")
    a("")
    a("#1 Vib")
    a("")
    a("// Hamiltonian operator")
    a("#2 Operator")
    a("   #3 Name")
    a("      h0")
    a("   #3 OperFile")
    a(f"      {mop_filename}")
    a("   #3 SetInfo")
    a("      type=energy")
    a("")
    a("// Basis input --- HARMONIC OSCILLATOR (not B-splines) ---")
    a("#2 Basis")
    a("   #3 Name")
    a("      basis")
    a("   #3 BasisType")
    a("      HO")
    a("   #3 HoBasis")
    a(f"      {ho_basis}")
    if use_scaling_freqs:
        a("   #3 UseScalingFreqs")
    a("")
    a("// VSCF")
    a("#2 Vscf")
    a("   #3 Name")
    a("      vscf_calc")
    a("   #3 Oper")
    a("      h0")
    a("   #3 Basis")
    a("      basis")
    if occup is not None:
        # Explicit occupation (space-separated) -> one targeted state. Avoids
        # the OccAllFund batch-setup crash at large mode counts.
        a("   #3 Occup")
        a("      " + " ".join(str(int(o)) for o in occup))
    elif ground_state_only:
        # Ground state only -- also avoids the OccAllFund crash (needed at
        # large N even when we just want the ground-state amplitudes).
        a("   #3 OccGroundState")
    else:
        a("   #3 OccGroundState")
        a("   #3 OccAllFund")
    a("")
    a(f"// {vcc_method} calculation")
    a("#2 Vcc")
    a("   #3 Method")
    a(f"      {vcc_method}")
    a("   #3 Name")
    a("      vcc_calc")
    a("   #3 Oper")
    a("      h0")
    a("   #3 Basis")
    a("      basis")
    a("   #3 UseAllVscf")
    a("   #3 ItEqResidThr")
    a(f"      {it_eq_resid_thr}")
    if it_eq_max_it is not None:
        a("   #3 ItEqMaxIt")
        a(f"      {it_eq_max_it}")
    # The VCC block's OWN IoLevel controls how many solution-vector components
    # are printed: n_out = max(5, IoLevel) (MidasCpp src/vcc/Vcc.cc). Set it high
    # so ALL active-space t1/t2 amplitudes appear in the .mout for extraction.
    a("   #3 IoLevel")
    a(f"      {vcc_print_elements}")

    if include_vci:
        a("")
        a("// VCI validation")
        a("#2 Vcc")
        a("   #3 Method")
        a(f"      VCI[{vcc_method[vcc_method.find('[') + 1:vcc_method.find(']')]}]")
        a("   #3 Name")
        a("      vci_calc")
        a("   #3 Oper")
        a("      h0")
        a("   #3 Basis")
        a("      basis")
        a("   #3 ItEqResidThr")
        a("      1.0e-10")
        a("   #3 Rsp")
        a("      #4 EigenVal")
        a(f"         {vci_eigenvals}")
        a("      #4 ItEqResidThr")
        a("         1.0e-10")

    a("")
    a("#1 Analysis")
    a("")
    a("#0 Midas Input End")
    a("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(
        build_midas_input(
            mop_filename="h2o_dimless_qff_numeric.mop",
            n_modals=4,
            use_scaling_freqs=True,
            vcc_method="VCC[2]",
        )
    )
