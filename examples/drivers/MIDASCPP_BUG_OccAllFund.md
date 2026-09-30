# MidasCpp bug report: `#3 OccAllFund` crashes (`std::out_of_range`) at large mode counts

**Version:** MidasCpp 2025.10.0 (release build, macOS arm64)
**Severity:** crash (uncaught C++ exception, `exit 255`)
**Support:** midascpp.support@chem.au.dk / https://source.coderefinery.org/midascpp/midascpp/-/issues

## Summary

A VSCF calculation using `#3 OccAllFund` (compute all fundamentals) terminates
with an uncaught `std::out_of_range` from `basic_string::substr` once the system
has roughly **≥ 7 modes**. The same input runs fine at ≤ 6 modes. The
ground-state path (`#3 OccGroundState` alone) is unaffected at any size.

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  basic_string::substr: __pos (which is 18446744073709551615) > this->size() (which is 24)
```

`__pos = 18446744073709551615` is `std::string::npos`, i.e. a `find`/`find_last_of`
that returned `npos` is being passed to `substr` without a guard. The crash
occurs during setup, **before** the basis line (`Basis-Type: PROD-ALLHO`) is
printed, so it is in the all-fundamentals batch setup / state-labeling, not in
the SCF iteration itself.

## Empirical bracket (glycine QFF, `.mop` with 27 modes, subsets of the low modes)

| modes | `OccGroundState + OccAllFund` |
|------:|-------------------------------|
| 3 (H2O)   | OK |
| 4         | OK |
| 6         | OK (unrelated modal-size error only if HoBasis < modals) |
| 10        | **CRASH** `std::out_of_range` |
| 15        | **CRASH** |
| 20        | **CRASH** |
| 27        | **CRASH** |

Not caused by the operator filename: tested with a numbered file
(`glycine_1.mop`) — still crashes. Not caused by iteration count or the lib
path (those are separate, benign setup issues).

## Minimal reproducer

1. Take any `.mop` with ≥ 10 modes (e.g. a 15-mode subset of a glycine QFF).
2. Input:

```
#0 MIDAS Input
#1 General
   #2 IoLevel
      5
#1 Vib
#2 Operator
   #3 Name
      h0
   #3 OperFile
      subset.mop
   #3 SetInfo
      type=energy
#2 Basis
   #3 Name
      basis
   #3 BasisType
      HO
   #3 HoBasis
      6
   #3 UseScalingFreqs
#2 Vscf
   #3 Name
      vscf_calc
   #3 Oper
      h0
   #3 Basis
      basis
   #3 OccGroundState
   #3 OccAllFund          # <-- remove this line and it runs fine
#2 Vcc
   #3 Method
      VCC[2]
   #3 Name
      vcc_calc
   #3 Oper
      h0
   #3 Basis
      basis
   #3 UseAllVscf
#1 Analysis
#0 Midas Input End
```

3. `midascpp.x input.inp` → `std::out_of_range` and `exit 255` for ≥ ~7 modes;
   removing `#3 OccAllFund` (ground state only) runs to completion at any size.

## Likely location

`basic_string::substr` with a `find`-returned `npos` on a length-24 string,
reached on the all-fundamentals setup path. Candidate: the state/label or
analysis-filename construction that runs once per fundamental
(e.g. `Vscf::...` naming code that does
`name.find_last_of(...)` then `substr(...)` without checking for `npos`), which
only exercises the failing branch when the number of generated fundamental
states is large. A guard (`if (pos == npos) pos = 0;` or skip) should fix it.

## Workaround used in this pipeline (no MidasCpp rebuild)

Run one targeted calculation per fundamental with an explicit occupation vector
instead of the batch keyword:

```
#2 Vscf
   ...
   #3 Occup
      1 0 0 0 0 0 0 0 0 0 0 0 0 0 0    # mode 0 excited to modal 1; rest ground
```

This never enters the crashing batch code and works at 15+ modes. Implemented in
`fundamentals.py` (`run_all_fundamentals`); excitation energies are
`omega_i = E(fundamental_i) - E(ground)`.
