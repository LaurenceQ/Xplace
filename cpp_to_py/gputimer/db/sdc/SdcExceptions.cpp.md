# SdcExceptions.cpp

## Purpose
SDC handlers for case analysis and false-path constraints.

## Main Entry Points
- `_read_sdc(SetCaseAnalysis)`
- `_read_sdc(SetFalsePath)`

## Data Ownership
Handlers mutate `GTDatabase` case values and constraint-arc disable metadata.

## Invariants
Case-analysis activity/timing effects and false-path matching by clocks and
pins must stay unchanged.

## CUDA/C++ Boundary Notes
No CUDA runtime calls or kernel launches.

## Acceptance Tests
- Build/install.
- SDC exception-heavy timing cases stay aligned with OpenROAD references.
