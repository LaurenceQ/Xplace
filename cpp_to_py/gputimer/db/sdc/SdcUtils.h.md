# SdcUtils.h

## Purpose
Shared helper functions for SDC parser handlers.

## Main Entry Points
- `is_transition_defined_cpu()`
- `warn_missing_sdc_object()`
- `add_pin_name_target_variants()`

## Data Ownership
No ownership. Helpers operate on borrowed strings, timing arcs, and target sets.

## Invariants
Pin-name variant generation and verbose warning behavior must match the former
`GTDatabase_sdc.cpp` helpers.

## CUDA/C++ Boundary Notes
No CUDA runtime dependency.

## Acceptance Tests
- Build/install.
- SDC-heavy timing smoke remains aligned with OpenROAD semantics.
