# GTDatabase_sdc.cpp

## Purpose
Top-level SDC read flow and post-processing after command dispatch.

## Main Entry Points
- `GTDatabase::preparePinNameMapForSdc()`
- `GTDatabase::readSdc()`

## Data Ownership
Mutates `GTDatabase` timing, clock, uncertainty, and pin metadata owned by the
database. Handler methods live in `db/sdc/`.

## Invariants
SDC command dispatch order and final clock/test tensor materialization must
remain unchanged.

## CUDA/C++ Boundary Notes
No CUDA runtime calls. This file may create Torch tensors but does not launch
kernels.

## Acceptance Tests
- Build/install.
- ISPD2025 route timing smoke remains aligned with OpenROAD no-CRPR timing.
