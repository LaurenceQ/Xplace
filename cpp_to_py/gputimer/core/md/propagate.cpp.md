# propagate.cpp

## Purpose
Public `GPUTimer` timing wrapper layer. It translates class members into named
CUDA launch argument structs.

## Main Entry Points
- `GPUTimer::update_timing()`
- `GPUTimer::propagate_infer_timing()`

## Data Ownership
No arrays are owned here. All pointers are borrowed from `GPUTimer` state and
passed through to CUDA launchers.

## Invariants
Public Python API names and timing behavior stay unchanged. Argument bundling
must not alter level order, DMP timing formulas, or SDC semantics.

## CUDA/C++ Boundary Notes
This `.cpp` file declares no kernels and includes no CUDA runtime headers.
Launches remain in `.cu` files.

## Acceptance Tests
- Build and install `gputimer`.
- Run route timing compare smoke for at least one visible and one blind design.
