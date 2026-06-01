# ExplicitRcTree.cu

## Purpose
CUDA implementation for explicit RC tree flattening, resistance/capacitance
calculation, and propagation back into timing arrays.

## Main Entry Points
- `flatten_rc_tree()`
- `propagate_rc_tree()`
- `calc_res_cap()`

## Data Ownership
Temporary device buffers allocated in these launchers are freed before return.
Timing arrays are borrowed from `GPUTimer`/Torch.

## Invariants
Explicit-tree Elmore delay and cap accumulation semantics remain unchanged.
OpenROAD segment-derived RC graphs must preserve pin-cap handling.

## CUDA/C++ Boundary Notes
This file owns CUDA runtime calls for explicit RC trees. Star-RC update moved
to `rc/StarRc.cu`.

## Acceptance Tests
- Build/install.
- Route-segment timing compare smoke remains within the 1% timing gate.
