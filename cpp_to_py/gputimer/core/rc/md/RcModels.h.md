# RcModels.h

## Purpose
Names RC CUDA launcher inputs so star-RC and explicit-tree-RC code can be split
without reintroducing long unstructured argument lists.

## Main Entry Points
- `RcStarModel` for the simple star RC update path.
- `RcGraphDeviceView` and `RcPropagateScratch` for explicit RC trees.

## Data Ownership
Pointers are borrowed from `GPUTimer`, Torch tensors, or temporary CUDA buffers
owned by the caller.

## Invariants
OpenROAD route-segment RC semantics and DMP timing formulas are not changed by
this header.

## CUDA/C++ Boundary Notes
The header has no CUDA runtime dependency. Kernel launches remain in `.cu`
files.

## Acceptance Tests
- Route timing compare smoke remains within the established 1% timing gate.
