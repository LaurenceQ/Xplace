# propagate_infer.cu

## Purpose
CUDA timing propagation for ML/OpenROAD inference delay data already loaded into
`arcDelay`.

## Main Entry Points
- `propagate_infer_timing_impl(const InferTimingModel&)`

## Data Ownership
All arrays are borrowed from `GPUTimer`; no persistent allocations are owned by
this file.

## Invariants
ML net-delay validity, ideal-clock seeding, constraint RAT setup, and backward
RAT propagation semantics remain unchanged.

## CUDA/C++ Boundary Notes
The host wrapper in `propagate.cpp` builds `InferTimingModel`; this `.cu` file
owns the kernel launches and CUDA synchronization.

## Acceptance Tests
- Build/install.
- Run inference timing smoke if inference CSV fixtures are available.
