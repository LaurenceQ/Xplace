# StarRc.cu

## Purpose
CUDA implementation for the simple star-RC timing update used by
`GPUTimer::update_rc_timing()`.

## Main Entry Points
- `update_rc_timing_cuda(const RcStarModel&)`

## Data Ownership
All arrays are borrowed from `GPUTimer`; this file does not allocate persistent
device memory.

## Invariants
Clock-net wirelength suppression, unit conversion, load accumulation, root
delay, root resistance, and impulse formulas are preserved.

## CUDA/C++ Boundary Notes
The `.cpp` caller passes `RcStarModel`; this `.cu` file launches the kernel.

## Acceptance Tests
- Build/install.
- Timing smoke with star-RC update path if used by the caller.
