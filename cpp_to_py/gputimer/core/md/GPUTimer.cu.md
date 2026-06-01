# GPUTimer.cu

## Purpose
CUDA-backed GPUTimer initialization, teardown, persistent device state reset,
and power LUT probe support.

## Main Entry Points
- `GPUTimer::initialize()`
- `GPUTimer::~GPUTimer()`
- `GPUTimer::update_states()`
- `GPUTimer::report_power_internal_lut_cuda_probe()`

## Data Ownership
This file owns allocation/free of persistent CUDA arrays stored on `GPUTimer`
and GPU LUT allocator mirrors.

## Invariants
Initialization order, DMP forward schedule release, and installed Python module
state must stay unchanged.

## CUDA/C++ Boundary Notes
CUDA runtime usage is isolated in this `.cu` file and sibling CUDA files. Host
`.cpp` wrappers must not launch kernels.

## Acceptance Tests
- Build/install.
- `python run_timer.py --designName <design>` initializes and updates states
  without stale installed `cpybin`.
