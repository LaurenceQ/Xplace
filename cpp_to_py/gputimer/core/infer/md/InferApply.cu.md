# InferApply.cu

## Purpose
Applies parsed inference slew and delay data to GPU timing arrays.

## Main Entry Points
- `GPUTimer::apply_infer_data()`

## Data Ownership
Host scratch arrays are allocated and freed in this function. `GPUTimer` owns
the persistent GPU timing arrays.

## Invariants
Net-delay and cell-delay update rules, units, and arc lookup behavior must stay
unchanged.

## CUDA/C++ Boundary Notes
This file performs CUDA synchronization and host/device copies, so it remains a
`.cu` translation unit.

## Acceptance Tests
- Build/install.
- Inference timing smoke if fixtures are available.
