# PowerCudaLaunch.h

## Purpose
Collects the long CUDA power launcher parameter lists into named device-view
and configuration structs.

## Main Entry Points
- `PowerActivityCudaModel` for activity, switching, internal, and leakage
  computation in the default CUDA power path.
- Chunk launch args for internal and leakage row processing.

## Data Ownership
All fields are borrowed pointers into device tensors, GPUTimer-owned arrays, or
host vectors. The launchers do not take ownership.

## Invariants
Power grouping, Liberty expression evaluation, sequential feedback activity,
DMP load usage, and OpenROAD-aligned component formulas must stay unchanged.

## CUDA/C++ Boundary Notes
This header is safe for `.cpp` call sites. CUDA runtime calls and kernel
launches remain in `.cu` files.

## Acceptance Tests
- `report_power_total_cuda()` component totals stay within 1% of OpenROAD.
- Group/component compare rows pass with `openroad_log_report_power`.
