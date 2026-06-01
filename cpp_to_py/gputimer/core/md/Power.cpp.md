# Power.cpp

## Purpose
Host-side GPUTimer power implementation. It builds Liberty-derived power inputs,
launches CUDA activity/component calculations, and exposes public report APIs.

## Main Entry Points
- `report_power_group_codes()`
- `report_power_activity_cpu()`
- `compute_power_activity_cuda()`
- `report_power_total_cuda()` and related report helpers

## Data Ownership
Torch tensors own temporary host/device buffers. `GPUTimer` owns persistent
timing and power LUT device state. CUDA launch argument structs borrow these
pointers only for the duration of a call.

## Invariants
OpenROAD group priority is macro, pad, clock, sequential, combinational.
Activity rules, DMP load use, Liberty power expression handling, and DEF-order
summary behavior must stay unchanged.

## CUDA/C++ Boundary Notes
This `.cpp` file may call CUDA launch wrappers declared in
`power/PowerCudaModel.h`, but it does not include CUDA runtime headers or
launch kernels directly.

## Acceptance Tests
- `report_power_total_cuda()` pybind API remains unchanged.
- 12-case route power/timing compare passes all component and group 1% gates.

## Line Target Exception
This file currently remains above the soft 800-line target because
`compute_power_activity_cuda()` still owns the coupled host input-build and
report orchestration path. The first pass moved the CUDA launcher boundary to
structs; a follow-up split should move inventory, grouping, CPU activity, CUDA
input build, and report wrappers into sibling files without changing APIs.
