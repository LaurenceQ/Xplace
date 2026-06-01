# PowerCudaActivity.cu

## Purpose
CUDA implementation for power activity propagation plus switching, internal,
and leakage component kernels.

## Main Entry Points
- `run_power_activity_cuda_launcher(const PowerActivityCudaModel&)`
- `run_power_internal_denom_chunk_cuda_launcher()`
- `run_power_internal_contrib_chunk_cuda_launcher()`
- `run_power_leakage_rows_chunk_cuda_launcher()`
- `run_power_leakage_summary_chunk_cuda_launcher()`

## Data Ownership
Temporary CUDA buffers are allocated and freed inside the launchers. Input and
output arrays are borrowed through `PowerCudaLaunch.h` argument structs.

## Invariants
Do not change power expression BDD semantics, sequential feedback iteration,
clock activity seeding, DMP load formulas, or OpenROAD-aligned component
equations.

## CUDA/C++ Boundary Notes
This file owns CUDA runtime calls and kernel launches for power. `.cpp` files
must call these wrappers rather than launching kernels directly.

## Acceptance Tests
- `report_power_total_cuda()` returns unchanged components.
- Full ISPD2025 compare must pass component and group/component 1% gates.

## Line Target Exception
This file exceeds the soft 800-line target because it still contains the
coupled activity frontier, BDD expression, and component kernels moved out of
`propagate.cu` as one behavior-preserving block. A later split should separate
activity traversal from component kernels after this launcher-struct boundary is
validated.
