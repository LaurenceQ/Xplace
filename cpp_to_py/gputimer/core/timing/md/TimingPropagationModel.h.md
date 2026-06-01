# TimingPropagationModel.h

## Purpose
Defines compact host-to-CUDA argument bundles for timing propagation launchers.

## Main Entry Points
- `TimingPropagationModel`: normal STA propagation inputs.
- `InferTimingModel`: ML/OpenROAD inference timing propagation inputs.
- `update_timing_cuda()` and `propagate_infer_timing_impl()` declarations.

## Data Ownership
All pointers are borrowed from `GPUTimer` device arrays or Torch tensors. The
structs do not allocate, free, or retain ownership.

## Invariants
`level_list_end_cpu` must outlive the launcher call and must describe the same
device `level_list`. Pointer fields preserve the old launcher semantics.

## CUDA/C++ Boundary Notes
This header has no CUDA runtime calls and can be included from `.cpp` files.
Kernel launches remain in `.cu` translation units.

## Acceptance Tests
- Build `gt` and installed `gputimer`.
- Run route timing/power compare smoke after any propagation changes.
