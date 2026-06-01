# read_infer.cu

## Purpose
Compatibility stub for the inference loader split.

## Main Entry Points
No functions are defined here. The former contents now live under
`core/infer/`.

## Data Ownership
None.

## Invariants
Public `GPUTimer::read_infer()`, `GPUTimer::read_opr_gt_infer()`, and
`GPUTimer::apply_infer_data()` APIs remain unchanged.

## CUDA/C++ Boundary Notes
CUDA copy/update work is in `infer/InferApply.cu`; CSV parsing is in `.cpp`
files.

## Acceptance Tests
- Build/install.
- Inference CSV smoke if fixtures are available.
