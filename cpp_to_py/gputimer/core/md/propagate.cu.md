# propagate.cu

## Purpose
CUDA implementation of standard timing propagation. Power CUDA kernels were
split out so this file stays focused on STA forward/backward propagation.

## Main Entry Points
- `update_timing_cuda(const TimingPropagationModel&)`

## Data Ownership
All device arrays are owned by `GPUTimer` or Torch tensors. This file launches
kernels over borrowed pointers only.

## Invariants
Level traversal order, LUT timing queries, AT/RAT update rules, and DMP-fed RC
delay use remain unchanged.

## CUDA/C++ Boundary Notes
This file owns CUDA runtime synchronization for timing propagation. `.cpp`
callers pass `TimingPropagationModel` and do not launch kernels directly.

## Acceptance Tests
- Build/install after CMake reconfigure.
- Route timing compare smoke on ISPD2025 segment input.
