# gputiming_host.cu

## Purpose
Implements host-side GPU LUT allocator setup, device memory copies, and cleanup for timing and power LUTs.

## Main Entry Points
- `GPULutAllocator::AllocateBatch`
- `GPULutAllocator::CopyToGPU`
- `GPUPowerLutAllocator::AllocateBatch`
- `GPUPowerLutAllocator::CopyToGPU`

## Data Ownership
Allocator instances own their host arrays and CUDA buffers after `AllocateBatch`/`CopyToGPU`; destructors release those buffers.

## Invariants
LUT order, timing sense encoding, latch clock arc tagging, and power LUT template variable mapping match the previous inline implementation.

## CUDA/C++ Boundary Notes
This is a CUDA translation unit because it owns CUDA runtime allocation and copy calls. Standard `.cpp` files must not include CUDA runtime usage.

## Acceptance Tests
Rebuild and run route/timing/power compare smoke tests after LUT allocator changes.
