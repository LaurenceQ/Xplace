# propagate.cpp

## Purpose
C++ wrapper that implements GPU method declarations for standard (non-ML) timing propagation.

## Methods

### update_timing() [Public GPUTimer method]
Standard static timing analysis using STA library lookups.

**Flow**:
1. Forward pass: Propagate arrival times (AT) from primary inputs
2. Backward pass: Propagate required arrival times (RAT) from primary outputs
3. Compute slack = RAT - AT at all pins

**Calls**: `update_timing_cuda()` implemented in propagate.cu

## Structure

- **Method in GPUTimer**: Wrapper that calls CUDA implementation
- **Implementation**: `update_timing_cuda()` in propagate.cu with forward/backward kernels

## Relationship to Inference

This file provides the **standard STA path**. For ML inference, use:
- `propagate_infer_timing()` instead (pre-loads arcDelay with ML predictions)

## Comparison

| Aspect | Standard | Inference |
|--------|----------|-----------|
| File | propagate.cpp + propagate.cu | propagate.cpp + propagate_infer.cu |
| arcDelay | Computed from LUT at runtime | Pre-loaded from .infer CSV |
| Method | `update_timing()` | `propagate_infer_timing()` |
| Slew | From LUT | From ML prediction (read_infer) |
