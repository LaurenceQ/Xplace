# InferCsv.cpp

## Purpose
Parses TimingPredict `.infer` CSV files where node IDs map directly to GPUTimer
pin IDs.

## Main Entry Points
- `GPUTimer::read_infer()`

## Data Ownership
Parsed vectors are local temporaries and are passed to `apply_infer_data()`.

## Invariants
CSV field ordering, nan/parse skip behavior, and nanosecond-to-internal-unit
conversion must remain unchanged.

## CUDA/C++ Boundary Notes
This `.cpp` file performs no CUDA runtime calls. GPU updates are delegated to
`InferApply.cu`.

## Acceptance Tests
- Build/install.
- TimingPredict inference CSV smoke if available.
