# utils.part.cu

## Purpose
GPUTimer host/CUDA support code.

## Main Entry Points
File-local helpers/kernels included by the sibling wrapper translation unit.

- This implementation slice is included by a wrapper `.cpp` or `.cu` file and is excluded from standalone CMake compilation.

## Data Ownership
This file does not change ownership of timer/database memory. Device pointers and host containers are owned by the caller or the enclosing GPUTimer/GTDatabase object unless explicitly allocated and freed in the same implementation path.

## Invariants
- Preserve DMP formulas, OpenROAD route-segment RC semantics, SDC semantics, and power activity rules.
- Keep public Python and pybind-visible names unchanged.
- Keep touched source files at or below the 800-line soft target; this file has no exception.

## CUDA/C++ Boundary
CUDA runtime calls, kernel launches, and device helpers remain in CUDA implementation files.

## Acceptance Tests
- Reconfigure and build from `build` with `cmake ..`, `make -j8`, and `make install`.
- Run at least the visible `ariane` route/power/timing compare smoke after CUDA or timing changes.
