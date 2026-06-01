# GTDatabase.cpp

## Purpose
GTDatabase timing graph, SDC, or database ownership implementation.

## Main Entry Points
Methods or wrapper functions declared by the corresponding GPUTimer/database interface.

## Data Ownership
This file does not change ownership of timer/database memory. Device pointers and host containers are owned by the caller or the enclosing GPUTimer/GTDatabase object unless explicitly allocated and freed in the same implementation path.

## Invariants
- Preserve DMP formulas, OpenROAD route-segment RC semantics, SDC semantics, and power activity rules.
- Keep public Python and pybind-visible names unchanged.
- Keep touched source files at or below the 800-line soft target; this file has no exception.

## CUDA/C++ Boundary
This C++ file must not include CUDA runtime headers or launch kernels; CUDA work is called through wrappers.

## Acceptance Tests
- Reconfigure and build from `build` with `cmake ..`, `make -j8`, and `make install`.
- Run at least the visible `ariane` route/power/timing compare smoke after CUDA or timing changes.
