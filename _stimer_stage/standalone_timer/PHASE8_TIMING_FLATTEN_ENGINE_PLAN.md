# Phase 8 Timing Flatten And First Evaluation Plan

## Goal

Convert the parser-owned timing objects into compact arrays and run the first
real cell-delay/slew evaluation path. This phase is still single-level timing:
it evaluates timing graph arcs from Liberty LUTs. Full topological propagation,
RC load delay, SDC clock endpoints, and DMP net delay are later phases.

## Scope

Implement:

- `TimingFlatDB`
  - one compact row per instantiated timing graph arc;
  - compact LUT descriptors;
  - contiguous vectors for indices and values;
  - per-arc input slew/load probe values;
  - per-arc delay/output slew/arrival outputs.
- CPU evaluator
  - bilinear interpolation from compact LUT arrays;
  - max arrival and max slew summary;
  - deterministic fallback when a LUT is missing.
- CUDA evaluator
  - plain C++ wrapper declared in headers;
  - all CUDA runtime calls, kernel launch, sync, and first-failure checks in
    `.cu`;
  - one thread per timing arc for this phase;
  - event timing around the kernel and explicit `cudaDeviceSynchronize`.

## Data Rules

- Do not put CUDA runtime headers or kernel launches in `.cpp` files.
- Do not pass long parameter lists into kernels; use a compact launch struct.
- Keep vectors in C++ DB objects for now, but kernel arguments must be pointer
  structs built in the `.cu` wrapper.
- Use Liberty LUT variable metadata when selecting interpolation axes:
  - input/related transition variables use the per-arc input slew probe;
  - output capacitance variables use the per-arc load capacitance probe;
  - unknown variables fall back to the first index value.

## Current Probe Values

Until SDC and RC effective capacitance are wired into propagation:

- per-arc input slew probe comes from the first available LUT transition index;
- per-arc load cap probe comes from the first available output capacitance
  index;
- arc arrival is `delay + 0` for this single-level seed.

These placeholders are intentionally stored in the flat DB so replacing them
with real propagated slews and RC/DMP loads later does not change the CUDA
kernel signature.

## Validation

Required checks:

- build in conda `gnn`;
- `ctest --output-on-failure`;
- minimal DEF/SPEF flow with CPU and CUDA;
- CPU and CUDA max arrival/max slew agree on the minimal case;
- no `libtorch`/`libpython` linkage;
- no CUDA runtime calls in `.cpp`/`.h`.

## Done For This Phase

This phase is done when `stimer_run` prints compact timing table counts and
nonzero evaluated delay/slew/arrival from parsed Liberty LUTs, with CUDA timing
measured after an explicit device sync.

## Current Status

Implemented:

- `TimingFlatDB` compact arc/LUT/index/value arrays;
- CPU bilinear LUT evaluator;
- CUDA one-thread-per-arc evaluator with struct launch arguments;
- explicit `cudaDeviceSynchronize` before event elapsed time read;
- CPU/CUDA result summaries in `stimer_run`.

Verified on the minimal Liberty/DEF/SPEF case:

- `flat_arcs=2`, `flat_luts=8`, `flat_values=32`;
- CPU `max_delay=0.111`, `max_output_slew=0.131`, `max_arrival=0.111`;
- CUDA matches CPU with `cpu_cuda_max_abs_error=0`;
- CTest passes;
- no Torch/Python linkage;
- no CUDA runtime calls in `.cpp`/`.h`.
