# OpenroadInferCsv.cpp

## Purpose
Parses OpenROAD ground-truth `.infer` CSV files and maps OpenROAD node IDs to
GPUTimer pin IDs by name.

## Main Entry Points
- `GPUTimer::read_opr_gt_infer()`

## Data Ownership
Parsed vectors are local temporaries. `host_pinGT_AT` is owned by `GPUTimer`;
`timing_raw_db.pinGT_AT` receives a Torch tensor copy.

## Invariants
Pin-name normalization, GT AT storage, and CSV field ordering must stay
unchanged.

## CUDA/C++ Boundary Notes
This `.cpp` file performs parsing and Torch tensor construction only. GPU
array mutation is delegated to `InferApply.cu`.

## Acceptance Tests
- Build/install.
- OpenROAD inference CSV smoke if available.
