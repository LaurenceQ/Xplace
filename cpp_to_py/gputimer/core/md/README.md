# GPUTimer Core Index

This directory implements GPU-accelerated timing, RC, endpoint, inference, and
power paths. Per-file sibling `.md` specs are the source of detailed ownership,
invariants, CUDA/C++ boundary notes, and acceptance tests.

## Primary Files

- `GPUTimer.h`, `GPUTimer.cpp`, `GPUTimer.cu`: public timer class, lifecycle,
  persistent CUDA state, and state reset.
- `propagate.cpp`, `propagate.cu`, `propagate_infer.cu`: timing propagation
  wrappers and CUDA kernels. Argument structs live in `timing/`.
- `timing/EndpointSlack.cu`: endpoint slack kernels and endpoint debug dump.
- `timing/PathTrace.cpp`, `timing/PathTrace.cu`: critical path reporting and CUDA path trace kernels.
- `FluteRcTree.cpp`, `ExplicitRcTree.cu`, `rc/StarRc.cu`: host RC orchestration, explicit
  tree CUDA, and star-RC CUDA. RC argument structs live in `rc/`.
- `Power.cpp`, `power/PowerCudaActivity.cu`: host power reporting/input build
  and CUDA activity/component kernels. Power launch structs live in `power/`.
- `infer/InferCsv.cpp`, `infer/OpenroadInferCsv.cpp`, `infer/InferApply.cu`:
  inference CSV parsing and CUDA application.
- `openroad/`: OpenROAD GR/route-segment RC import, geometry, cache, and debug helpers.
- `levelize.cu`, `spef.cpp`, `dump.cpp`, `utils.cuh`:
  supporting timing graph utilities.

## Boundary Rules

- `.cpp` files must not include CUDA runtime headers or launch kernels.
- Kernel launches and CUDA runtime error checks belong in `.cu` files.
- Public Python-visible method names and pybind names stay stable.
- DMP formulas, OpenROAD route-segment RC semantics, SDC semantics, and power
  activity/grouping rules are behavior-preserving constraints.

## Validation

After adding or splitting source files, re-run CMake before building because
`GLOB_RECURSE` source lists are configured-time state:

```bash
cd /research/d7/ascstd/qkduan25/Xplace/build
source ~/.bashrc
conda activate gnn
cmake ..
make -j8
make install
```
