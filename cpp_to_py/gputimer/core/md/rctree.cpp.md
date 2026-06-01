# FluteRcTree.cpp

## Purpose
Host-side RC timing update orchestration, including star RC and Flute explicit
RC tree construction.

## Main Entry Points
- `GPUTimer::update_rc_timing()`
- `GPUTimer::update_rc_timing_flute()`
- `FluteRCTree()`

## Data Ownership
Torch/GPUTimer own timing arrays. Host vectors built for Flute are temporary
and copied into CUDA launchers.

## Invariants
DMP/OpenROAD timing semantics and RC unit conversions must not change. The
`RcStarModel` bundle is only a launcher-shape refactor.

## CUDA/C++ Boundary Notes
This `.cpp` file builds argument structs and host RC graph data. CUDA runtime
usage and kernel launches remain in `.cu` files.

## Acceptance Tests
- Build/install.
- Route timing smoke remains within 1% WNS/TNS gates.

## Line Target Exception
This file is slightly above the soft 800-line target because it still contains
the host Flute topology construction path. CUDA star and explicit-tree work was
split first; a later host-only pass should move Flute topology helpers out.
