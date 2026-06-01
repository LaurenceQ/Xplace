# OpenROAD RC Sources

## Purpose
Builds GPUTimer host RC graphs from OpenROAD GR TSV and saved global-route
segment inputs, plus route-segment RC debug/compare helpers.

## Source Layout
- `openroad/OpenroadRcInternal.h`: shared internal structs and helper declarations.
- `openroad/OpenroadRcCache.cpp`: route-segment graph cache hashing and binary I/O.
- `openroad/OpenroadRcParse.cpp`: name aliases and OpenROAD row/token parsing.
- `openroad/OpenroadRcGeometry.cpp`: grid, layer RC, orientation, and route-point geometry.
- `openroad/OpenroadRcGraphUtil.cpp`: local RC graph node/edge helpers and tree cleanup.
- `openroad/OpenroadRcPin.cpp`: timer pin resolution and OpenROAD pin route location.
- `openroad/OpenroadGrRcBuilder.cpp`: OpenROAD GR RC TSV graph builder.
- `openroad/OpenroadRouteSegmentsBuilder.cpp`: saved route-segment RC graph builder.
- `openroad/OpenroadRcDebug.cpp`: debug dump and GR-vs-route comparison helpers.

## Main Entry Points
- `GPUTimer::build_openroad_gr_rc()`
- `GPUTimer::build_openroad_route_segments_rc()`
- `GPUTimer::debug_dump_openroad_gr_rc_net()`
- `GPUTimer::debug_dump_openroad_route_segments_rc_net()`
- `GPUTimer::debug_compare_openroad_route_segments_rc()`

## Data Ownership
Returns `HostRcGraph` values by ownership transfer. Uses `GTDatabase` and rawdb
geometry as borrowed references.

## Invariants
OpenROAD saved route-segment semantics, layer/geometry handling, pin attachment
rules, and debug TSV comparisons must stay unchanged.

## CUDA/C++ Boundary Notes
Host-only parser/build code. No CUDA runtime headers or kernel launches belong
in these files.

## Acceptance Tests
- Route-segment timing smoke against OpenROAD no-CRPR references.

