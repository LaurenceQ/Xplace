# SdcClockConstraints.cpp

## Purpose
SDC handlers for clock creation, clock uncertainty/transition/latency, and
clock propagation state.

## Main Entry Points
- `_read_sdc(CreateClock)`
- `_read_sdc(SetClockUncertainty)`
- `_read_sdc(SetClockTransition)`
- `_read_sdc(SetClockLatency)`
- `_read_sdc(SetPropagatedClock)`
- `_read_sdc(SetIdealNetwork)`

## Data Ownership
Handlers mutate `GTDatabase` clock maps, transition/latency overrides, and
propagated clock markers.

## Invariants
OpenROAD/OpenSTA-compatible ideal/propagated clock semantics and clock
uncertainty handling must remain unchanged.

## CUDA/C++ Boundary Notes
No CUDA runtime calls or kernel launches.

## Acceptance Tests
- Build/install.
- Clock-sensitive designs preserve WNS/TNS alignment.
