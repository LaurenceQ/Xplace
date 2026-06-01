# SdcTimingConstraints.cpp

## Purpose
SDC handlers for units, input/output timing constraints, loads, driving cells,
and max transition.

## Main Entry Points
- `_read_sdc(SetUnits)`
- `_read_sdc(SetInputDelay)`
- `_read_sdc(SetInputTransition)`
- `_read_sdc(SetDrivingCell)`
- `_read_sdc(SetOutputDelay)`
- `_read_sdc(SetLoad)`
- `_read_sdc(SetMaxTransition)`

## Data Ownership
Handlers mutate `GTDatabase` timing tensors and metadata in place.

## Invariants
SDC unit conversion, min/max/rise/fall masks, driving-cell arc selection, and
output delay uncertainty adjustments must stay unchanged.

## CUDA/C++ Boundary Notes
No CUDA runtime calls or kernel launches.

## Acceptance Tests
- Build/install.
- SDC timing smoke against OpenROAD no-CRPR references.
