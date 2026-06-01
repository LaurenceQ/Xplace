# SdcUtils.cpp

## Purpose
Owns shared SDC parser helper implementations used by the split constraint readers.

## Main Entry Points
- `is_transition_defined_cpu`
- `warn_missing_sdc_object`
- `add_pin_name_target_variants`

## Data Ownership
The helpers do not own database state. They inspect caller-owned timing arcs, strings, and target sets.

## Invariants
SDC object warning behavior remains gated by `GPUTIMER_VERBOSE_SDC_WARNINGS`; pin-name variants preserve both slash and colon spellings.

## CUDA/C++ Boundary Notes
This is a standard C++ translation unit and must not include CUDA runtime headers or launch kernels.

## Acceptance Tests
Rebuild GPUTimer and run ISPD2025 route/timing/power compare cases after SDC parser changes.
