# Pitfalls

- Keep timing pass/fail separate from 4x speed pass/fail.
- Do not diagnose `bsg_chip` timing as a skip-fanout bug. It passes timing
  with `GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP=300`.
- Do not run `visible/NV_NVDLA_partition_c` or `blind/ariane` with missing
  fanout skip `300` when checking correctness; use `0`.
- Do not run `mempool_group/cluster` with skip `0`; missing clock nets can
  create huge fallback RC work and memory growth.
- Do not re-enable default DMP/route/debug prints in normal timing runs.
- Do not materialize per-slot Liberty threshold arrays such as `slot_vth`,
  `slot_vl`, `slot_vh`, or `slot_slew_derate`.
- Do not change DMP forward branch before reading `08_dmp_fallback_branch.md`.
- Do not use one unordered `clocks.begin()` period for all setup tests.
- Do not ignore `set_clock_uncertainty`; setup and hold semantics differ.
- For ideal-clock designs, use SDC clock transition for register clock pins
  where OpenROAD does.
- If OpenROAD and Xplace differ by ns-level AT, inspect SDC/timing semantics
  before tuning RC.
- If RC graph and TSV agree but WNS/TNS miss, suspect timing semantics.
- Pin-name normalization has caused false bugs; compare normalized names.
- `set_design.tcl` may hardcode `bsg_chip`; do not source blindly.
- Build from `build` and run `make install`; Python loads installed `cpybin`.
