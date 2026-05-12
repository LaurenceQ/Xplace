# OpenROAD GR RC Alignment Notes

## Current Sky130 RAT/Slack Target

The current implementation target is sky130 no-CRPR RAT/slack alignment across
the available cases. Use the no-CRPR OpenROAD CSV directory as the primary
ground truth while GPUTimer does not implement CRPR/CPPR:

```text
/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130_crpr_off
```

Use OpenROAD CSV `required_*_ns` fields as direct RAT ground truth. Default
CRPR-on CSV `slack_*_ns` is CRPR/CPPR-aware OpenSTA path slack; compare it
directly only when CRPR/CPPR implementation is in scope. Relevant CSV fields:

- `required_default_min_rise_ns`
- `required_default_min_fall_ns`
- `required_default_max_rise_ns`
- `required_default_max_fall_ns`
- `slack_default_min_rise_ns`
- `slack_default_min_fall_ns`
- `slack_default_max_rise_ns`
- `slack_default_max_fall_ns`

Do not claim sky130 is aligned from AT/slew-only checks. Report-level WNS/TNS
and pin-level RAT/slack still need endpoint-level checks, and the expected
reference is no-CRPR OpenSTA unless CRPR/CPPR is explicitly being implemented.

### Current Full Sky130 Result

Validated on 2026-05-12 against the 21 stored sky130bench no-CRPR pin CSVs.

- Max AT diff: `0.0008621186035142614 ns` at `des`
- Max slew diff: `0.0015450762886810576 ns` at `aes192`
- Max RAT diff: `0.0005569473413089554 ns` at `aes192`
- Max endpoint RAT diff: `0.0004482284521500901 ns` at `aes256`
- Max clock-pin RAT diff: `0.0003824261254887773 ns` at `aes256`
- Max slack diff: `0.0011262941735843413 ns` at `aes256`

The same DMP run against the default CRPR-on CSVs has the same AT/slew maxima
but RAT/slack differences up to about `9.17 ns`. This is the expected
OpenSTA CRPR/CPPR/tag required semantic gap, not a current RC/arrival/slew
alignment issue.

Known CRPR-aware slack gap from a current `blabla` SPEF run:

- Xplace SPEF: `wns_early=-6.301 ns`, `tns_early=-5955.472 ns`, `wns_late=18.846 ns`, `tns_late=0`
- OpenROAD CSV endpoints for `blabla`:
  - min rise: `WNS=-5.49665594 ns`, `TNS=-4458.33498188 ns`
  - min fall: `WNS=-5.53634453 ns`, `TNS=-4392.72812385 ns`
  - max rise/fall slacks are positive in the CSV

This no longer implies an open DMP RC/forward mismatch by itself: later checks showed AT, slew, and endpoint RAT are aligned, while CSV `slack_*_ns` includes OpenSTA CRPR/CPPR that Xplace does not implement.

### Active Debug Session

Historical task: use the available comparison tooling and code instrumentation
to close the sky130 RAT mismatch, starting from `blabla` SPEF/CSV. This is now
closed for the no-CRPR target across the stored 21-case sky130bench set. Keep
OpenROAD CSV `required_*_ns` as the direct RAT reference. Treat default CSV
`slack_*_ns` as CRPR/CPPR-aware path slack; compare no-CRPR WNS/TNS by using
the no-CRPR CSVs unless CRPR implementation becomes the task.

Priority checks before code changes:

- Confirm Xplace reports are from the installed extension after `make install`.
- Normalize endpoint names using only the final `/` -> `:` conversion plus escaped bracket cleanup before declaring pins missing.
- Split mismatch into AT, RAT, and slack components per attr; if AT matches but RAT does not, inspect constraint backward propagation, clock semantics, and SDC parsing rather than RC.
- Preserve the already verified ariane GR RC scalar-resistance semantics; do not reintroduce per-attr GR edge resistance.

Confirmed during the `blabla` debug pass:

- Xplace DMP SPEF AT matches OpenROAD CSV within about `3.4e-5 ns` max, and slew within about `1.4e-4 ns` max.
- Endpoint RAT from `required_default_*_ns` matches Xplace endpoint RAT within about `1.6e-5 ns`.
- The remaining endpoint slack mismatch is explained by OpenSTA CRPR/CPPR semantics: OpenROAD CSV `slack_default_*_ns` comes from `Sta::slack()` / `PathEndClkConstrained::slack()`, whose `requiredTime()` includes `checkCrpr()`. Xplace currently reports simple no-CRPR slack from `AT/RAT`.
- For `blabla`, `OpenROAD slack - simple(AT,RAT) slack` on endpoints reaches about `5.15 ns` early and `5.67 ns` late, matching the observed slack/WNS gap pattern.
- Do not chase this mismatch in Xplace timing unless CRPR/CPPR implementation becomes an explicit task. For no-CRPR alignment, compare OpenROAD `required_*_ns` and recompute slack from `arrival/required` instead of using CSV `slack_*_ns`.

## Verified Result

On the ariane GR RC comparison after the latest fixes:

- OpenROAD max WNS/TNS: `-0.51014727 / -1446.95485491 ns`
- Xplace GR-DMP max WNS/TNS: `-0.510147557 / -1446.95491513 ns`
- WNS delta: about `0.00000029 ns`
- TNS delta: about `0.000060 ns`
- Largest finite endpoint slack delta observed: about `0.0000021 ns`

Representative critical endpoint:

- `issue_stage_i/i_scoreboard/mem_q_reg[3]_sbe_ex_tval[35]:D`
- OpenROAD max fall slack: `-0.51014727 ns`
- Xplace max fall slack: `-0.510147572 ns`
- Xplace max fall AT/RAT/slew: `1.768820286 / 1.258672714 / 0.010567620 ns`

RAM bus endpoint that previously had missing RAT is now finite and aligned:

- `i_cache_subsystem/i_nbdcache/sram_block[7].data_sram/macro_mem[5].i_ram:addr_in[4]`
- OpenROAD max rise/fall slacks: about `-0.0298801 / -0.00528244 ns`
- Xplace corresponding slacks: about `-0.029880285 / -0.005282760 ns`

## GR RC Semantics

- OpenROAD GR RC dump is treated as exactly one OpenSTA scene in the DMP GR flow.
- GR wire resistance is scalar per edge. Do not invent per-corner or per-attr edge resistance.
- Node wire cap from the OpenROAD dump is copied to all DMP attrs.
- Pin cap remains from the Liberty/DMP pin-cap path, where min/max/rise/fall differences can be per attr.
- `edge_res_per_attr`, `num_edges * NUM_ATTR` edge R, and guessed GR corner mapping were removed and should not be reintroduced.

## Key Fixes That Made Alignment Work

- Liberty conditional `when` timing arcs must not be collapsed by `encode_arc()`. OpenSTA keeps multiple conditional arcs as alternatives; Xplace now pushes all `timing_arcs_` into propagation candidates.
- Liberty `bus(...)` groups in fakeram libs must be expanded to bit pins such as `addr_in[4]`, otherwise RAM bus setup/hold tests never enter the timing graph.
- Liberty LUT axes and values must be scaled by each library's units. This matters for fakeram `capacitive_load_unit (1,pf)` versus Nangate fF-based assumptions.
- DMP `rd < 1e-2` threshold is in ohms in OpenROAD semantics and must be converted to internal resistance units via `res_unit`.
- Timing threshold/derate data should come from Liberty and be propagated through the DMP data path. Driver timing output/slew thresholds and load pin input thresholds both matter.

## Useful Entry Points

- Xplace Python GR entry: `/research/d7/ascstd/qkduan25/Xplace/src/core/timing_opt.py`, method `update_timing_dmp_gr`.
- GPUTimer pybind entry: `/research/d7/ascstd/qkduan25/gputimer_merged/PyBindCppMain.cpp`, method `init_dmp_rc_gr`.
- GR RC parser/build path: `/research/d7/ascstd/qkduan25/gputimer_merged/core/rctree.cpp`, function `GPUTimer::build_openroad_gr_rc`.
- Explicit DMP RC upload/scaling: `/research/d7/ascstd/qkduan25/gputimer_merged/core/DmpRC.cu`.
- Liberty parsing fixes: `/research/d7/ascstd/qkduan25/Xplace/cpp_to_py/common/lib/LibtertyReader.cpp` and `Liberty.h`.
