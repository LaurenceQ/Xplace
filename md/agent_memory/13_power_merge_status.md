# Power Merge Status

Scope: CUDA power from `../Xplace_power_merged_20260513_094803`.
- Power code: `cpp_to_py/gputimer/core/Power.cpp` and `propagate.cu`.
- APIs include activity, switching, internal, leakage, total, row probes.
- Liberty parser keeps functions, internal/sequential/leakage power, clock-gate attrs.
- 2026-05-15 fix: propagated activity can overwrite clock-seeded activity;
  fixed NVDA manual gated-clock switching.
- 2026-05-15 fix: internal lookup uses ideal clock slew; fixed NVDA gate power.
- 2026-05-15 fix: internal clock-slew override applies only to ideal clocks;
  fixed sky130 SPEF `blabla`/`zipdiv` internal drift.
- 2026-05-15 fix: clock-origin activity overwrite only for ideal clocks;
  fixed sky130 `picorv32a`/`wbqspiflash` activity drift.
- 2026-05-15 latch/case-analysis support did not change mempool_group power.
- Power accept is component-level: internal, switching, leakage, total each <=1% vs OpenROAD.
- Timing pass treats OpenROAD WNS/TNS `0` with Xplace nonnegative slack as pass.
- Power compare auto skip: `0` for visible NVDA, blind ariane/bsg/NVDA;
  timing matrix still uses `300` for blind bsg/NVDA.
- Always run with `CUDA_VISIBLE_DEVICES=0` unless user changes GPU.

- `...powerfix_midcases/`: visible/blind ariane, bsg, NVDA pass timing/power.
- Visible NVDA component errs: internal `7.4e-05`, switching `0.00192`.
- Visible/blind ariane total pass (`0.00476`/`0.00257`) but switching noisy.
- Blind bsg/NVDA total power pass records used skip `0`; group uses skip `300`.
- `...powerfix_group/`: mempool_group timing passes; power fails visible
  `0.0185`, blind `0.10845`.

- Macro/leakage align; mismatch is standard-cell dynamic/activity.
- Visible group: total err `0.0185`; seq `0.00288`, combo `0.0353`.
- Blind group: total err `0.10845`; seq `0.0720`, combo `0.1686`.
- OR blind root probe: `5854` roots, `0` cycle back edges.
- X root stats match OR: `5854` seeds; cycle-root seeding was a no-op.
- Therefore root set/cycle roots are not the current blind mismatch.
- Broad seq-output root seeding is wrong: `16.2367 W` vs OR `4.5166 W`.
- `MAX_PASSES=200/1000` gives `4.2630/4.3014 W` vs OR `4.5166 W`;
  even 1000 passes stays `4.76%` low, so pass count is not enough.
- OpenSTA default max passes is also `50`.
- `XPLACE_POWER_USE_DMP_LOAD=0` makes switching zero; timing `pinLoad` is empty
  in route-segment DMP, so power must not use it directly.
- `XPLACE_POWER_VOLTAGE=1.1` did not change blind group power.
- 2026-05-21 update: CPU/CUDA activity are aligned for the blind/bsg debug
  cone after matching OpenROAD clock-root semantics. Probe
  `result/debug_bsg_cpu_cuda_probe_clockfix_20260521_232928` had 154 common
  pins, max duty diff `7.03e-05`, max density abs diff `6406` on ~`3.1e7`.
  Treat this as numerical CUDA/CPU drift, not a semantic activity split.
- 2026-05-22 update: visible/blind `mempool_cluster` now pass component power
  acceptance with default CUDA power path after seeding no-driver load pins as
  `floating_load_input` roots and keeping broad seq-feedback output seeding
  default off. Acceptance run:
  `result/ispd25_power_component_cluster_visblind_floatroot_default_20260522_105500`.
  Rel_errs: visible internal `0.000445`, switching `0.001165`, leakage
  `4.20e-06`, total `0.000529`; blind internal `3.14e-05`, switching
  `8.43e-05`, leakage `2.69e-06`, total `4.69e-06`.
- Selected OR-top instance powers are close; next check coverage/low-activity
  population by type, especially MUX2_X1 and CLKBUF_X3.
- Fixed SDC `printf` noise; missing-object/unit prints are env-gated.

- 2026-05-15 sky130 Xplace-only: reused cached OR CSVs, 21/21 power ok;
  worst total `picorv32a 3.692918e-06`.
- 2026-05-25 sky130 group/component audit:
  `result/sky130_power_group_all_20260525`. Added
  `tools/power_alignment/compare_sky130_power_groups.py` to compare patched
  OpenROAD `report_power` group rows against current Xplace CUDA power
  summarized in DEF `COMPONENTS` order using `report_power_group_codes()`.
  Total/component power remains aligned for all 21 cases: total rel_errs are
  about `1e-9..4e-6`, and DEF instance matching is complete with zero
  unmatched rows. Group/component 1% fails for all 21 cases because clock
  group power is under-classified in Xplace and lands in combinational. Worst
  examples: `aes128 clock.switching=0.3466`, `jpeg_encoder
  clock.switching=0.3950`, `xtea clock.internal=0.4321`; several small designs
  show large combinational relative errors because their OpenROAD
  combinational baseline is tiny and the missing clock power is counted there.
  The SDC reader still logs `set_propagated_clock not supported yet`, matching
  the suspected clock-network classification root.
- 2026-05-25 sky130 group/component fix:
  `report_power_group_codes()` now propagates clock-network nets through CORE
  combinational BUF/INV cells whose output function is the direct or inverted
  clock input, matching OpenROAD `Power::inClockNetwork()`/`ClkNetwork` BFS
  semantics closely enough for the sky130 CTS netlists. Rebuilt and installed
  Xplace. Acceptance run:
  `result/sky130_power_group_all_after_clocknet_20260525`. All 21/21 sky130
  cases pass group/component 1%; DEF component matching is complete with zero
  unmatched rows. Worst per-case group.component rel_err is
  `wbqspiflash combinational.internal=5.328143e-05`; worst total rel_err is
  `picorv32a=3.745422e-06`. Previous large clock/combinational swaps
  disappear; e.g. `aes128` worst is now `clock.internal=1.221340e-08`.
- 2026-05-15 ISPD visible/ariane: `my_dump_power` four CSVs, total err
  `0.00455499`; cache `result/ispd25_route_power_openroad_golden_cache`.
