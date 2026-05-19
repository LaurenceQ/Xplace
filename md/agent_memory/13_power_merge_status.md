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
- Power accept is component-level: internal, switching, leakage each <=1% vs OpenROAD; total-only pass is stale.
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
- CPU activity probe diverges from CUDA and clamps many pins to clock density;
  CPU activity path is not a reliable reference for this flow.
- Selected OR-top instance powers are close; next check coverage/low-activity
  population by type, especially MUX2_X1 and CLKBUF_X3.
- Fixed SDC `printf` noise; missing-object/unit prints are env-gated.

- 2026-05-15 sky130 Xplace-only: reused cached OR CSVs, 21/21 power ok;
  worst total `picorv32a 3.692918e-06`.
- 2026-05-15 ISPD visible/ariane: `my_dump_power` four CSVs, total err
  `0.00455499`; cache `result/ispd25_route_power_openroad_golden_cache`.
