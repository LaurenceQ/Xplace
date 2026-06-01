# Power Group Alignment Notes

## OpenROAD grouping rule

Checked `/research/d7/ascstd/qkduan25/GNNTimer/openroad/src/sta/power/Power.cc`.
`Power::power(...)` groups instances in this priority:

1. `macro`: `cell->isMacro() || cell->isMemory() || cell->interfaceTiming()`
2. `pad`: `cell->isPad()`
3. `clock`: `inClockNetwork(inst, clk_network)`
4. `sequential`: `cell->hasSequentials()`
5. `combinational`

`inClockNetwork` returns false if any output pin is not in the clock network;
otherwise true. `ReportPower.cc` prints `Sequential`, `Combinational`,
`Clock`, `Macro`, `Pad`, and `Total` rows.

## Xplace implementation

- Added `GPUTimer::report_power_group_codes()` and pybind exposure.
- `tools/compare_ispd25_route_power_timing.py` now writes `power_group` in
  Xplace instance CSV, summarizes group/component power, and requires every
  group/component to pass 1%.
- The compare summary records Xplace power stage timing:
  `xplace_power_s`, `xplace_power_group_codes_s`,
  `xplace_power_group_summary_s`, and `xplace_write_power_csv_s`.
- Default power path is CUDA activity via `report_power_total_cuda()`.
  CPU activity is only used when `XPLACE_POWER_USE_CPU_ACTIVITY_FOR_POWER`
  is set truthy.
- If an old OpenROAD golden log lacks group rows but OpenROAD instance CSV is
  present, the compare script can reconstruct OpenROAD group sums using the
  Xplace `power_group` column.
- OpenROAD `report_power` is the acceptance oracle. Do not replace it with the
  double-precision sum of `my_dump_power` CSV rows.
- Historical note: upstream-style OpenROAD accumulated `PowerResult` values in
  single-precision float and in DEF component/orderdb instance order. The CSV
  dump stores per-instance values; summing it in Python double differed from
  the printed `report_power` table by more than 1% on `mempool_cluster`.
- Current local OpenROAD changed `PowerResult` to double. The compare script
  therefore summarizes Xplace power for acceptance by streaming DEF
  `COMPONENTS` order with double accumulation
  (`xplace_power_sum_order=def_components_double`). OpenROAD CSV order is only
  a fallback/source check, not the primary oracle.

## 2026-05-24 alignment fixes

Built and installed from `build` with `conda activate gnn`, `make -j8`,
`make install`.

- Missing Liberty power-expression ports are now treated as constant 0 unless
  an explicit const-port file supplies another value. The compare script no
  longer auto-generates a Verilog-derived const-port file by default; set
  `XPLACE_POWER_AUTO_CONST_PORT_FILE=1` only for diagnostics.
- Internal-power input slew now uses the ideal clock transition for sequential
  clock input pins and for pins on `pin_is_clk`/`net_is_clock` clock nets.
  This is only a power-LUT slew override; it does not re-enable activity
  propagation through sequential CK pins or through gated clocks.
- Added debug-only internal arc probe CSV support in
  `tools/compare_ispd25_route_power_timing.py`:
  `XPLACE_POWER_INTERNAL_ARC_PROBE_CSV` plus optional
  `XPLACE_POWER_INTERNAL_ARC_INST_LIST_FILE`.

## Current measured status

- `visible/NV_NVDLA_partition_c` passes after missing-port const0 and clock
  slew fixes:
  `result/ispd25_power_visible_nvdla_clkslew_pinclk_20260524`,
  `pass=True`, `wns_err=2.06476e-05`, `ptotal_err=5.80752e-05`,
  worst group/component `combinational.internal=0.00334668`.
- Regression checks pass:
  `visible/bsg_chip` worst `sequential.leakage=0.00119477`,
  `visible/mempool_tile_wrap` worst `clock.internal=0.00180981`.
- Full visible+blind sweep
  `result/ispd25_power_all_after_clkslew_pinclk_20260524` passes all
  completed non-cluster rows: visible `ariane`, `bsg_chip`,
  `NV_NVDLA_partition_c`, `mempool_tile_wrap`; blind `ariane`, `bsg_chip`,
  `NV_NVDLA_partition_c`, `mempool_tile_wrap`, `mempool_group`.
- `visible/mempool_group` in that no-instance sweep reported
  `power_group_components_pass=False` because OpenROAD returned `-15` and the
  compare script could not reconstruct group rows without an instance CSV. A
  single rerun with instance CSV passes:
  `result/ispd25_power_visible_mempool_group_after_clkslew_inst_20260524`,
  `pass=True`, `ptotal_err=0.00010461`, worst
  `clock.internal=0.000613093`.
- Current-code `mempool_cluster` group/component reruns now pass on both
  splits with default CUDA power and `--no-instance-power-csv`:
  `result/ispd25_power_visible_mempool_cluster_csvgroup_gpu1_20260524` and
  `result/ispd25_power_blind_mempool_cluster_csvgroup_gpu1_20260524`.
  The OpenROAD group oracle source is
  `openroad_csv_by_xplace_group_codes`; unmatched OpenROAD CSV rows were zero
  for both splits.
- Later 2026-05-24 correction: those CSV-group rows were not the requested
  acceptance target. Re-ran `mempool_cluster` against the OpenROAD
  `report_power` log rows using DEF-order float32 Xplace summaries:
  `result/ispd25_power_visible_mempool_cluster_reportpower_deforder_gpu1_20260524`
  and
  `result/ispd25_power_blind_mempool_cluster_reportpower_deforder_gpu1_20260524`.
  Both have `xplace_power_sum_order=def_components_float32`,
  `openroad_power_group_source=openroad_log_report_power`, zero unmatched DEF
  rows, and pass timing/power/group gates. Total power relative errors:
  visible `2.3094106356844595e-05`, blind `2.201154584522056e-05`.
- Final 2026-05-24 double `report_power` audit:
  `result/ispd25_power_all_reportpower_double_20260524`.
  OpenROAD was rebuilt after changing `PowerResult` and `ReportPower` plumbing
  to double, and Xplace summaries use DEF-order double accumulation. All
  visible/blind rows for `ariane`, `bsg_chip`, `NV_NVDLA_partition_c`,
  `mempool_tile_wrap`, `mempool_group`, and `mempool_cluster` pass timing,
  total/component power, and every group/component row against
  `openroad_log_report_power`. All DEF rows matched with zero unmatched
  instances.
- Root-cause confirmation on `mempool_cluster`: old `my_dump_power` CSV double
  totals were visible `9.5206522925` and blind `9.5547557693`, while old
  float `report_power` printed `9.3742122700` and `9.4016513800`. After the
  double patch, `report_power` prints visible `9.5206522200` and blind
  `9.5547556900`, matching the CSV double sums within print precision.

## 2026-05-24 continuation audit

- `result/ispd25_power_component_cluster_visblind_floatroot_default_20260522_105500`
  remains component-only historical evidence. It predates the clock-network
  group classification fixes and must not be used to close current group gates.
- Current script/code evidence for completed rows:
  `run_xplace_worker` calls `report_power_total_cuda()` by default, writes
  `power_activity_engine="cuda"`, records `power_group_codes` and
  `power_group_summary` stages, and `flatten_row` requires every
  group/component pass bit for `power_pass_1pct`.
- `visible/mempool_cluster`: `pass=True`, `ptotal_err=2.98728e-05`, worst
  group/component `clock.internal=0.00071415`.
- `blind/mempool_cluster`: `pass=True`, `ptotal_err=2.87662e-05`, worst
  group/component `clock.internal=0.000766698`.
- Aggregate audit across visible/blind
  `ariane`, `bsg_chip`, `NV_NVDLA_partition_c`, `mempool_tile_wrap`,
  `mempool_group`, and `mempool_cluster` found 12/12 pass with no missing
  rows or failures. The current preferred evidence is the double
  `report_power` full sweep
  `result/ispd25_power_all_reportpower_double_20260524`; the older aggregate
  note mixed the main non-cluster sweep, `visible/mempool_group` instance
  rerun, and two cluster CSV-group reruns.

## Activity findings

- OpenROAD clock pin activity uses the specific clock period/duty from
  `Power::findActivity()`, not the global minimum period.
- Xplace now computes per-clock-pin density/duty from
  `gtdb.pin_clock_periods`, `pin_clock_rise_edges`, and
  `pin_clock_fall_edges`.
- Clock activity seeding and clock-tree propagation are intentionally
  separated: every clock pin gets activity, but only non-sequential clock load
  pins are enqueued to propagate through clock buffers/gates. Enqueuing
  sequential CK pins caused extra D-to-Q propagation versus OpenROAD.
- The old bsg sequential/combinational group mismatch and mempool_tile clock
  gate mismatch are superseded by the 2026-05-24 fixes above. Do not resume the
  old late-pass tolerance or ordered-queue diagnostics unless a new run
  regresses.

## 2026-06-01 CUDA direct-expression activity fix

- Current full-sweep evidence:
  `result/codex_power_force_cuda_all_defaultmax8_20260601/summary.csv`.
  It forces CUDA with `XPLACE_POWER_AUTO_CPU_ACTIVITY_PIN_LIMIT=0` and uses
  default direct-expression max vars 8. All 12 visible/blind ISPD2025 rows pass
  timing, total/component power, and every group/component row. All
  `*.xplace.json` files report `power_activity_engine="cuda"`.
- Root cause for the small-case activity mismatch was the direct polynomial
  activity path using algebraically equivalent but differently rounded float
  formulas versus the BDD/reference path. BDD-only CUDA was correct; the BDD
  implementation was not the bad edit.
- Fix: keep the fast direct expression path, but compute AND/OR/XOR activity
  with BDD-style `fmaf` rounding, and default
  `XPLACE_POWER_DIRECT_EXPR_MAX_VARS` to 8. This preserves the large-case fast
  path while making small bsg-style feedback cases align.
- Representative full-sweep rows:
  visible/bsg `ptotal_err=3.44928e-08`, worst
  `combinational.switching=1.81653e-06`; visible/group power 14.708s versus
  OpenROAD 752.294s (51.15x), worst `clock.internal=0.000613238`;
  visible/cluster power 49.054s versus 2607.337s (53.15x);
  blind/group 14.834s versus 746.888s (50.35x); blind/cluster 50.215s versus
  2603.604s (51.85x).
- Stage profile accounting was also fixed after the full sweep: top-level
  `XPLACE_STAGE power` had included CPU `inst_total` summation and report
  wrapper cleanup that had no `[power_stage_profile]` row. New profile labels
  `total_cpu_sum` and `report_total_unprofiled` close that accounting gap.
  Verification artifacts:
  `result/codex_power_stage_profile_sum2_ariane_20260601` and
  `result/codex_power_stage_profile_sum2_mempool_group_20260601`; the latter
  passes with 57.71x power speed and top/substage power-time diff 0.000369s
  (0.003%).

## 2026-06-01 report-order summary removal

- The old default `report_power_order_summary` path was doing DEF-order
  Python serial double accumulation over every instance. On
  `visible/mempool_cluster` this meant 11.31M instance rows and about 34M
  scalar `.item()` reads, which explained the 83.193s stage. It is not needed
  for normal 1% OpenROAD power acceptance.
- `tools/compare_ispd25_route_power_timing.py` now defaults to
  `power_sum_order=torch_parallel_no_order`: total power is a torch double
  reduce over component tensors, and group power uses `torch.bincount` by
  `power_group_codes`. The DEF-order path is retained behind
  `--strict-report-power-order` or `XPLACE_POWER_STRICT_REPORT_ORDER=1`.
- Validation:
  `result/codex_power_torch_summary_ariane_20260601` passes with
  `power_total_summary=0.000828s`, `power_group_summary=0.099584s`, and no
  `report_power_order_summary` stage.
  `result/codex_power_torch_summary_mempool_group_20260601` passes with
  `power_total_summary=0.000960s`, `power_group_summary=0.041031s`, and no
  `report_power_order_summary` stage; previous full-sweep DEF-order
  `visible/mempool_group` report-order time was 22.091s.
  Strict-path smoke test
  `result/codex_power_torch_summary_ariane_strict_20260601` still reports
  `power_sum_order=def_components_double` and runs
  `report_power_order_summary=0.737912s`.
- Full forced-CUDA sweep after the change:
  `result/codex_power_torch_summary_all_20260601/summary.csv`. All 12 rows
  pass timing, total/component power, and every group/component row; every JSON
  reports `power_activity_engine=cuda` and
  `power_sum_order=torch_parallel_no_order`. No row has a
  `report_power_order_summary` stage.
- Large-case summary times from that full sweep:
  visible/group `power_total_summary=0.001588s`,
  `power_group_summary=0.041865s`; visible/cluster
  `power_total_summary=0.031263s`, `power_group_summary=0.273047s`;
  blind/group `power_total_summary=0.001180s`,
  `power_group_summary=0.040609s`; blind/cluster
  `power_total_summary=0.034732s`, `power_group_summary=0.269061s`.
  Power-stage speedups vs cached OpenROAD power are visible/group 53.42x,
  visible/cluster 50.68x, blind/group 52.33x, blind/cluster 46.83x.

## 2026-06-01 GPU tensor reduce

- Follow-up after the CPU reduce objection: `report_power_total_cuda()` now
  asks `compute_power_activity_cuda()` to return the existing torch CUDA power
  tensors instead of copying them to CPU. `inst_total` is formed on GPU, and
  the C++ stage profile prints `total_gpu_sum`; the old power-tensor download
  path is effectively zero. Other detailed probe/report helpers still default
  to CPU tensors to avoid changing debug CSV behavior.
- Python total summary reduces the CUDA tensors directly. Group summary now
  stacks the three component tensors on GPU and uses one `index_add_` by
  `power_group_codes`; the result is only a 5x3 tensor copied back for JSON.
- Full forced-CUDA validation after this change:
  `result/codex_power_gpu_indexadd_all_20260601/summary.csv`. All 12 rows pass
  timing, total/component power, and every group/component row; every row
  records `xplace_power_tensor_device=cuda:0` and
  `xplace_power_tensor_is_cuda=True`; no row has
  `report_power_order_summary`.
- Large-case GPU summary times from that final sweep:
  visible/group `power_total_summary=0.000512s`,
  `power_group_summary=0.031480s`; visible/cluster
  `power_total_summary=0.001061s`, `power_group_summary=0.064546s`;
  blind/group `power_total_summary=0.000602s`,
  `power_group_summary=0.035884s`; blind/cluster
  `power_total_summary=0.000998s`, `power_group_summary=0.065355s`.
  Power-stage speedups vs cached OpenROAD power are visible/group 57.96x,
  visible/cluster 46.58x, blind/group 51.94x, blind/cluster 51.03x.

## Next debugging targets

- No active numeric group/power mismatch remains for the 12-case visible/blind
  ISPD2025 power group/component acceptance set.
- If a future OpenROAD golden is regenerated, keep `report_power` log rows as
  the acceptance oracle. CSV totals are useful diagnostics only; with the
  patched local OpenROAD, use DEF-order double Xplace summaries to match
  `report_power`.
