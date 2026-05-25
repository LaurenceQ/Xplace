# Power Acceptance Plan
Scope: ISPD2025 NanGate45 route power versus OpenROAD `report_power`.

## Acceptance Standard
- OpenROAD `report_power` log rows are the oracle; compare against the printed
  internal, switching, leakage, total, and group rows.
- OpenROAD power CSVs are diagnostic dumps only. Do not use Python
  double-precision CSV sums as the acceptance oracle.
- Required pass: internal, switching, leakage, and total each <=1% rel_err.
- Total-only <=1% is not sufficient; all four component columns must pass.
- Pin activity density/duty within 5% of OpenROAD is a debug target, not the
  first-round pass gate.
- If power fails: instance compare first, then component-specific rows; run full
  pin activity diff only when switching/internal points to activity.

## Script State
- `tools/compare_ispd25_route_power_timing.py` default OpenROAD is the primary
  reference binary:
  `/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad`.
- `power_pass_1pct` now equals internal && switching && leakage && total.
- Keep per-component columns: `power_internal_pass_1pct`,
  `power_switching_pass_1pct`, `power_leakage_pass_1pct`,
  `power_total_pass_1pct`.
- Current local OpenROAD has `PowerResult` changed from `float` to `double`.
  For acceptance with that binary, Xplace power is summarized in DEF
  `COMPONENTS` order with double accumulation
  (`xplace_power_sum_order=def_components_double`) to match the patched
  OpenROAD `report_power` rows.

## 2026-05-19 Sweep
- Output: `result/ispd25_power_component_sweep_visblind_20260519_204804`.
- Command used visible+blind, designs ariane/bsg/NVDA/mempool_tile/group,
  `--missing-fanout-skip 300`, `--threads 16`, `--openroad-threads 16`.
- Build/install before run succeeded in conda `gnn`.
- All 10 cases got complete GNNTimer/OpenROAD golden CSVs and Xplace compare
  CSVs; no `GOLDEN_FAIL`.

## Results
- PASS: visible/bsg_chip, visible/NV_NVDLA_partition_c,
  visible/mempool_tile_wrap, visible/mempool_group, blind/mempool_tile_wrap.
- NEEDS_FIX: visible/ariane switching `9.04%`; blind/ariane switching `32.41%`.
- NEEDS_FIX: blind/bsg_chip internal `2.75%`, switching `34.85%`, total `1.07%`.
- NEEDS_FIX: blind/NV_NVDLA_partition_c switching `1.05%`.
- NEEDS_FIX: blind/mempool_group internal `7.21%`, switching `19.09%`,
  total `11.04%`.

## 2026-05-21 Blind CUDA Clock-Fix Sweep
- Output: `result/ispd25_power_component_sweep_blind_cuda_clockfix_20260521_233625`.
- Command: blind split, designs `ariane`, `bsg_chip`,
  `NV_NVDLA_partition_c`, `mempool_tile_wrap`, `mempool_group`,
  default CUDA activity, GNNTimer OpenROAD golden reused.
- Result: 5/5 pass component acceptance; no mem_cluster case run.
- Component rel_errs:
  - `blind/ariane`: internal `1.248e-05`, switching `0.001900`,
    leakage `8.813e-07`, total `5.873e-06`.
  - `blind/bsg_chip`: internal `0.000389`, switching `0.002316`,
    leakage `3.847e-06`, total `0.000127`.
  - `blind/NV_NVDLA_partition_c`: internal `2.139e-05`,
    switching `0.000304`, leakage `1.397e-05`, total `1.687e-05`.
  - `blind/mempool_tile_wrap`: internal `4.948e-06`,
    switching `2.406e-06`, leakage `2.944e-08`, total `3.446e-06`.
  - `blind/mempool_group`: internal `1.400e-05`,
    switching `0.004476`, leakage `1.323e-05`, total `0.001613`.
- Single-case bsg confirmation before sweep:
  `result/ispd25_power_component_blind_bsg_clockfix_20260521_233314`,
  all four components pass.

## 2026-05-22 Mempool Cluster Visible+Blind Sweep
- Output: `result/ispd25_power_component_cluster_visblind_floatroot_default_20260522_105500`.
- Command: visible+blind split, design `mempool_cluster`, default CUDA power,
  `--skip-openroad`, `--missing-fanout-skip 300`, GNNTimer/OpenROAD golden
  CSVs reused from `result/ispd25_route_power_openroad_golden_cache`.
- Result: 2/2 pass component acceptance.
- Component rel_errs:
  - `visible/mempool_cluster`: internal `0.000445`, switching `0.001165`,
    leakage `4.200e-06`, total `0.000529`.
  - `blind/mempool_cluster`: internal `3.138e-05`, switching `8.429e-05`,
    leakage `2.693e-06`, total `4.692e-06`.
- Generated compare CSVs:
  - `compare/visible_mempool_cluster_power_compare.csv`
  - `compare/blind_mempool_cluster_power_compare.csv`
- Worst total rows after alignment:
  - visible:
    `gen_groups[0].i_group/gen_tiles[4].i_tile/i_local_resp_interco/gen_outs[6].i_rr_arb_tree/FE_OFC34225_n_70`,
    abs diff `4.295e-05`.
  - blind:
    `gen_groups[2].i_group/gen_remote_interco[1].i_remote_interco/gen_lic.i_xbar/resp_xbar/FE_OFC100259_gen_remote_interco_1__slave_remote_resp_rdata_8__data_4`,
    abs diff `1.256e-05`.

## 2026-05-24 Group/Component Alignment Sweep
- Fixes before sweep:
  missing Liberty power-expression ports default to const0 unless explicitly
  supplied; auto Verilog const-port generation is off by default; internal
  power LUT slew uses ideal clock transition for sequential CK pins and
  pins on `pin_is_clk`/`net_is_clock` clock nets.
- Build/install succeeded in conda `gnn`.
- Main sweep output:
  `result/ispd25_power_all_after_clkslew_pinclk_20260524`.
- Completed non-cluster rows pass timing, component power, and group/component
  power:
  visible `ariane`, `bsg_chip`, `NV_NVDLA_partition_c`,
  `mempool_tile_wrap`; blind `ariane`, `bsg_chip`,
  `NV_NVDLA_partition_c`, `mempool_tile_wrap`, `mempool_group`.
- `visible/NV_NVDLA_partition_c` single confirmation:
  `result/ispd25_power_visible_nvdla_clkslew_pinclk_20260524`,
  `pass=True`, total power rel_err `5.80752e-05`, worst group/component
  `combinational.internal=0.00334668`.
- `visible/mempool_group` needs an instance CSV when OpenROAD returns `-15`
  without group rows. Rerun with instance CSV:
  `result/ispd25_power_visible_mempool_group_after_clkslew_inst_20260524`,
  `pass=True`, total power rel_err `0.00010461`, worst group/component
  `clock.internal=0.000613093`.
- Historical CSV-oracle note: `visible/blind mempool_cluster` passed after the
  DMP memory fixes and the OpenROAD CSV group-oracle workaround. This is now
  superseded for acceptance by the `report_power` rows below.
  - Visible output:
    `result/ispd25_power_visible_mempool_cluster_csvgroup_gpu1_20260524`,
    `pass=True`, total power rel_err `2.98728e-05`, worst group/component
    `clock.internal=0.00071415`.
  - Blind output:
    `result/ispd25_power_blind_mempool_cluster_csvgroup_gpu1_20260524`,
    `pass=True`, total power rel_err `2.87662e-05`, worst group/component
    `clock.internal=0.000766698`.
- Historical single-precision `report_power` rerun: `visible/blind
  mempool_cluster` were re-run against OpenROAD `report_power` log rows with
  DEF-order float32 Xplace summaries:
  `result/ispd25_power_visible_mempool_cluster_reportpower_deforder_gpu1_20260524`
  and
  `result/ispd25_power_blind_mempool_cluster_reportpower_deforder_gpu1_20260524`.
  Both pass timing, component power, and every group/component row. Total power
  rel_errs are `2.3094106356844595e-05` visible and
  `2.201154584522056e-05` blind.
- Double `report_power` full 12-case audit:
  `result/ispd25_power_all_reportpower_double_20260524`.
  OpenROAD was rebuilt from `/research/d7/ascstd/qkduan25/OpenROAD` after
  changing `PowerResult` storage/accessors and `ReportPower` formatting
  plumbing to double. All 12 visible/blind rows pass timing, component power,
  and every group/component row against the patched OpenROAD `report_power`
  log oracle. All rows use `openroad_power_group_source=openroad_log_report_power`
  and `xplace_power_sum_order=def_components_double`; DEF rows matched with
  zero unmatched instances.
- Double-run total power rel_errs: visible `ariane=4.947947e-06`,
  `bsg_chip=1.540981e-07`, `NV_NVDLA_partition_c=5.807313e-05`,
  `mempool_tile_wrap=3.592344e-05`, `mempool_group=1.046101e-04`,
  `mempool_cluster=2.987311e-05`; blind `ariane=3.025418e-06`,
  `bsg_chip=3.958099e-05`, `NV_NVDLA_partition_c=2.423427e-07`,
  `mempool_tile_wrap=3.035552e-05`, `mempool_group=3.433124e-05`,
  `mempool_cluster=2.876628e-05`.
- Cluster CSV/report sanity after the OpenROAD double patch:
  old `my_dump_power` CSV double totals are visible `9.5206522925` and blind
  `9.5547557693`; old float `report_power` printed `9.3742122700` and
  `9.4016513800`; patched double `report_power` prints `9.5206522200` and
  `9.5547556900`.
