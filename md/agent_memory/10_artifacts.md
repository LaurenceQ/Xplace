# Important Artifacts

## Current Matrix

- `result/ispd25_direct_route_latest/xplace_openroad_all_case_matrix.md`
- `result/ispd25_direct_route_latest/xplace_openroad_all_case_matrix.csv`

## Speed Matrix

- `result/ispd25_direct_route_latest/quiet_4x_all_case_20260514_gpu0/xplace_openroad_all_case_matrix.csv`
- `result/ispd25_direct_route_latest/quiet_4x_all_case_20260514_gpu1/xplace_openroad_all_case_matrix.csv`

## Fallback A/B

- `result/ispd25_direct_route_latest/fallback_ab/summary.csv`
- `result/ispd25_direct_route_latest/fallback_ab_fused/summary.csv`
- `result/ispd25_direct_route_latest/fallback_ab/hybrid_vs_full_arc_visible_ariane.md`

## bsg Evidence

- `result/ispd25_direct_route_latest/evidence/bsg_chip.route_segments_vs_openroad_gr_rc.log`
- `result/ispd25_direct_route_latest/logs/visible/bsg_chip.direct_route.log`

## NVDLA Evidence

- `result/ispd25_direct_route_latest/evidence/root_cause_20260513/visible_NV_NVDLA_partition_c/`
- `result/ispd25_direct_route_latest/logs/visible/NV_NVDLA_partition_c.direct_route.log`

## Recent Timer Split Verification

- `result/codex_timer_placement_align/logs/visible/mempool_tile_wrap.timer_only.log`
- `result/codex_timer_placement_align/logs/visible/mempool_tile_wrap.placement_import.log`
- `result/codex_timer_placement_align/logs/blind/mempool_tile_wrap.timer_only.log`
- `result/codex_timer_placement_align/logs/blind/mempool_tile_wrap.placement_import.log`

## Power Speed Final 2026-05-31

- 12-case final validation:
  `result/codex_power_final12_pinmap_hash8g_20260531/summary.csv`
- Large-case stage logs:
  `result/codex_power_final12_pinmap_hash8g_20260531/logs/xplace/*mempool*.log`
- Nsight Systems hotspot profile:
  `result/nsight_power_20260531/visible_mempool_group_final.nsys-rep`
- Final status: all 12 pass; large power-stage speedups are
  visible/group 56.83x, visible/cluster 56.19x, blind/group 56.34x,
  blind/cluster 53.86x.

## Power CUDA Activity / Stage Profile 2026-06-01

- Full forced-CUDA 12-case validation after direct-expression activity fixes:
  `result/codex_power_force_cuda_all_defaultmax8_20260601/summary.csv`.
  Command used `XPLACE_POWER_AUTO_CPU_ACTIVITY_PIN_LIMIT=0`,
  `XPLACE_POWER_PROFILE_STAGES=1`, `XPLACE_POWER_PRINT_PASS_STATS=1`,
  `--skip-openroad`, and OpenROAD double cache
  `result/ispd25_route_power_openroad_double_cache_20260524`.
- Status: 12/12 pass; every Xplace JSON reports
  `power_activity_engine="cuda"`.
- Large-case power-stage speedups from the full sweep:
  visible/group 51.15x, visible/cluster 53.15x, blind/group 50.35x,
  blind/cluster 51.85x.
- Stage-profile sum fix evidence after adding `total_cpu_sum` and
  `report_total_unprofiled` profile rows:
  `result/codex_power_stage_profile_sum2_ariane_20260601` and
  `result/codex_power_stage_profile_sum2_mempool_group_20260601`.
  `visible/ariane` power top/substage diff is 0.000183s (0.008%);
  `visible/mempool_group` pass speed is 57.71x and power top/substage diff is
  0.000369s (0.003%).

## Power Torch Summary / No DEF-Order Default 2026-06-01

- Fast default summary evidence:
  `result/codex_power_torch_summary_ariane_20260601` and
  `result/codex_power_torch_summary_mempool_group_20260601`.
- Both pass timing/power/group gates with
  `power_sum_order=torch_parallel_no_order` and no
  `report_power_order_summary` stage. Representative times:
  ariane `power_total_summary=0.000828s`, `power_group_summary=0.099584s`;
  mempool_group `power_total_summary=0.000960s`,
  `power_group_summary=0.041031s`.
- Strict DEF-order debug path remains available and was smoke tested in
  `result/codex_power_torch_summary_ariane_strict_20260601`
  (`power_sum_order=def_components_double`,
  `report_power_order_summary=0.737912s`).
- Full 12-case forced-CUDA validation after switching the default summary path:
  `result/codex_power_torch_summary_all_20260601/summary.csv`.
  Status: 12/12 pass; all JSON files report `power_activity_engine=cuda` and
  `power_sum_order=torch_parallel_no_order`; no case runs the old
  `report_power_order_summary` stage.
- Large-case power-stage speedups in that sweep:
  visible/group 53.42x, visible/cluster 50.68x, blind/group 52.33x,
  blind/cluster 46.83x. Cluster torch summary stages are small:
  visible/cluster `power_total_summary=0.031263s`,
  `power_group_summary=0.273047s`; blind/cluster
  `power_total_summary=0.034732s`, `power_group_summary=0.269061s`.
- Final GPU-tensor reduce sweep after keeping power tensors on CUDA:
  `result/codex_power_gpu_indexadd_all_20260601/summary.csv`.
  Status: 12/12 pass; all rows have `xplace_power_tensor_device=cuda:0`,
  `xplace_power_tensor_is_cuda=True`, `power_activity_engine=cuda`, and no
  `report_power_order_summary`.
- Large-case GPU summary stages in the final sweep:
  visible/group `power_total_summary=0.000512s`,
  `power_group_summary=0.031480s`; visible/cluster
  `power_total_summary=0.001061s`, `power_group_summary=0.064546s`;
  blind/group `power_total_summary=0.000602s`,
  `power_group_summary=0.035884s`; blind/cluster
  `power_total_summary=0.000998s`, `power_group_summary=0.065355s`.

## Timer-Only Mid Reruns
`result/timer_only_visible_mid_20260514/logs/visible/`
`result/timer_only_blind_mid_20260514/logs/blind/`

## Sky130 CRPR-Off Rerun

- 2026-05-14 GPU0: `result/sky130_crpr_off_rerun_20260514/summary.csv`
- Command uses `TimingPredict/data/netlists` and
  `GNNTimer/csv_graph_sky130_crpr_off`.
- Status: 21/21 ok; wall 8:44.98, max RSS 3805704 KB.
- Worst: AT `8.62e-4` des, slew `1.545e-3` aes192, RAT `5.57e-4`
  aes192, endpoint RAT `4.48e-4` aes256, slack `1.124e-3` aes256.
