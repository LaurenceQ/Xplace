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
