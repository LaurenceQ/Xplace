# Xplace Agent Memory Index

Active scope: ISPD2025 Xplace/GPUTimer direct timing, runtime, memory, and
power versus OpenROAD references.

Read this after `AGENTS.md`; open focused notes before grepping logs.

## Read Order

- Timing correctness: `02_openroad_gr_rc_alignment.md`, then
  `06_case_status.md`, then `05_current_case_debugging.md`.
- bsg/NVDLA/ariane skip-fanout: `07_skip_fanout_policy.md`, then
  `05_current_case_debugging.md`.
- DMP timing path: `08_dmp_timing_path.md`, then
  `12_dmp_arc_vs_fallback_profile.md`.
- Route-gradient derivatives: `22_dmp_pi_rc_derivative_chain.md`, then
  `24_route_grad_impl_status.md`.
- Speed or memory: `09_speed_memory_status.md`, then
  `17_route_stage_speed_audit.md`, then
  `20_mempool_cluster_gpu_memory.md`, then
  `21_dmp_fanout_wave_arc_level_plan.md`, then
  `23_read_graph_rc_hotspots.md`.
- Power accept/debug: `18_power_acceptance_plan.md`, then `15_power_alignment_flow.md`, `16_nangate45_power_alignment.md`, `13_power_merge_status.md`.
- Power group/component alignment: `19_power_group_alignment.md`.
- Hypotheses/results: `14_hypotheses_validation.md`.
- Commands and paths: `04_tools.md`.
- Trap check before edits: `03_pitfalls.md`.
- Memory maintenance: `11_memory_policy.md`.

## Active Files

- `01_human_feedback.md`: user constraints, repo scope, communication rules.
- `02_openroad_gr_rc_alignment.md`: current timing target and references.
- `03_pitfalls.md`: high-risk mistakes to avoid.
- `04_tools.md`: canonical paths and minimal run commands.
- `05_current_case_debugging.md`: case-specific triage notes.
- `06_case_status.md`: current pass/fail and speed rows.
- `07_skip_fanout_policy.md`: missing-high-fanout fallback settings.
- `08_dmp_timing_path.md`: current single DMP timing propagation path.
- `09_speed_memory_status.md`: current optimization state and bottlenecks.
- `10_artifacts.md`: important summaries, logs, and evidence locations.
- `11_memory_policy.md`: rules for adding, deleting, merging, and splitting
  memory notes after important conclusions.
- `12_dmp_arc_vs_fallback_profile.md`: arc-level/fallback kernel timings,
  stage timings, and WNS/TNS consistency.
- `13_power_merge_status.md`: CUDA power merge status and validation.
- `14_hypotheses_validation.md`: compact hypothesis/test/result log.
- `15_power_alignment_flow.md`: OpenROAD oracle/cache and compare workflow.
- `19_power_group_alignment.md`: OpenROAD power group rule, Xplace group
  implementation, current group-level status, and next activity mismatches.
- `20_mempool_cluster_gpu_memory.md`: mempool_cluster GPU memory attribution,
  per-array keep/remove candidates, and recommended memory optimization order.
- `21_dmp_fanout_wave_arc_level_plan.md`: proposed fanout-wave split,
  compact wave scratch memory budget, and arc-level backward RAT plan.
- `22_dmp_pi_rc_derivative_chain.md`: DMP PI/RC derivative-chain notes.
- `23_read_graph_rc_hotspots.md`: read/prep/graph/RC hotspot profile,
  substage timing evidence, and optimization strategy.
- `24_route_grad_impl_status.md`: current route segment gradient implementation,
  validation evidence, and remaining derivative-chain gaps.

## Do Not Confuse

- 1% timing pass/fail is separate from 4x speed pass/fail.
- `bsg_chip` timing passes with missing-high-fanout skip `300`.
- `visible/NV_NVDLA_partition_c` and `blind/ariane` need skip `0`;
  `mempool_group/cluster` need skip `300` for memory/time.

Archive: `archive/` contains older sky130 and historical notes; read only on request.
