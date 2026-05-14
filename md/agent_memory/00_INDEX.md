# Xplace Agent Memory Index

Active scope: ISPD2025 Xplace/GPUTimer direct `--route_segments` timing,
runtime, CPU memory, and GPU memory versus CRPR-off OpenROAD.

Read this file after `AGENTS.md`. Then open only the focused file needed for
the question; do not grep all logs first.

## Read Order

- Timing correctness: `02_openroad_gr_rc_alignment.md`, then
  `06_case_status.md`, then `05_current_case_debugging.md`.
- bsg/NVDLA/ariane skip-fanout: `07_skip_fanout_policy.md`, then
  `05_current_case_debugging.md`.
- DMP branch / arc-level: `08_dmp_fallback_branch.md`, then
  `12_dmp_arc_vs_fallback_profile.md`.
- Speed or memory: `09_speed_memory_status.md`.
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
- `08_dmp_fallback_branch.md`: full arc-level vs hybrid/fused fallback facts.
- `09_speed_memory_status.md`: current optimization state and bottlenecks.
- `10_artifacts.md`: important summaries, logs, and evidence locations.
- `11_memory_policy.md`: rules for adding, deleting, merging, and splitting
  memory notes after important conclusions.
- `12_dmp_arc_vs_fallback_profile.md`: arc-level/fallback kernel timings,
  stage timings, and WNS/TNS consistency.

## Do Not Confuse

- 1% timing pass/fail is separate from 4x speed pass/fail.
- `bsg_chip` timing passes with missing-high-fanout skip `300`.
- `visible/NV_NVDLA_partition_c` and `blind/ariane` need skip `0`.
- `mempool_group/cluster` need skip `300` for memory/time.

## Archive

`archive/` contains older sky130 and historical notes. Do not read it unless
explicitly returning to that work.
