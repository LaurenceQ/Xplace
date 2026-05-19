# NanGate45 Power Alignment
Scope: ISPD2025 NanGate45 route-segment power, especially `mem*`.

## Golden / Baseline
- Cached OR golden: `result/ispd25_route_power_openroad_golden_cache/openroad_dump`.
- Reuse `visible_mempool_group_power*.csv` unless inputs or OR dump code change.
- OR visible/mempool_group total `4.589715868 W`; X baseline
  `4.494483077 W`, err `2.0749%`; leakage aligned, internal/switching low.

## Flow Rules
- First compare cached OR CSV vs X CSV; if unexplained, trace OR
  `ensureActivities()` per pass/pending regs.
- X pin probes use `XPLACE_POWER_PROBE_PIN_LIST_FILE`; CPU probe uses
  `XPLACE_POWER_PROBE_CPU_ACTIVITY=1`; X-only runs use `--skip-openroad`.
- Path trace keeps only `trace_path.tsv/json`, OR/X trace TSV, compare md/json.

## Root Debug
- Root TSV run:
  `result/ispd25_nangate45_mempool_group_rootdebug_visible_20260519_135155`.
- OR roots `5851`, seeded `5850`; X actual seeds `5851`; OR-seeded but X-not
  `0`; X extra seed is `clk_i`. Missing root/frontier set is rejected.

## Path Trace 2026-05-19
- Path source/X/OR/compare: `...pathtrace_only_20260519_150851`,
  `...pathtrace_xcpu_20260519_151620`, `...pathtrace_or_20260519_151620`,
  `...pathtrace_compare_20260519_153000`.
- Common seed root on traced paths: `group_id_i[1]`.
- Paths reach `FE_RC_95112_0:ZN`, `FE_OCPC470995_soc_qvalid:A/Z`.

## Current Finding
- Full X activity run: `result/ispd25_nangate45_mempool_group_fullpin_visible_20260519_170614`.
- X full pin CSV has `12,013,632` rows; OR full pin CSV has `12,002,433`
  rows; all OR rows matched by pin name.
- Full diff before latch fix: `pin_activity_diff/pin_activity_diff_report.md`;
  mismatches `4,960,335`, OR-nonzero/X-zero `2,562,795`.
- g99/A2 path from common seed `group_id_i[1]`: steps 1-11 match; first full
  final-activity divergence is `FE_RC_119370_0/ZN` (`AND2_X4`) because side
  input `FE_RC_119370_0/A1` is OR `1.3417414062e5`/duty `1.6987e-5`, X `0/0`.
- Therefore g99/A2 final OR `4.52164256e8`/duty `0.6143747568`, X `0/1`;
  g99/ZN OR `4.52164256e8`/duty `0.38562524319`, X `0/0`.
- Source trace: `.../source_trace_fe_rc119370_a1_fullbranches/source_trace.md`.
- Strict B2 trace: `.../seed_output_fe_rc97449_b2_strict/seed_output_and_branch_trace.md`.
- `FE_RC_97449_0` inputs do not all match: B2 is OR `1.328965625e5`/`1.6868e-5`,
  X `0/0`; B1 also tiny OR-nonzero/X-zero.
- Max-diff from `clk_en_reg/D` cycles through `data_req_q_reg_hit`; no all-match gate.
- Root cause found: X Liberty parser ignored latch `data_in`; power used only
  `next_state/clocked_on`, so `DLL_X1/DLH_X1` latch outputs were never seq-seeded.
- Fix: parse `data_in` as next-state and use latch `enable` as seq clock expr.
- Latch-fix probe only: Ptotal err `0.00157496`; `clk_en_reg/Q` CUDA
  `2.8259098e7`/duty `0.0036529`; full CSV diff not rerun yet.
