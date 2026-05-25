# Power Alignment Flow
Scope: OpenROAD/OpenSTA power oracle and Xplace CUDA power compare.

## Verified 2441 Paths
- OpenROAD repo: `/research/d7/ascstd/qkduan25/GNNTimer/openroad`.
- OpenROAD bin: `.../openroad/build-check/bin/openroad`.
- Deps env: `/research/d7/ascstd/qkduan25/app/openroad-deps/env.sh`.
- Verified hash `31fc25440f489b2587a9b801c9cfcd06344a0178`; commands exist:
  `my_dump_graph`, `my_dump_pins`, `my_dump_power`.
- Xplace python: `/home/qkduan25/.conda/envs/gnn/bin/python` (`Python 3.10.19`).

## Current ISPD2025 Driver
- Use `tools/compare_ispd25_route_power_timing.py` for ISPD route-segment compare.
- Resolve golden cache to absolute path; OpenROAD cwd is benchmark root.
- Golden paths: `<cache>/openroad_dump/<split>_<design>_power*.csv` plus manifest.
- Valid ISPD golden requires all four power CSVs plus manifest; reject instance-only dumps.
- Acceptance uses the normal OpenROAD `report_power` log rows. CSV dumps are
  optional diagnostics for instance/pin/arc/leakage debugging, not the oracle.
- Do not use `report_power -format json`; for diagnostics, use `my_dump_power`
  four CSVs where the selected OpenROAD build supports it: instance, pins,
  internal arcs, leakage.
- Default OpenROAD for acceptance is
  `/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad`. Use a
  GNNTimer/OpenROAD debug build explicitly only when a diagnostic CSV dump
  command is needed.
- Reuse golden unless DEF, SDC, libs, route_segments, OpenROAD bin, Tcl, or
  dump code changed; use `--force-openroad-golden` only then.

## Required Compare Semantics

- OpenROAD `report_power` is the golden oracle. Xplace must compute power
  itself, then summarize in DEF `COMPONENTS` order. Current local OpenROAD has
  `PowerResult` changed to double, so the acceptance compare uses DEF-order
  double accumulation (`xplace_power_sum_order=def_components_double`).
- OpenROAD CSV sums are diagnostics only; Python double sums of CSV rows can
  differ from upstream-style single-precision `report_power` on large designs
  and should not silently replace the log oracle.
- Compare by instance name, never by internal id.
- Report internal/switching/leakage/total plus worst instance/component.
- Accept only if internal, switching, leakage, and total are each within 1% of OpenROAD.
- Activity debug target: each pin density/duty ideally within 5% of OpenROAD.
- Debug order after a `report_power` failure: first confirm the `report_power`
  total/group row being compared, then use CSV dumps if available to diff
  instance rows and inspect worst pin/internal-arc/leakage rows before changing
  code.
- Only after CSV diff identifies the failing component should source be read:
  switching -> load/activity code; internal -> arc/table/duty code; leakage ->
  when/PG/default code. Do not guess from total-only numbers.
- For activity mismatch, first use final CSVs and OpenROAD source semantics. If
  still unclear, instrument/probe `ensureActivities()` pass-by-pass and record
  target pin, pending register output, density/duty, and enqueue/update changes.
- For pin activity source tracing, follow the mempool_group method: start from
  the largest activity-diff input/pin in the failing cone, walk upstream by
  comparing every input of the current cell between OpenROAD and Xplace, then
  continue through the input with the largest activity mismatch. Stop only when
  all inputs match but the output differs, a seq/root/PI source differs, or a
  cycle prevents progress. Use this traced first source divergence as evidence
  for the next hypothesis.
- Before changing propagation semantics, write the hypothesis and the exact
  evidence needed to prove/disprove it. Do not add heuristic switches or
  broad experiments until the first X/OpenROAD divergence is observed.
- Preserve OpenSTA semantics: switching load cap not Ceff; activity follows
  `ensureActivities()`; internal/leakage use Liberty table/when rules.

## Root Debug
- OR roots: `Power::seedActivities()`/`levelize_->roots()`; dump with
  `OR_POWER_DUMP_ROOTS_FILE`, optional `OR_POWER_ROOT_PROBE_PINS_FILE`.
- X dump: `XPLACE_POWER_DUMP_ROOTS_FILE`, optional
  `XPLACE_POWER_ROOT_PROBE_PINS_FILE`; compare via
  `tools/power_alignment/compare_power_roots.py`.
- Judge: OR root not X candidate => graph/fanin; X candidate not seed => seed
  selection; X seed but zero => seed/enqueue.

## 2026-05-21 bsg Clock-Root Activity Fix
- Evidence path followed the mempool_group method on blind/bsg:
  worst switching instance `.../FE_DBTC1033_n_96`, then
  `g839 -> g847 -> rd_circ_ptr/o_5_sv2v_reg_reg` feedback cone.
- After seq-output phase was matched to OpenROAD, first remaining pass
  divergence moved upstream to `g2665/g2668 -> g850/g860`.
- Expanded pass trace showed Xplace, but not OpenROAD, set many cone pins to
  constant `0/1` at pass 1. Root dump showed those pins were not root/seed in
  either engine, so the source was earlier propagation, not `g2665` logic.
- Key evidence: before fix, Xplace `after_comb pending=214635` while OPR had
  `35`; after removing clock-root fanout propagation, Xplace and OPR pending
  counts matched through the first 12 pass snapshots:
  `35, 57, 2117, 2163, 386, 27461, 132685, ...`.
- Root cause: Xplace seeded clock pins like OpenROAD but also enqueued their
  fanout, making almost every sequential cell pending from clock activity.
  OpenROAD `seedActivities()` excludes clock roots from BFS because clock
  activity is baked in and queried directly by sequential clock expressions.
- Fix: clock pins still get `clock_density/0.5` activity, but are not primary
  input seeds and do not enqueue adjacent vertices/level queues in CPU or CUDA.

## 2026-05-22 Cluster Floating-Load Root Fix
- Evidence path followed the mempool_group upstream method on
  `visible/mempool_cluster`, starting from worst switching/total instance
  `gen_groups[0].i_group/gen_dmas[1].i_axi_dma_backend/i_axi_dma_data_mover/i_fifo_r_emitter/FE_OFC54589_read_req_o_9`.
- Before the fix, Xplace had `FE_OFC54589.../A=0/0` and `ZN=0/1` while
  OpenROAD had `A/ZN density=8.067841024e9`, `A duty=0.451616466`,
  `ZN duty=0.548383534`. Walking upstream reached
  `i_burst_request_fifo/mem_q_reg[0]_deburst` and its source
  `gen_spill_reg.a_data_q_reg_deburst/SI`.
- Key source divergence: `gen_spill_reg.a_data_q_reg_deburst/SI` is a
  no-driver load pin on net `dma_req_split_deburst`; OpenROAD marks it as an
  `input` root with `33.333336e6/0.5`, but Xplace did not seed it. This left
  the DMA burst feedback cone at zero.
- A broad seq-feedback-output seed was falsified: with
  `seq_feedback=547652`, visible cluster overdrove to `26.9218627 W` total
  vs OpenROAD `9.5206523 W`.
- Correct fix: seed no-driver load pins as `floating_load_input` power roots
  in CPU and CUDA activity, but keep `XPLACE_POWER_SEED_SEQ_FEEDBACK_OUTPUTS`
  default false. The env switch remains for diagnostics only.
- Isolation probe with floating-load roots and seq feedback disabled aligned
  the target cone: `mem_q_reg[0]_deburst/Q=1.93459392e8/0.08173846` and
  `g25728/Z=8.067840512e9/0.45161742`, matching OpenROAD
  `1.93459136e8/0.08173835` and `8.067841024e9/0.45161647`.
