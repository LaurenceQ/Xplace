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

## Source Trace Method From Mempool Group
- Start from the biggest OR/X activity mismatch in the failing component.
- If the mismatching point is a cell output, compare all input pins of that
  cell in OpenROAD and Xplace. If all inputs match but output differs, the
  hypothesis is local expression/arc/eval semantics.
- If any input differs, select the input with the largest activity gap and walk
  upstream through its driver. Repeat until reaching a seq output/root/PI, a
  cycle, or an all-inputs-match gate.
- Use the first source divergence found by this walk as the evidence supporting
  the next hypothesis. Do not skip directly to broad event-order or heuristic
  propagation changes.
- Concrete mempool_group artifacts:
  `source_trace_fe_rc119370_a1_fullbranches/source_trace.md`,
  `seed_output_fe_rc97449_b2_strict/seed_output_and_branch_trace.md`, and
  `trace_clk_en_reg_d_maxdiff/seed_output_and_branch_trace.md`.

## Ariane Activity Observations 2026-05-20
- Full visible/ariane pin diff uses existing GNNTimer/OpenROAD CSV:
  `result/ispd25_route_power_openroad_golden_cache/openroad_dump/visible_ariane_power_pins.csv`
  vs `result/ispd25_power_debug_visible_ariane_fullpin_20260519_234917/activity/visible_ariane_xplace_pin_activity.csv`.
- Pin matching is complete: OpenROAD rows `477454`, matched `477454`.
  Mismatches `32669`; OR-nonzero/X-zero `3930`; X-nonzero/OR-zero `140`.
- Power default after rebuild still fails only switching:
  `result/ispd25_power_debug_visible_ariane_default_after_experiments_20260520_134232`,
  Pint `0.477%`, Psw `9.044%`, Pleak `0.102%`, Ptotal `0.455%`.
- Root sets are not the issue: OR-seeded but X-not-seeded is empty; X extra
  root is `clk_i`.
- Observed OpenROAD trace sequence, not root cause: in one pass
  `issue_stage_i/i_scoreboard/FE_RC_10686_0/ZN` is temporarily set to
  zero-density duty `1`, then the FE chain reaches `csr_regfile_i/g87881/A2`.
  At that moment OpenROAD computes `csr_regfile_i/g87881/ZN`
  `7.692308e7/0.5`; later final FE-chain static duties return to the same
  final values that Xplace has.
- Xplace final CSV keeps downstream pins zero (`g128097/ZN`,
  `debug_mode_q_reg/D/Q`, `inc_ADD_UNS_OP/g3976/ZN`,
  `cycle_q_reg[2]/D`). This is an observed mismatch only; do not label it an
  event-order bug until the first differing X/OR event is proven.
- Experiments rejected:
  `--missing-fanout-skip 0` no change; `XPLACE_POWER_USE_TIMING_LEVELS=1`
  worsens switching to about `19.6%`; `XPLACE_POWER_ACTIVITY_MAX_COMB_SWEEPS=1`
  worsens to about `19%`; seq-input snapshot only improves Psw to `8.57%`;
  keep-max density overestimates Psw by about `295%`; duty-history density
  becomes too slow and was killed after several minutes.
- Heuristic source patches from these experiments were reverted. Future ariane
  work must state the hypothesis first and gather event-level evidence before
  changing propagation semantics.

### Ariane Source-Trace Evidence 2026-05-20
- Mem-style traces from the largest mismatching inputs:
  `trace_cycle_q2_d_maxdiff/seed_output_and_branch_trace.md`,
  `trace_perf16_2_d_maxdiff/seed_output_and_branch_trace.md`,
  `trace_debug_mode_d_maxdiff/seed_output_and_branch_trace.md`,
  and `source_evidence_top10/source_evidence.md`.
- Common seed/root activity is still rejected as source: common seeded pins
  compared `216`, common seed mismatches `0`, OR seeded but X not actual `0`;
  X only extra actual seed is `clk_i`.
- `csr_regfile_i/cycle_q_reg[2]/D` max branch:
  `D -> inc_ADD_UNS_OP/g3976/ZN -> cycle_q_reg[2]/Q -> D`; both `A/B`
  inputs of `g3976` mismatch, so no all-inputs-match combinational gate is
  proven there.
- `i_perf_counters/perf_counter_q_reg[16][2]/D` max branch similarly cycles:
  `D -> inc_ADD_UNS_OP225/g4269/Z -> perf_counter_q_reg[16][2]/Q -> D`.
- The shared `debug_mode_q_reg/D` branch cycles through
  `g120510/ZN -> g121412/ZN -> g123841/ZN -> g128097/ZN ->
  FE_OFC21521_debug_mode/ZN -> FE_OFC6923_debug_mode/ZN ->
  debug_mode_q_reg/Q -> D`. At `debug_mode_q_reg/Q`, `CK` and `RN` match
  OpenROAD, while only `D` mismatches.
- Full top-10 DFS found no case where all inputs match but driver output
  mismatches. Frontiers are cycles (`156`) plus two depth-limit tails that are
  inside the same `debug_mode` cycle.
- X `power_zero_fanin_candidate` pins that are not actual seeds correlate with
  the large activity misses: `39608` candidate-not-seed pins present in
  OpenROAD, `3110` activity mismatches, `252` OR-nonzero/X-zero. Top rows are
  DFF/SDFF Q/QN, especially `i_perf_counters/perf_counter_q_reg[16][0:3]` and
  `csr_regfile_i/cycle_q_reg[0:5]`.
- Source read supports this direction: `power_enqueue_adjacent()` skips cell
  arcs into sequential Q/QN; Q/QN are updated by pending seq processing after a
  load pin changes. `power_zero_fanin_candidate` pins are only actual seeds
  when `XPLACE_POWER_SEED_POWER_LEVEL_ROOTS` or related debug envs are enabled,
  so default ariane can leave a closed seq feedback cone at zero if no external
  activity change marks the seq pending.
- Evidence-supported next hypothesis: ariane switching deficit is dominated by
  sequential feedback/candidate-not-seed activity semantics, not by root set
  mismatch and not by a proven local Liberty combinational eval mismatch.

### Ariane Hypotheses To Prove Next
- H1: OpenROAD and Xplace treat non-root sequential outputs with zero power
  fanin differently when they sit in activity feedback cycles. Prove/disprove
  by instrumenting one small cone (`debug_mode_q_reg` or `cycle_q_reg[2]`) and
  recording whether X ever updates Q/QN from D after D changes.
- H2: X has the right graph edges but skips enqueue/update for candidate-not-seed
  sequential outputs. Prove/disprove by tracing enqueue decisions for
  `debug_mode_q_reg/Q`, `cycle_q_reg[2]/Q`, and
  `perf_counter_q_reg[16][2]/Q`.
- H3: If X does update those Q/QN pins, then the mismatch is an activity formula
  or control condition difference for DFF/SDFF Q/QN; prove/disprove with a
  one-cell trace showing D/CK/RN/SE/SI inputs and computed Q/QN activity.

### Ariane Debug-Mode Loop Event Evidence 2026-05-20
- Event trace artifacts:
  `result/ispd25_power_debug_visible_ariane_eventtrace_20260520_debug_loop/`.
  Key files are `openroad_activity_path_trace_g87881.tsv`,
  `xplace_cpu_activity_path_trace_g87881.tsv`, and
  `trace_path_debug_mode_g87881.tsv`.
- Root/seed is not the direct source for this loop: in the OpenROAD root dump,
  `debug_mode_q_reg/Q`, `debug_mode_q_reg/D`, `g120510/ZN`, `g121412/ZN`,
  `g123841/ZN`, and `g128097/ZN` are probes only, `in_levelize_roots=0`,
  `was_seeded=0`. In Xplace, `debug_mode_q_reg/Q` is a
  `power_zero_fanin_candidate` but not an actual seed.
- OpenROAD first makes the loop nonzero in pass 7, not through Q first. The
  first positive chain is:
  `FE_OCPC16137_commit_instr_o_201/Z` becomes duty-only `0/1`, then
  `g87881/A2` becomes `0/1`, then with `g87881/A1=7.69231e7/0.5`
  OpenROAD evaluates `g87881/ZN=7.69231e7/0.5`. That drives
  `g128097/A2=7.69231e7/0.5`, `g128097/ZN=7.69231e7/0.5`, then the loop side:
  `g123841/A2/ZN -> g121412/A/ZN -> g120510/B1/ZN ->
  debug_mode_q_reg/D=6.48298e7/0.892815`.
- OpenROAD then marks `debug_mode_q_reg/D` pending in pass 7. In pass 8 it
  enqueues/visits `debug_mode_q_reg/Q` and evaluates it to
  `6.48298e7/0.892815`, after which Q drives the `FE_OFC6923 ->
  FE_OFC21521 -> g128097/A1` feedback side.
- Xplace CPU trace reaches the same pins/edges but never creates a positive
  update on this path. It sets `g87881/A1=7.69231e7/0.5`, but
  `FE_OCPC16137_commit_instr_o_201/Z`, `g87881/A2`, `g87881/ZN`,
  `g128097/A2`, `g128097/ZN`, `g120510/ZN`, `debug_mode_q_reg/D`, and
  `debug_mode_q_reg/Q` all remain zero-density. Xplace marks/seq-seeds
  `debug_mode_q_reg/Q` once, but the seq seed value is `0/0`, so the feedback
  loop is never dynamically activated.
- This refines the previous hypothesis: the local DFF Q formula is not the
  first observed divergence for this loop. The earliest observed divergence is
  event-order/iteration through a side path feeding `g87881/A2` and then
  `g128097/A2`. OpenROAD preserves a pass-7 nonzero intermediate long enough
  to activate the feedback loop; Xplace's ordered CPU propagation only observes
  the zero final value on that side path, so the loop stays zero.
- Follow-up trace
  `openroad_activity_path_trace_fe_rc10686.tsv` /
  `xplace_cpu_activity_path_trace_fe_rc10686.tsv` shows why
  `FE_OCPC16137_commit_instr_o_201/Z` becomes `0/1` only in OpenROAD. Both
  engines initially propagate `issue_stage_i/i_scoreboard/FE_RC_10686_0`
  inputs `A1/A2/A3/A4` to `0/1`. In OpenROAD pass 7, `A3` is revisited from
  upstream `FE_RC_10826_0/ZN` and changes `0/1 -> 0/0`; with the other NAND4
  inputs still `0/1`, `FE_RC_10686_0/ZN` evaluates to `0/1`. This duty-only
  state propagates through `FE_RC_8784_0/Z` and
  `FE_OCPC16137_commit_instr_o_201/Z` to `g87881/A2=0/1`, allowing
  `g87881/ZN=7.69231e7/0.5` because `g87881/A1=7.69231e7/0.5`. In OpenROAD
  pass 8, `A3` returns to `0/1` and the side path final state returns to
  `0/0`.
- Xplace does not miss the initial duty-only propagation: it also sets
  `FE_RC_10686_0/A1/A2/A3/A4` to `0/1`. The missing behavior is the later
  pass-7 `A3: 0/1 -> 0/0` revisit and the resulting `FE_RC_10686_0/ZN=0/1`
  intermediate. Therefore this evidence points to activity iteration/revisit
  order for duty-only changes, not a simple global duty initialization bug.

### Ariane FE_RC_10826 / mem_q[4]_sbe_valid Evidence 2026-05-20
- Continued upstream trace files:
  `openroad_activity_path_trace_fe_rc10826.tsv`,
  `xplace_cpu_activity_path_trace_fe_rc10826.tsv`,
  `openroad_activity_path_trace_memq4_sbe_valid.tsv`,
  `xplace_cpu_activity_path_trace_memq4_sbe_valid.tsv`,
  `openroad_activity_path_trace_g380355.tsv`, and
  `xplace_cpu_activity_path_trace_g380355.tsv` under
  `result/ispd25_power_debug_visible_ariane_eventtrace_20260520_debug_loop/`.
- `FE_RC_10826_0` is `AOI21_X4`. OpenROAD pass 7 drives
  `FE_RC_10826_0/ZN: 0/1 -> 0/0` because `B1` changes `0/0 -> 0/1` from
  `issue_stage_i/i_scoreboard/mem_q_reg[4]_sbe_valid/Q`, while `B2` is still
  `0/1`. This makes `FE_RC_10686_0/A3: 0/1 -> 0/0`. Xplace never produces this
  pass-7 `B1=0/1` event; its same Q and B1 stay `0/0`.
- The first OpenROAD nonzero source for that Q is a pending-seq step. In pass 6,
  `g380355/ZN` changes `0/1 -> 0/0`, so `FE_RC_10436_0/B2` changes
  `0/1 -> 0/0`, `FE_RC_10436_0/ZN` evaluates `0/0 -> 0/1`, and
  `mem_q_reg[4]_sbe_valid/D` changes `0/0 -> 0/1`. OpenROAD marks this D pin
  `seq_pending`; in pass 7 it seq-seeds `mem_q_reg[4]_sbe_valid/Q` to `0/1`.
  In pass 8, D and Q return to `0/0`.
- Xplace reaches the same DFF and combinational pins, but `FE_RC_10436_0/ZN`
  and `mem_q_reg[4]_sbe_valid/D` remain `0/0`. It performs a seq seed for
  `mem_q_reg[4]_sbe_valid/Q` only with `0/0`, so there is no later Q-to-B1
  activation for `FE_RC_10826_0`.
- The next upstream difference is `g380355`. OpenROAD pass 6 evaluates
  `g380355/ZN` to `0/0` because `g380355/A1` becomes `0/1` from
  `FE_OCPC16134_FE_DBTN55_n_56863/Z` while `g380355/A2` is still `0/1` from
  `id_stage_i/issue_q_reg_sbe_ex_valid/Q`. Xplace has the same relevant graph
  edges and does toggle `issue_q_reg_sbe_ex_valid/Q` in earlier passes, but by
  its pass 6 coarse trace `A1=0/0`, `A2=0/0`, `B1=0/1`, `B2=0/0`, and
  `g380355/ZN` remains `0/1`.
- Current evidence-supported hypothesis: the mismatch is an activity
  fixed-point scheduling/phase difference across sequential feedback, not a
  missing direct fanout, not missing `g87881/A1`, and not simple duty
  initialization. OpenROAD allows a short-lived aligned input combination
  across `id_stage_i/issue_q_reg_sbe_ex_valid/Q`,
  `FE_OCPC16134_FE_DBTN55_n_56863/Z`, and `FE_OCPC15824_n_56863/Z`; Xplace's
  ordering collapses or phases these updates differently, so the downstream
  `mem_q_reg[4]_sbe_valid/D/Q` activation never occurs.

### Ariane Full-Pin Activity Pass Snapshot 2026-05-20
- Added env-gated full-pin snapshot dump for activity propagation debug:
  OpenROAD uses `OR_POWER_ACTIVITY_SNAPSHOT_CSV` and
  `OR_POWER_ACTIVITY_SNAPSHOT_MAX_PASS`; Xplace CPU uses
  `XPLACE_POWER_ACTIVITY_SNAPSHOT_CSV` and
  `XPLACE_POWER_ACTIVITY_SNAPSHOT_MAX_PASS`. Both dump `after_seed/pass=0`,
  `after_comb/pass=0`, and `after_seq_seed`/`after_pass` for passes up to the
  configured max. This is dump-only and does not change propagation semantics.
- Added `tools/power_alignment/compare_activity_pass_snapshots.py`, which
  streams OpenROAD/Xplace snapshot CSVs by `pass,tag` and joins on
  `pin_name_norm` to produce full compare, per-snapshot summary, and first
  divergence CSVs.
- Smoke artifact:
  `result/ispd25_power_activity_snapshot_visible_ariane_20260520_195307/`.
  Both OpenROAD and Xplace snapshot CSVs have 6,690,531 lines: 14 snapshot
  points times 477,895 pins plus header. Summary shows all snapshot groups
  joined with no missing pins.
- Focused first-divergence evidence from the full snapshot confirms the earlier
  path-trace story: `id_stage_i/issue_q_reg_sbe_ex_valid/Q` first diverges at
  pass 2 `after_seq_seed`; `FE_RC_10009_0_dup1/A1` diverges at pass 3
  `after_pass`; `g380355/A2` diverges at pass 4 `after_pass`; and
  `FE_RC_10009_0_dup1/A2/ZN`, `g380355/A1/ZN` diverge at pass 6 `after_pass`.

### Ariane SDFFR Seq-Seed Timing Detail 2026-05-20
- For `id_stage_i/issue_q_reg_sbe_ex_valid` (`SDFFR_X2`), the liberty
  sequential next state is `(SE SI) + (!SE D)` and the physical output pin
  `Q` has `function: IQ`.
- OpenROAD `Power::seedRegOutputActivities` evaluates `seq.data()` and stores
  the result only in `seq_activity_map_` for the internal sequential output
  (`IQ`/`IQN`). It then enqueues physical output pins whose function references
  those internal pins. The visible physical `Q` pin activity is written later
  when the BFS visitor evaluates the `Q` output function `IQ`.
- Therefore an OpenROAD snapshot at `after_seq_seed` still shows the previous
  physical `Q` value; the newly seeded `IQ` value appears on physical `Q` at
  the following `after_pass` snapshot. Xplace CPU currently writes physical
  `Q/QN` directly during `seq_seed`, so `after_seq_seed` snapshots can look one
  phase earlier in Xplace even when both engines use the same liberty
  `next_state` expression.
- Concrete CSV evidence: at pass 1 `after_pass`, OPR and Xplace agree on
  `D=0, SE=1, SI=1, Q=0, QN=1`; at pass 2 `after_seq_seed`, Xplace already
  shows `Q=1, QN=0` while OpenROAD still shows the previous `Q=0, QN=1`; at
  pass 2 `after_pass`, OpenROAD physical `Q/QN` catches up and both engines
  agree at `Q=1, QN=0`.

### Ariane Automated Divergence Trace 2026-05-20
- Added `tools/power_alignment/trace_activity_divergence.py`. It parses the
  gate-level Verilog hierarchy, loads one `pass/tag` group from
  `activity_snapshot_compare.csv`, and walks from a target pin to its driver
  cone. For each driver cell it prints the output activity plus all input
  activities, and follows only `DEBUG_MISMATCH` inputs so clock/RN numerical
  density differences do not hijack the trace.
- Running it from `id_stage_i/issue_q_reg_sbe_ex_valid/SE` at its first
  divergence (`pass=3 after_pass`) reproduces the manual path through
  `id_stage_i` buffer/inverter fanout, `issue_stage_i/i_scoreboard`, and
  `issue_stage_i/i_issue_read_operands`, then into
  `ex_stage_i/lsu_i/lsu_bypass_i/status_cnt_q_reg[0]`.
- The earlier source found by re-running from
  `ex_stage_i/lsu_i/lsu_bypass_i/status_cnt_q_reg[0]/D` at its first
  divergence is `ex_stage_i/lsu_i/i_store_unit/g26402/ZN` at
  `pass=0 after_comb`. `g26402` is `NAND3_X1 (.A1(dtlb_hit_i), .A2(n_65),
  .A3(n_102), .ZN(n_110))`. At that snapshot all three inputs are aligned at
  duty 0, but OpenROAD evaluates `ZN` to duty 1 while Xplace leaves `ZN` at
  duty 0/origin 0. This is the first current evidence of an
  input-aligned/output-divergent combinational cell and points to Xplace not
  visiting/evaluating this initially-zero-input NAND cone, rather than an
  upstream activity mismatch.
- Code inspection of this divergence: OpenROAD's visitor evaluates a driver
  output whenever that output vertex is visited. Its `evalActivity()` uses the
  liberty function BDD; for `NAND3_X1`, `ZN = !((A1 & A2) & A3)`, so
  `A1=A2=A3=0` gives `ZN duty=1` and density 0. OpenROAD also treats an
  origin-only change as `changed`, and a net sink copy from an unknown/zero
  driver writes a propagated zero activity to the sink. In the snapshot,
  `g26402/A2` becomes propagated zero in OpenROAD at `pass=0 after_comb`; that
  is enough to enqueue/evaluate adjacent cell output `g26402/ZN`.
- Xplace's CPU path has an equivalent function evaluator, but it is lazy:
  outputs are evaluated only when their output pin is queued. `g26402/A2`
  remains origin 0 in Xplace, so no input-side changed event reaches the NAND
  output; `g26402/ZN` is never visited/evaluated and stays origin 0/duty 0.
  There is an `eval_cell_outputs` helper in the code, but it is not called in
  the current CPU propagation loop.

### Ariane Experiment Gate From First-6-Pass CSV 2026-05-21
- User-approved rule for the next experiment: use the full-pin snapshots for
  passes 0..6 as evidence before changing propagation behavior. Do not start
  from broad heuristics; each code experiment must be tied to a concrete
  first-divergence row and must be checked by regenerating Xplace snapshots
  against the same OpenROAD CSV.
- Evidence CSVs:
  `result/ispd25_power_activity_snapshot_visible_ariane_20260520_195307/openroad_activity_pass_snapshots.csv`,
  `.../xplace_activity_pass_snapshots.csv`, and
  `.../compare/activity_snapshot_compare.csv`.
- The visible `SDFFR_X2` `Q` mismatch at pass 2 `after_seq_seed` is not the
  root cause by itself. OpenROAD seeds internal `IQ/IQN` first and physical
  `Q/QN` appears after the BFS output-function visit; Xplace writes physical
  `Q/QN` directly during seq seed. The same CSV shows `Q/QN` realign at pass 2
  `after_pass`, so this is a phase artifact unless it causes a later persistent
  divergence.
- The current earliest actionable source from the automated mem-style walk is
  `ex_stage_i/lsu_i/i_store_unit/g26402/ZN` at pass 0 `after_comb`, reached
  from `id_stage_i/issue_q_reg_sbe_ex_valid/SE -> ... ->
  ex_stage_i/lsu_i/lsu_bypass_i/status_cnt_q_reg[0]/D`. At that snapshot:
  `g26402/A1`, `g26402/A2`, and `g26402/A3` all have aligned activity
  `density=0,duty=0`, while OpenROAD has `g26402/ZN density=0,duty=1,origin=5`
  and Xplace has `density=0,duty=0,origin=0`.
- This supports the next concrete hypothesis: Xplace CPU's initial propagation
  misses some combinational output evaluations when all inputs are zero/unknown
  and no input-side changed event enqueues the output, whereas OpenROAD's
  visitor can still evaluate the output through propagated-zero/origin-only
  changes. The first experiment is therefore limited to an initial
  combinational-output evaluation pass before `run_queue(0)`, skipping
  sequential cells, then checking whether `g26402/ZN`, `status_cnt_q_reg[0]/D`,
  `issue_q_reg_sbe_ex_valid/SE`, and switching power move toward OpenROAD.

### Ariane BDD Activity Alignment 2026-05-21
- Follow-up rule used here: keep tracing from the earliest/largest snapshot
  mismatch to its driver and inputs; only patch when there is an
  input-aligned/output-divergent cell or OpenROAD source-code evidence for a
  semantic difference.
- Verification reused the existing OpenROAD oracle CSV
  `result/ispd25_power_activity_snapshot_visible_ariane_20260520_195307/openroad_activity_pass_snapshots.csv`.
  OpenROAD was not rerun. Xplace-only rerun artifacts:
  `result/ispd25_power_activity_snapshot_visible_ariane_dutyfloor_p6_20260521_155613/`,
  `result/ispd25_power_activity_snapshot_visible_ariane_rootpolarity_p6_20260521_160849/`,
  `result/ispd25_power_activity_snapshot_visible_ariane_dutyulp_p6_20260521_161931/`,
  and final
  `result/ispd25_power_activity_snapshot_visible_ariane_bdd_p6_20260521_163634/`.
- Evidence chain:
  - Before this step, `after_pass` debug mismatches were down to pass 4 only.
    `g2169/ZN` (`AOI22_X1`) had all inputs debug-aligned, but OpenROAD
    density was `79.65814972` and Xplace was `86.49736023`. The extra Xplace
    term was `rho(A1) * duty(A2) ~= 6.839`, where `duty(A2)=2.07235e-08`.
  - A global tiny-duty floor was tested as a falsifying check and rejected:
    it removed the `AOI22` mismatch, but introduced `g2087/ZN` (`NAND3_X1`)
    mismatch where OpenROAD explicitly kept the same-size term. For `g2087`,
    inputs were debug-aligned and the missing term was
    `rho(A1) * duty(A3) ~= 6.466`.
  - OpenROAD source confirms the real difference:
    `Power::evalBddActivity()` computes `Cudd_bddBooleanDiff()` and then
    calls recursive `evalBddDuty()` on the BDD. Xplace was using minterm
    enumeration for Boolean-diff probability. CUDD complemented-edge
    representation plus float recursion can round the tiny `AOI22` diff to
    zero while still preserving the tiny `NAND3` diff. A scalar duty threshold
    cannot represent both cases.
- Code conclusion kept: Xplace CPU `evalPowerExprActivity()` now builds a
  small local ROBDD with CUDD-style complemented-edge normalization, evaluates
  output duty recursively, and computes density as OpenROAD does:
  `sum(input_density * eval_bdd_duty(boolean_diff_bdd))`.
- Final snapshot compare artifact:
  `result/ispd25_power_activity_snapshot_visible_ariane_bdd_p6_20260521_163634/compare/activity_snapshot_summary.csv`.
  Pass 0 `after_comb` `debug_mismatch_count` is 0, and `after_pass`
  `debug_mismatch_count` is 0 for passes 1, 2, 3, 4, 5, and 6. Remaining
  `after_seq_seed` debug mismatches are the previously understood OpenROAD
  internal `IQ/IQN` seq seed vs Xplace physical `Q/QN` direct seed phase
  artifact; each checked pass is realigned by the corresponding `after_pass`
  snapshot.

### Ariane Component Acceptance 2026-05-21
- After CPU activity pass snapshots aligned with OpenROAD, the remaining
  `visible/ariane` component failure was isolated to the CUDA activity
  propagation path used by `report_power_total_cuda()`, not to the power
  formulas or loads. Pre-fix full compare:
  `result/ispd25_power_component_visible_ariane_cuda_bdd_small_20260521_175447/`.
  Component rel_errs were internal `0.0047731343`, switching `0.0904443251`,
  leakage `0.0010244151`, total `0.0045549950`; only switching failed.
- Evidence from the worst switching instance before the fix:
  `ex_stage_i/lsu_i/i_mmu/i_ptw/g13720` (`AND2_X2`) had OpenROAD pins
  `A1/A2/ZN density=4.73031296e+08` with `ZN duty=0.0852682665`. Xplace CPU
  probe was close to OpenROAD (`ZN density=4.73579456e+08`,
  `duty=0.0853045508`), while Xplace CUDA final activity was high
  (`ZN density=5.92293248e+08`, `duty=0.0837583989`). The trace from
  `g13720/ZN` walks into the `state_q_reg[2] Q -> D -> ... -> Q` feedback
  loop, so the remaining component failure is a CUDA propagation/fixed-point
  scheduling mismatch after the CPU/OpenROAD semantics were aligned.
- Historical note: this run used the CPU-aligned activity tensor as the
  activity input for CUDA power kernels while the CUDA activity propagator was
  still under repair. As of the 2026-05-22 cluster fix, default power uses CUDA
  activity again (`XPLACE_POWER_USE_CPU_ACTIVITY_FOR_POWER` defaults false);
  set `XPLACE_POWER_USE_CPU_ACTIVITY_FOR_POWER=1` only for explicit CPU-vs-CUDA
  debugging.
- Post-fix acceptance run:
  `result/ispd25_power_component_visible_ariane_cpu_activity_power_20260521_181200/`.
  It reused the GNNTimer/OpenROAD golden CSV and generated
  `xplace_dump/visible_ariane_power.csv` plus
  `compare/visible_ariane_power_compare.csv`.
  Component rel_errs are internal `0.0002046660`, switching `0.0002368497`,
  leakage `6.6631435e-08`, total `0.0001832558`; all four
  `power_*_pass_1pct` flags and aggregate `power_pass_1pct` are true.
  Worst rows after the fix: internal/total
  `csr_regfile_i/mtvec_rst_load_q_reg` abs diff `4.33e-05`, switching
  `i_cache_subsystem/i_fifo_w_channel/status_cnt_q_reg[1]` abs diff
  `8.69e-08`, leakage `csr_regfile_i/mstatus_q_reg_sxl[1]` abs diff
  `6.84e-09`.
- The same code path also passes `blind/ariane` using the existing GNNTimer
  golden cache. Acceptance run:
  `result/ispd25_power_component_blind_ariane_cpu_activity_power_20260521_181431/`.
  Component rel_errs are internal `0.0002045169`, switching `0.0010880400`,
  leakage `1.5181409e-08`, total `0.0001233369`; all four
  `power_*_pass_1pct` flags and aggregate `power_pass_1pct` are true.
  Generated artifacts are `xplace_dump/blind_ariane_power.csv` and
  `compare/blind_ariane_power_compare.csv`.

### Mempool Cluster Floating-Load Root Acceptance 2026-05-22
- `visible/mempool_cluster` initially failed after template-expression and
  row-chunk fixes because a DMA burst cone stayed at zero. Worst switching
  instance was
  `gen_groups[0].i_group/gen_dmas[1].i_axi_dma_backend/i_axi_dma_data_mover/i_fifo_r_emitter/FE_OFC54589_read_req_o_9`
  (`INV_X2`): OpenROAD switching `0.0003336329`, Xplace switching `0`.
- Upstream evidence:
  `FE_OFC54589.../A` is driven by `i_axi_dma_burst_reshaper/g25728/Z`.
  OpenROAD has `g25728/Z=8.067841024e9/0.45161647`; default Xplace had
  `0/0`. Walking strongest mismatching inputs reached
  `gen_spill_reg.a_data_q_reg_deburst/SI`, a no-driver load pin that OpenROAD
  treats as an `input` activity root (`33.333336e6/0.5`) but Xplace left zero.
- Falsified alternative: default seeding every sequential self-feedback output
  activated too many roots (`seq_feedback=547652`) and drove visible cluster
  total to `26.9218627 W` vs OpenROAD `9.5206523 W`. Keep
  `XPLACE_POWER_SEED_SEQ_FEEDBACK_OUTPUTS` default false.
- Fix kept: CPU and CUDA activity seed no-driver load pins as
  `floating_load_input` roots. With seq feedback disabled, the target probe
  aligns: `mem_q_reg[0]_deburst/Q=1.93459392e8/0.08173846` and
  `g25728/Z=8.067840512e9/0.45161742`, matching OpenROAD within float noise.
- Acceptance run:
  `result/ispd25_power_component_cluster_visblind_floatroot_default_20260522_105500/`.
  `visible/mempool_cluster` rel_errs: internal `0.00044509`,
  switching `0.00116530`, leakage `4.19979e-06`, total `0.00052946`.
  `blind/mempool_cluster` rel_errs: internal `3.13810e-05`,
  switching `8.42914e-05`, leakage `2.69274e-06`, total `4.69215e-06`.
  Both cases have all four component pass flags and aggregate
  `power_pass_1pct=true`.
