# mempool_cluster GPU Memory Attribution

Date: 2026-05-28

Scope: ISPD2025 `visible/mempool_cluster` direct route-segment timing/profile
run. The run hit a CUDA illegal memory access during endpoint slack summary, so
do not use that run's WNS/TNS. Memory attribution is still usable because the
profile covered GTDB tensor upload, RC graph construction, DMP RC propagation,
and timing scratch allocation before the failing summary phase.

## Artifacts

- Log:
  `result/codex_mempool_cluster_mem_profile_20260528/all_case/xplace_logs/visible/mempool_cluster.direct_route.log`
- CSV:
  `result/codex_mempool_cluster_mem_profile_20260528/xplace_openroad_all_case_matrix.csv`
- Profile command:
  `DMP_INIT_MEM_PROFILE=1 DMP_RC_PROFILE=1 python tools/run_ispd25_all_case_matrix.py --skip-openroad --split visible --design mempool_cluster --out result/codex_mempool_cluster_mem_profile_20260528 --threads 8 --gpu 1 --sample-interval 1 --timeout-min 40 --xplace-profile`

## Design Scale

- Route raw lines: 119,822,770
- Route segment rows: 84,539,461
- Final route RC graph: 138,456,626 nodes, 126,402,690 edges
- DMP timing graph: 43,896,051 pins, 12,712,714 nets, 180,923,409 arcs,
  6,129,924 tests, 175,584,204 pin slots
- Profile sampled peak process GPU memory: 44.81 GiB

## Logged GPU Memory Stages

- `before_topology_tensors`: 5.177 GiB used
- `after_topology_tensors`: 9.212 GiB used, +4.035 GiB
- `after_liberty_tensors`: 12.077 GiB used, +2.865 GiB
- `after_state_tensors`: 23.358 GiB used, +11.281 GiB
- `after_clock_tensors`: 24.603 GiB used, +1.258 GiB
- `GPUTimer::initialize after_core_cuda_mallocs`: 33.104 GiB used,
  +8.501 GiB before arcSlew removal
- `after_rc_transient_free`: about 33.001 GiB used
- `after_timing_scratch_alloc`: about 34.310 GiB used

## Per-Array Conclusions

### GTDB topology tensors, about 4.04 GiB

These are allocated in `GTDatabase::ExtractTimingGraph`.

| Array/group | Approx size | Current role | Recommendation |
| --- | ---: | --- | --- |
| `pin_forward_arc_list` | 0.674 GiB | Forward propagation, gate fanout lookup, power graph. | Do not delete. Candidate to release only after timing/power no longer need graph topology. |
| `pin_backward_arc_list` | 0.674 GiB | Test/backward fanin and endpoint constraint propagation. | Do not delete. |
| `timing_arc_from_pin_id` | 0.674 GiB | Arc-to-source pin mapping; DMP propagation/winner/path need it. | Do not delete. |
| `timing_arc_to_pin_id` | 0.674 GiB | Arc-to-sink pin mapping. | Do not delete. |
| `pin_fanout_list` | 0.674 GiB | Levelization by pin fanout. | Candidate: release after level/schedule build in direct timing mode, but power paths also use it. |
| `pin_forward_arc_list_end` | 0.164 GiB | CSR offset for forward arcs. | Do not delete. |
| `pin_backward_arc_list_end` | 0.164 GiB | CSR offset for backward arcs. | Do not delete. |
| `pin_fanout_list_end` | 0.164 GiB | CSR offset for fanout pins. | Candidate: release with `pin_fanout_list` after levelization. |
| `pin_num_fanin` | 0.164 GiB | Levelization indegree scratch. | Candidate: release after levelization if no later power/topology build uses it. |

No large clearly-dead topology tensor was found. The best topology cleanup is a
lifetime reduction for `pin_fanout_list`, `pin_fanout_list_end`, and
`pin_num_fanin`, not a simple global deletion.

### Liberty/test metadata, about 2.87 GiB

| Array/group | Approx size | Current role | Recommendation |
| --- | ---: | --- | --- |
| `timing_arc_id_map` | 1.348 GiB | Arc/early-late to liberty timing id; gate delay core. | Do not delete. |
| `arc_types` | 0.674 GiB | Net arc versus gate arc. | Do not delete, but compress from `int32` to `uint8_t` if kernels are updated. Saves about 0.5 GiB. |
| `arc_id2test_id` | 0.674 GiB | Arc to setup/hold test id. | Do not delete. Candidate sparse/packed representation because only 6.13M tests exist for 180.9M arcs. |
| `test_id2_arc_id` | 0.023 GiB | Endpoint slack/report test lookup. | Keep. |
| `test_id2_endpoint_id` | 0.023 GiB | Compact endpoint index. | Keep. |
| `endpoints_id`, `endpoint_unique_pin_ids`, `primary_output2_endpoint_id` | small | Endpoint slack/path/report. | Keep unless endpoint summary is fully disabled. |
| `dmp_*thresholds`, `dmp_*library_ids` | small/medium | DMP gate model metadata. | Keep. |

Best low-risk metadata win: compress `arc_types`. Larger but riskier win:
sparse or packed `arc_id2test_id`.

### GTDB state tensors, about 11.28 GiB

| Array/group | Approx size | Current role | Recommendation |
| --- | ---: | --- | --- |
| `pinSlew` | 0.654 GiB | Slew result and DMP gate input. | Do not delete. |
| `pinLoad` | 0.654 GiB | Load/ceff used by DMP RC/gate/power. | Do not delete. |
| `pinAT` | 0.654 GiB | Arrival time result. | Do not delete. |
| `pinRAT` | 0.654 GiB | Required time/slack result. | Do not delete. |
| `pinImpulse` | 0.654 GiB | Old RC/old propagation waveform path. | Direct DMP route candidate: conditionally skip allocation. |
| `pinRootDelay` | 0.654 GiB | Old RC/old propagation root delay. | Direct DMP route candidate: conditionally skip allocation. |
| `at_prefix_pin` | 0.654 GiB | Path predecessor pin. | Keep for path report. Candidate optional for WNS/TNS-only mode. |
| `at_prefix_arc` | 0.654 GiB | Path predecessor arc and driving-cell source tag. | Not safe to simply delete. For WNS/TNS-only mode, first move source tag elsewhere. |
| `at_prefix_attr` | 0.654 GiB | Path predecessor attr/source sentinel. | Same as above. |
| `arcDelay` | 5.39 GiB | Core arc delay state. | Keep. User confirmed it is core. |

Best direct-route state win: skip `pinImpulse` and `pinRootDelay`, about
1.31 GiB. Bigger optional win: make `at_prefix_*` disabled in no-path-report
mode, about 1.96 GiB, after separating the driving-cell source tag.

### GPUTimer core mallocs

| Array/group | Approx size | Current role | Recommendation |
| --- | ---: | --- | --- |
| `pinCap` | 0.981 GiB | Liberty/input pin capacitance used by DMP RC. | Do not delete. |
| `pinWireCap` | 0.654 GiB | Old RC wire cap path. | Direct DMP route candidate: conditionally skip allocation. |
| `pinRootRes` | 0.654 GiB | Old RC root resistance path. | Direct DMP route candidate: conditionally skip allocation. |
| `testRelatedAT` | 0.091 GiB | Setup/hold related clock AT. | Keep. |
| `testRAT` | 0.091 GiB | Test RAT. | Keep. |
| `testConstraint` | 0.091 GiB | Setup/hold constraint. | Keep. |
| `net_is_clock` | 0.047 GiB | Clock net flag. | Candidate: avoid upload or release after DMP `pin_flags` are built. |
| `pin_is_clk` | 0.164 GiB | Clock pin flag. | Candidate: replace with packed `pin_flags`. |
| `pin_is_ideal_clk` | 0.164 GiB | Ideal clock flag. | Candidate: replace with packed `pin_flags`. |
| `level_list` | 0.164 GiB | Levelized timing schedule. | Keep. |
| `primary_outputs` | small | Primary output endpoints. | Keep. |

Best direct-route core win: skip `pinWireCap` and `pinRootRes`, about
1.31 GiB. Clock flag cleanup can save another roughly 0.3 GiB.

### SDC/clock tensors, about 1.0-1.2 GiB

| Array/group | Approx size | Current role | Recommendation |
| --- | ---: | --- | --- |
| `pin_clock_slews` | 0.654 GiB | Ideal clock slew override. | Do not delete; convert dense per-pin array to sparse clock-pin table. |
| `pin_clock_rise_edges` | 0.164 GiB | Ideal clock rise edge. | Sparse clock-pin table candidate. |
| `pin_clock_fall_edges` | 0.164 GiB | Ideal clock fall edge. | Sparse clock-pin table candidate. |
| `pin_clock_periods` | about 0.164 GiB if present | Per-pin period. | Sparse clock-pin table candidate. |
| `test_clock_periods`, `test_setup_uncertainties`, `test_hold_uncertainties` | about 0.069 GiB | Endpoint slack constraints. | Keep. |
| DMP `test_clock_ids` | 5.8 MiB | Test to clock id. | Keep. |
| DMP `clock_periods` | tiny | Clock periods. | Keep. |

The dense pin clock tensors are mostly NaN/unused. The right fix is sparse
storage, not deletion.

### DMP RC graph/build fields

| Array/group | Approx size | Current role | Recommendation |
| --- | ---: | --- | --- |
| `edge_from` | 0.471 GiB | RC build. | Already freed before propagation. |
| `edge_to` | 0.471 GiB | RC build. | Already freed before propagation. |
| `edge_res` | 0.471 GiB | Explicit RC build. | Already freed before propagation. |
| `flat_net2edge_start_map` | 0.047 GiB | RC build CSR. | Already freed before propagation. |
| `root_dist` | 0.516 GiB | RC traversal scratch. | Already freed before propagation. |
| `cnts` | 0.516 GiB | RC traversal scratch. | Already freed before propagation. |
| `flat_net2node_start_map` | 0.047 GiB | Per-net node CSR for propagation. | Keep until RC propagation completes. |
| `node2pin_map` | 0.516 GiB | RC node to pin mapping. | Keep until RC propagation completes. |
| `node_cap` | 2.063 GiB | Node capacitance per attr. | Keep; candidate for RC chunking. |
| `includes_pin_caps` | 0.012 GiB | Pin-cap inclusion flag. | Keep; too small to matter. |
| `node_order` | 0.516 GiB | Propagation order. | Keep until RC propagation completes. |
| `parent_node` | 0.516 GiB | RC tree parent. | Keep until RC propagation completes. |
| `res_parent` | 2.063 GiB | Parent edge resistance per attr. | Candidate: if direct route has attr-identical resistance, store one float per node and save about 1.55 GiB. |

Current build-only free is already useful and should stay. The next RC graph
memory win is compression or chunking, not deleting required fields.

### DMP RC propagation/timing scratch

| Array/group | Approx size | Current role | Recommendation |
| --- | ---: | --- | --- |
| `y1` | 2.063 GiB | RC moment/down-cap scratch. | Do not delete; candidate for chunking/reuse. |
| `y2` | 2.063 GiB | RC delay/moment scratch. | Do not delete; candidate for chunking/reuse. |
| `y3` | 2.063 GiB | RC higher-moment scratch. | Do not delete; candidate for chunking/reuse. |
| `elmore_delay` | 0.654 GiB | DMP gate eval reads it. | Keep. |
| `C1` | 0.654 GiB | Pi model cap. | Keep. |
| `C2` | 0.654 GiB | Pi model cap. | Keep. |
| `r_pi` | 0.654 GiB | Pi model resistance. | Keep. |
| `pin_at_winner` | 1.308 GiB | Atomic arrival-time winner scratch. | Keep. Candidate compression only if winner payload/path needs are redesigned. |

The largest transient peak is `y1/y2/y3` at about 6.19 GiB. This is an
algorithm/lifetime optimization target, not a safe deletion target.

## Recommended Optimization Order

1. Direct-route conditional allocation: skip `pinWireCap`, `pinRootRes`,
   `pinImpulse`, and `pinRootDelay`. Estimated saving: about 2.62 GiB.
2. WNS/TNS-only mode: make `at_prefix_pin`, `at_prefix_arc`, and
   `at_prefix_attr` optional after moving driving-cell source tags out of
   prefix arrays. Estimated saving: about 1.96 GiB.
3. Sparse SDC clock storage: replace dense per-pin `pin_clock_*` tensors with
   compact clock-pin arrays. Estimated saving: about 1.0 GiB.
4. Metadata packing: compress `arc_types` to `uint8_t`; consider sparse/packed
   `arc_id2test_id`. Estimated saving: about 0.5-1.1 GiB.
5. RC peak reduction: compress `res_parent` if attr-invariant, then consider
   chunking or reusing `y1/y2/y3`. Potential saving: several GiB, but higher
   implementation risk.

Do not remove without replacement: `pinSlew`, `pinLoad`, `pinAT`, `pinRAT`,
`arcDelay`, `timing_arc_id_map`, forward/backward arc CSR needed by timing,
`C1`, `C2`, `r_pi`, `elmore_delay`, and `pin_at_winner`.

## Implementation Status 2026-05-28

Implemented the first two requested memory optimizations:

1. Direct-route legacy RC tensor skip:
   - `GTDatabase::skip_legacy_rc_tensors` is set when `create_gputimer()` sees
     `route_segments` or `gr_rc` kwargs.
   - In that mode GTDB does not allocate `pinImpulse` or `pinRootDelay`, and
     also skips the reference/ratio timing tensors.
   - `GPUTimer::initialize()` does not allocate `pinWireCap` or `pinRootRes` in
     that mode.
   - `GPUTimer::update_states()` now guards those four pointers before reset.

2. DMP RC `res_parent` compression:
   - DMP RC now stores `res_parent` as one float per RC node instead of
     `NUM_ATTR` floats per node.
   - `calc_dmp_rc()`, `propagate_dmp_rc()`, and the DMP RC debug dump were
     updated to read/write the per-node value.

Validation:

- `make -j8` passed.
- `make install` installed the rebuilt `gputimer` extension to
  `cpp_to_py/cpybin`.
- Import check with `/home/qkduan25/.conda/envs/gnn/bin/python` passed.
- A direct-route smoke on `visible/mempool_tile_wrap` ran through route RC
  parse, DMP RC build/propagation, timing scratch allocation, and DMP timing
  body, printing `SMOKE_DIRECT_DMP_TIMING_DONE`.
- The all-case wrapper still fails in `report_timing_slack()` endpoint update
  with CUDA illegal memory access, which is the same post-timing endpoint slack
  summary caveat observed on `mempool_cluster`; it should not be used as a
  WNS/TNS validation for this change.



## Endpoint Device Fix and Level Profile 2026-05-28

The endpoint CUDA illegal memory access was reproduced on `visible/ariane` and
isolated to `GPUTimer::update_endpoints()`, not to DMP RC or DMP timing. With
`DMP_PROFILE_KERNELS=1` and `CUDA_LAUNCH_BLOCKING=1`, all DMP forward/backward
kernels synchronized successfully before `update_endpoints_kernel0` reported
`illegal memory access`.

Root cause fixed in `cpp_to_py/gputimer/core/timing/EndpointSlack.cu`:

- `update_endpoints()` now switches the current CUDA device to
  `timing_raw_db.pinAT.get_device()` before allocating endpoint tensors or
  launching raw CUDA kernels.
- `endpoints0`, `endpoints1`, and `endpoint_pin_slacks` are allocated on the
  same device as `pinAT` instead of using an unqualified `torch::kCUDA` device.
- Endpoint kernels now guard `arc_id`, `to_pin_id`, `pin_idx`, and compact
  `endpoint_id` bounds.
- With `GPUTIMER_ENDPOINT_DEBUG=1`, the run prints the endpoint tensor devices.

Validation:

- Build and install completed from `build/` using `cmake --build . -j8` and
  `make install`.
- `visible/ariane` direct-route run passed with endpoint devices all on GPU 1:
  `target=1 pinAT=1 pinRAT=1 endpoints0=1 endpoints1=1 endpoint_pin_slacks=1`.
- `visible/mempool_cluster` direct-route profile passed, with no
  `GPUTIMER CUDA`, `GPUassert`, `Traceback`, or `illegal memory` entries after
  the endpoint device fix.

Artifacts:

- Ariane validation log:
  `result/codex_ariane_endpoint_device_fix_20260528/all_case/xplace_logs/visible/ariane.direct_route.log`
- mempool_cluster level profile log:
  `result/codex_mempool_cluster_level_profile_20260528/all_case/xplace_logs/visible/mempool_cluster.direct_route.log`

mempool_cluster result:

- all-case status: `run`, pass: `pass`
- OpenROAD WNS/TNS: `(-0.52236, -87190.2)`
- Xplace DMP WNS/TNS: `(-0.523, -87403.5)`
- diff: `(0.122448%, 0.244738%)`

mempool_cluster level profile summary:

- levels: 259, first L0, last L258
- total pins across levels: 43,896,051
- total arcs across levels: 180,923,409
- total gate arcs: 149,740,072
- total net arcs: 31,183,337
- total direct net arcs: 1,083,502
- total gate-net pairs: 212,932,016
- total pair lanes: 1,703,456,128
- total valid pair lanes: 851,728,064
- total invalid pair lanes: 851,728,064
- scratch capacity: 351,168,408 items

Maxima by level:

| Metric | Max | Level |
| --- | ---: | ---: |
| pins | 3,703,100 | L3 |
| arcs | 5,150,242 | L3 |
| gate arcs | 4,685,536 | L38 |
| net arcs | 3,703,100 | L3 |
| direct net arcs | 1,082,808 | L1 |
| gate-net pairs | 7,406,260 | L2 |
| pair lanes | 59,250,080 | L2 |
| valid pair lanes | 29,625,040 | L2 |
| gate lanes | 37,484,288 | L38 |

Top valid-pair-lane levels:

| Level | pins | arcs | gate | net | direct | pairs | valid lanes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 | 2,212,336 | 4,425,379 | 4,424,638 | 741 | 694 | 7,406,260 | 29,625,040 |
| L38 | 259,141 | 4,685,536 | 4,685,536 | 0 | 0 | 6,531,230 | 26,124,920 |
| L36 | 244,065 | 3,996,020 | 3,996,020 | 0 | 0 | 6,221,594 | 24,886,376 |
| L40 | 247,359 | 4,607,424 | 4,607,424 | 0 | 0 | 6,042,786 | 24,171,144 |
| L34 | 223,044 | 3,423,108 | 3,423,108 | 0 | 0 | 5,701,930 | 22,807,720 |
| L42 | 233,359 | 4,466,274 | 4,466,274 | 0 | 0 | 5,560,384 | 22,241,536 |

DMP kernel profile highlights:

- DMP forward levels: 258, total 3785.326 ms, max 114.383 ms at L38
- DMP backward launches: 258, total 1632.567 ms, max 1535.264 ms at L0
- `dmpGateKernel`: 258 launches, 3686.806 ms total, max 113.627 ms at L38
- `dmpBackwardKernel`: 258 launches, 1632.567 ms total, max 1535.264 ms at L0
- `dmpTestKernel`: 258 launches, 40.806 ms total
- `dmpPinWinnerKernel`: 258 launches, 33.417 ms total
- `dmpNetWinnerKernel`: 188 launches, 10.732 ms total
- `dmpDirectNetKernel`: 2 launches, 3.058 ms total

Memory stages in the fixed mempool_cluster run:

- `after_state_tensors`: 22.050 GiB used
- `after_clock_tensors`: 23.294 GiB used
- `GPUTimer::initialize after_core_cuda_mallocs`: 25.095 GiB used
- `GPUTimer::initialize after_state_backups`: 25.104 GiB used
- `after_rc_transient_free`: 25.0 GiB used range, with 54.260 GiB free on a
  79.252 GiB GPU
- `after_timing_scratch_alloc`: 26.3 GiB used range, with 52.952 GiB free on a
  79.252 GiB GPU
