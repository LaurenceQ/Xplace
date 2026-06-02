# Speed And Memory Status

Optimize end-to-end direct `--route_segments` Xplace time. Avoid rerunning
OpenROAD; use saved matrix runtimes as reference.

## Known Bottlenecks

- Small/mid cases are dominated by Python import, design load, timing DB setup,
  GPUTimer init, and route segment/cache overhead.
- Large cases are dominated by route segment parse/finalize and timing memory.
- GPU allocation can appear early, but that does not mean CUDA kernels are the
  bottleneck.

All-case matrix completed visible/blind `mempool_cluster`:

```text
Xplace wall about 678s/683s
peak sampled CPU RSS about 169.7/171.5 GiB
peak sampled GPU memory about 77.3/77.7 GiB
```

2026-05-24 route-power cluster reruns now complete under occupied GPU1 after
the DMP memory reductions and release-before-power path. Current measured
peaks with another user's process resident:

```text
visible/mempool_cluster: wall 769s, peak RSS 128.98 GiB, peak GPU 47.85 GiB
blind/mempool_cluster:   wall 902s, peak RSS 129.90 GiB, peak GPU 48.00 GiB
```

Both runs used default CUDA power and `--no-instance-power-csv`; the extra
OpenROAD CSV group-oracle reconstruction took about 67s/70s on CPU.

- Removed dead per-slot thresholds and waveform scratch arrays
  (`k*`, `p*`, `A/B/D`, `rd/t0/dt`, `ceff`, `vo_*`, `dmp_alg_kind`).
- Replaced per-pin/per-timing threshold copies with pin/timing library-id
  maps plus compact per-library threshold arrays.
- `set_driving_cell` now stores only source metadata; direct-net computes
  virtual local state on the fly.
- Compact endpoint pin slack storage.
- Skip direct-route ref/ratio and state-backup tensors.
- Defer timing scratch until after RC propagation.
- Release transient RC graph arrays after GPU upload.
- Do not re-enable default debug/profile output.

Automatic direct-route thread lift was removed. `run_timer.py` now leaves
`--num_threads` at the normal CLI/default value unless explicitly set.

## 2026-05-30 Power Speed

Targeted power-stage speedup without changing the `xplace_power_s` measurement
scope in `tools/compare_ispd25_route_power_timing.py` (`report_power_total_cuda`
inside the existing `time_stage("power", ...)`).

Profiler evidence on `visible/mempool_group`:

```text
result/codex_power_profile_visible_mempool_group_20260530
component_rows 25.886s
launcher       11.721s
chunk_components 28.641s
XPLACE_STAGE power 79.142s
```

Implemented low-risk host-side grouping cleanup in CUDA power input build:

- internal-power denom groups now use packed `(to_pin, pg_id)` `uint64_t`
  keys instead of constructing `"pin|pg"` strings for millions of rows;
- leakage groups now use packed `(node_id, pg_id)` `uint64_t` keys instead of
  `"node|pg"` strings;
- env-gated `XPLACE_POWER_PROFILE_STAGES=1` remains available for host-side
  power staging diagnostics and is off by default;
- power CUDA launch checks default to launch-error checks, with
  `XPLACE_POWER_CUDA_SYNC_CHECKS=1` preserving per-kernel sync checks for
  first-failure debug.

Validation:

```text
result/codex_power_speed_visible_mempool_group_key64_20260530
visible/mempool_group: pass=True timing=True power=True groups=True
OpenROAD power 752.294081s, Xplace power 72.821018s, speedup 10.3307x
WNS rel_err 9.77207e-05, TNS rel_err 5.45869e-04
power rel_errs: internal 6.48290e-05, switching 1.81402e-04,
leakage 2.90355e-07, total 1.03317e-04
worst group/component: clock.internal 6.13274e-04
```

Smoke regression:

```text
result/codex_power_key64_visible_ariane_20260530
visible/ariane: pass=True timing=True power=True groups=True
```

All blind mid cases passed timing, but none hit 4x speedup:

```text
ariane 1.07x, bsg_chip 2.55x, NV_NVDLA_partition_c 1.61x
mempool_tile_wrap 1.11x, mempool_group 3.51x
```

Later 2026-05-30 power work reached the requested large-case target without
changing the `xplace_power_s` measurement scope:

- Nsight Systems on `visible/mempool_group` showed the remaining CUDA hotspot
  after component-row/chunk cleanup was activity propagation:
  `power_visit_level_kernel` took about 10.83s over 12682 launches.
- A host-driven level queue prototype was correct but slower
  (`mempool_group` power 33.02s, launcher 14.54s), so it was removed from the
  default path.
- The effective fix was a direct activity expression evaluator for expressions
  whose variables are unique. It propagates `(density, duty)` directly with
  boolean algebra and falls back to the existing BDD path for repeated-variable
  expressions. This avoids most BDD work in `power_visit_level_kernel` and
  `power_seed_seq_kernel`; `power_eval_expr_activity` now reports stack 0 in
  `cuobjdump`, while fallback/component expr kernels can still show unknown
  stack when they call the BDD path.

Clean installed validation:

```text
result/codex_power_directactivity_clean_visible_mempool_group_20260530
visible/mempool_group: pass=True timing=True power=True groups=True
OpenROAD power 752.294081s, Xplace power 22.169988s, speedup 33.933x
WNS rel_err 9.77207e-05, TNS rel_err 5.45869e-04
power rel_errs: internal 1.21990e-03, switching 2.27144e-03,
leakage 2.03775e-06, total 1.52537e-03
worst group/component: combinational.internal 3.48331e-03
stage highlights: launcher 2.902961s, chunk_components 1.556311s,
component_rows 5.802583s, peak GPU 16.56 GiB
```

Smoke after cleanup/install:

```text
result/codex_power_directactivity_clean_visible_ariane_20260530
visible/ariane: pass=True timing=True power=True groups=True
```

## 2026-05-31 Power Large Target Complete

Goal was to reach 50-60x power-stage speedup for the large mempool cases
without changing the `xplace_power_s` measurement scope, without caching, and
while keeping timing/power/group/component checks within the existing 1% gate.

Additional Nsight Systems evidence:

```text
result/nsight_power_20260531/visible_mempool_group_final.nsys-rep/sqlite
visible/mempool_group: XPLACE_STAGE power 13.784289
power_visit_level_kernel remained the top GPU kernel bucket.
CUDA API time was still launch/sync/allocation heavy: about 1.87M launches,
with cudaLaunchKernel, cudaDeviceSynchronize, cudaMalloc, and cudaMemcpy as
the main API buckets.
```

Implemented final low-risk speedups:

- `PowerCudaInputRows.cpp`: internal denom grouping now builds local
  per-output-pin pg maps, and leakage grouping builds local per-node pg maps.
  This removes millions of repeated global unordered-map lookups while keeping
  the same row semantics.
- `PowerCudaInputBuild.cpp`: default power row chunk budget increased from
  2 GiB to 8 GiB. Env overrides still work:
  `XPLACE_POWER_ROW_CHUNK_BYTES`,
  `XPLACE_POWER_INTERNAL_ROW_CHUNK_BYTES`, and
  `XPLACE_POWER_LEAKAGE_ROW_CHUNK_BYTES`.
- `PowerActivityHostUtils.cpp`: `classifyPowerPins` and clock-gate map build
  now precompute libcell/port flags and use array lookups in the node-pin loop.

Build/install:

```text
cd /research/d7/ascstd/qkduan25/Xplace/build
conda run -n gnn make -j8
conda run -n gnn make install
```

Final validation command:

```text
conda run -n gnn env XPLACE_POWER_PROFILE_STAGES=1 \
  XPLACE_POWER_PRINT_PASS_STATS=1 \
  python tools/compare_ispd25_route_power_timing.py \
    --out result/codex_power_final12_pinmap_hash8g_20260531 \
    --openroad-golden-cache result/ispd25_route_power_openroad_double_cache_20260524 \
    --openroad-ref-out result/ispd25_route_power_openroad_double_cache_20260524 \
    --skip-openroad --threads 16 --gpu 1 \
    --missing-fanout-skip auto --no-instance-power-csv
```

Final 12-case result: all pass timing, total power, group power, and
component/group 1% checks.

```text
visible ariane                 pass speed= 6.72x xpower= 3.060s
visible bsg_chip               pass speed= 2.78x xpower=58.448s
visible NV_NVDLA_partition_c   pass speed= 1.76x xpower=23.244s
visible mempool_tile_wrap      pass speed= 1.42x xpower=18.715s
visible mempool_group          pass speed=56.83x xpower=13.238s
visible mempool_cluster        pass speed=56.19x xpower=46.400s
blind   ariane                 pass speed= 4.02x xpower= 5.182s
blind   bsg_chip               pass speed= 2.35x xpower=66.452s
blind   NV_NVDLA_partition_c   pass speed= 1.96x xpower=18.512s
blind   mempool_tile_wrap      pass speed= 1.44x xpower=18.245s
blind   mempool_group          pass speed=56.34x xpower=13.257s
blind   mempool_cluster        pass speed=53.86x xpower=48.341s
```

Large-case stage highlights:

```text
case                      pin_maps component_rows uploads levelize launcher power     peak_gpu
visible/mempool_group        0.777          3.266   0.886    1.468    2.588 13.238s  18.790 GiB
visible/mempool_cluster      4.782         11.990   2.942    5.075    6.619 46.400s  49.342 GiB
blind/mempool_group          0.764          3.193   1.002    1.682    2.619 13.257s  18.750 GiB
blind/mempool_cluster        4.791         12.036   4.041    5.869    6.029 48.341s  49.518 GiB
```

Accounting pitfall: `[power_stage_profile]` is emitted inside
`compute_power_activity_cuda`, not around the whole Python
`XPLACE_STAGE power` timer.  For small/mid cases the dominant label is
`auto_cpu_activity_for_power`; omitting it makes the substage sum look wrong by
tens of seconds.  Even with every label included, `XPLACE_STAGE power` also
contains the post-return `inst_total_cpu = internal + switching + leakage`
tensor addition and the Python `time_stage` CUDA sync, so a small residual
remains.  In the final 12-case run the residual was about 0.02-0.14s for
small/mid, 0.43-0.45s for mempool_group, and 1.32-1.37s for mempool_cluster.

Worst group/component relative errors stayed well below 1%:

```text
visible/mempool_group:   combinational.internal 0.00348331
visible/mempool_cluster: sequential.switching   0.00273900
blind/mempool_group:     clock.internal         0.000566828
blind/mempool_cluster:   sequential.switching   0.00135132
```

`cuobjdump --dump-resource-usage` after the final build:

```text
power_eval_expr_activity: REG 116 STACK 0
power_eval_expr_diff_duty: REG 102 STACK 0
power_eval_expr_duty: REG 109 STACK 0
power_internal_*_fast_kernel: stack 0
power_leakage_row_fast_kernel: stack 0
generic internal/leakage row kernels and activity kernels still show
STACK:UNKNOWN where the BDD/fallback path is reachable.
```

Do not compare `report_power_order_summary` against `xplace_power_s`: the
former is outside the measured power-stage scope used by the compare CSV.
For `blind/mempool_cluster` it still took 83.863s after `XPLACE_STAGE power`.

2026-05-14 DMP cleanup smoke passed after build/install:
visible ariane stayed `-0.510/-1456.030` after helper state refactor and
winner/file cleanup; latest smoke 14.17s wall, RSS 1863868 KB.
`cuobjdump` regs unchanged: gate 180, direct-net 189, driving-cell 170;
no stack/local spill. `DmpKernels.cuh` and `DmpWaveform.cuh` were merged into
`DmpCudaUtils.cuh`; `DmpBackward.cu` was merged into `DmpTiming.cu`.

2026-06-02 power CUDA activity struct/live-range refactor:

- Checkpoint before refactor: `a154221 Checkpoint power and timing refactor state`.
- Power device helpers were moved into struct/member ops like the DMP style:
  `PowerActivityOps`, `PowerLevelQueueOps`, `PowerExprView`, BDD/direct expr
  evaluator structs, component expr/contrib ops.
- Deep duplicate storage was removed. `PowerLevelQueueOps` now inherits
  `PowerActivityOps` instead of carrying duplicate `model/scratch` fields.
  `PowerBddExprEval` and `PowerDirectExprEval` hold a `const PowerExprView*`
  instead of copying the view. BDD var pre-scan arrays were removed; the first
  pass inserts sorted vars directly into the context, and the second pass only
  calls `findVar()` so it cannot shift var indices after BDD nodes exist.
  `PowerComponentExprOps` holds one `PowerExprView` and only updates
  `node_id` per row.
- Remaining overlaps are shallow pointer views (`model`, `scratch`, queue/view
  pointers, expression ops/start/count, node-port pin maps). No duplicated large
  arrays or deep objects remain outside the BDD context's real working arrays.
- ptxas final spot checks:
  - Queue ordered/persistent kernels: 56 regs, 104/128B stack, 0 spill.
  - `enqueueMissingFuncOutputs`: 160B stack, 84/84B spill.
  - `processPinFrontier`: 120B stack, 44/44B spill.
  - Components slow denom/contrib kernels: 26/50 regs, 64B stack, 0 spill.
  - BDD fallback remains stack-dominated by `PowerBddContextCuda` arrays:
    fallback stack 20040B, 8/8B spill; removing pre-arrays did not change the
    ptxas max frame.
- Build/install passed in `gnn`.
- 12-case forced-CUDA compare passed:
  `result/codex_power_struct_refactor_pressure_final_12case_20260602`.
  Max WNS rel err `0.000498006084581` on `visible/mempool_tile_wrap`;
  max total-power rel err `0.000103499845952` on `visible/mempool_group`;
  max group/component rel err `0.00528588664294` on
  `blind/bsg_chip:combinational.internal`.
