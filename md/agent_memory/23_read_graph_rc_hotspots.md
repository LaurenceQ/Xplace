# Read / Graph / RC Hotspots

Date: 2026-06-01.

Scope: ISPD2025 direct route-segment timing/power runs after CUDA power
alignment. The bottleneck target is the non-power front half:
`read_input`, `preprocess_timing`, `build_timing_graph`, and `build_rc`.

## Instrumentation Added

- `XPLACE_IO_PROFILE=1`: C++ raw DB `Database::load()` and `setup()`
  phase timings.
- `DMP_RC_PROFILE=1`: route-segment RC substage timings around cache load,
  DMP initialization, RC kernels, and post-RC timing prep.
- `GPUTIMER_ROUTE_SEG_CACHE_PROFILE=1`: route cache hit path timings without
  disabling cache.

These are profile-only knobs. Default behavior is unchanged.

Build used:

```bash
cd /research/d7/ascstd/qkduan25/Xplace/build
conda run -n gnn make -j8
conda run -n gnn make install
```

Primary profile log:

```text
result/codex_stage_hotspot_20260601/visible_mempool_group_split_v2.log
```

## Mempool Group Breakdown

Case: `visible/mempool_group`, route cache hit, skip fanout 300.

`read_input` subparts:

```text
rawdb_load              42.773 s
  read_lef               0.064 s
  read_liberty           0.104 s
  read_def              40.229 s
  read_def_pg            2.376 s
rawdb_setup              1.022 s
  setup_regions          1.020 s
gpdb_setup              12.206 s
preprocess_design_info   1.037 s
PlaceData                0.003 s
```

Conclusion: `read_def` is the primary read hotspot. `gpdb_setup` is the
second read-side hotspot. LEF/lib parse is noise. `read_def_pg` is measurable
but not the main issue.

`preprocess_timing`:

```text
preprocess_timing        0.123 s
```

Conclusion: not worth optimizing first.

`build_timing_graph`:

```text
build_timing_graph      35.574 s
  read_sdc_json          7.878 s
  extract_timing_graph  16.581 s
    thresholds           1.254 s
    pin_name_map         1.618 s
    traverse_pins        2.658 s
    net_arcs             1.433 s
    cell_arcs            7.379 s
    pin_fanout_lists     0.471 s
    pin_arc_lists        0.938 s
    topology_tensors     0.622 s
  read_sdc_into_gtdb     9.091 s
  timer_init             0.985 s
  levelize               0.981 s
```

Conclusion: SDC parse/application costs 16.969 s total, slightly larger than
timing graph extraction. Inside extraction, `cell_arcs` is the largest single
phase.

`build_rc` cache-hit path:

```text
update_states                    0.001 s
init_dmp_rc_route_segments       2.919 s
  design_signature               2.011 s
  route cache binary read        0.632 s
  initialize_dmp_rc_explicit     0.184 s
  release_host_rc_graph          0.029 s
  calc_res_cap_dmp kernel        0.030 s
  propagate_rc_tree_dmp kernel   0.027 s
  prepare_timing_after_rc        0.002 s
```

Conclusion: RC is no longer kernel-bound. On cache hit, the largest RC cost is
the per-run net/pin name design-signature hash before cache validation. The
actual route cache read and GPU initialization are secondary.

## Large File Scale

Visible large cases:

```text
mempool_group.def                 2.2 GiB,  27,884,609 lines
mempool_cluster.def               9.1 GiB,  96,017,640 lines
mempool_group.route_segments      1.4 GiB,  33,484,669 lines
mempool_cluster.route_segments    5.1 GiB, 119,822,770 lines
mempool_group route cache         1.2 GiB
mempool_cluster route cache       4.1-4.3 GiB
```

Full sweep stage timings already showed the scaling:

```text
visible/mempool_group    read 55.322 s  graph 36.093 s  rc  3.589 s
visible/mempool_cluster  read 207.976 s graph 117.565 s rc 14.423 s
blind/mempool_group      read 56.304 s  graph 33.411 s  rc  3.274 s
blind/mempool_cluster    read 203.546 s graph 115.081 s rc 13.895 s
```

The profile ratios match file size scaling: cluster is mostly the same
hotspots magnified by much larger DEF/route/cache/name lists.

## Optimization Strategy

Priority 1: raw DEF/front-end cache or timing-only DEF path.

- The 40.229 s `read_def` on group scales to roughly 170-200 s on cluster.
- Best repeated-run win: persist a binary timer-design cache after
  `rawdb.load/setup + gpdb.setup + preprocess_design_info`, keyed by platform
  LEF/lib, DEF size/mtime, SDC size/mtime, route-segment skip policy, and
  code cache version.
- Best single-run structural win: add a timing/direct-route DEF reader that
  only builds component/pin/net/celltype/placement/layer data required by
  GPUTimer and route-segment pin mapping. It should skip placement-only
  constructs and power-grid/special-net bodies.
- Safe short-term win: skip `readDEFPG()` in direct route timing/power mode
  after validating it does not feed route-segment timing or power. It only
  saves 2.4 s on group but should save around 8-10 s on cluster.

Priority 2: timing-only GPDB setup.

- `gpdb_setup` is 12.206 s on group and likely around 45-55 s on cluster.
- Add a direct timing mode in GPDB setup that skips placement-only structures
  and names unless required by debug/reporting. Python already requests
  `include_names=False`; the C++ setup still pays for broader DB construction.

Priority 3: graph build cache and SDC fast path.

- SDC costs 16.969 s on group: `read_sdc_json` 7.878 s plus
  `read_sdc_into_gtdb` 9.091 s.
- Cache parsed/compiled SDC constraints by SDC file size/mtime and design
  signature, or fold the common ISPD SDC subset into direct C++ application.
- Optimize `cell_arcs` with per-libcell arc templates, then instantiate arcs
  per cell without repeated Liberty pointer/string lookup work.
- Longer-term: persist a timing graph cache after SDC application and graph
  extraction. This avoids both SDC and `cell_arcs` on repeated compare runs.

Priority 4: RC cache-hit cleanup.

- Remove or cheapen the per-run net/pin-name design signature on trusted direct
  route cache hits. Use route file size/mtime, DEF size/mtime, num nets/pins,
  unit fields, missing-fanout policy, and cache format version. This should
  cut about 2.0 s on group and roughly 7-8 s on cluster.
- Mmap or pinned-host route cache loading can help after that, but current
  binary read is only 0.632 s for a 1.2 GiB group cache; it is not first.

