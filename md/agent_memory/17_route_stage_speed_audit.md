# Route Stage Speed Audit

## Stage Boundaries

- OpenROAD timed Tcl `tools/openroad_ispd25_route_power_timed.tcl`:
  `read_input` = Liberty/LEF/DEF/SDC/setRC/no-CRPR; `read_route_segments` =
  `read_global_route_segments`; `build_rc` = `estimate_parasitics -global_routing`.
- Xplace worker `tools/compare_ispd25_route_power_timing.py`: `read_input` =
  `load_design`; `build_timing_graph` = `GPUTimer(...)`; `build_rc` =
  `update_states`, `set_ideal_clock(true)`, `init_dmp_rc_route_segments`.
- Xplace has no separate read-route stage; route text/cache load, HostRcGraph
  finalize, GPU upload/init, DMP RC calc/propagation are all inside `build_rc`.

## mempool_group Runtime Evidence

- OpenROAD powerfix_group visible: read_input 43.590s, read_route 91.635s,
  build_rc 101.332s, timer 210.315s, power 746.055s.
- OpenROAD powerfix_group blind: read_input 42.853s, read_route 88.997s,
  build_rc 101.608s, timer 457.062s, power 734.362s.
- Xplace powerfix_group visible used `--missing-fanout-skip 0`: read_input
  60.611s, build_timing_graph 28.257s, build_rc 125.577s.
- Xplace auto/300 with cache hit: visible casefix build_rc 3.662s, blind
  powerfix/casefix build_rc 3.626/4.159s.
- Xplace auto/300 cold-ish visible latchfix build_rc 48.445s, likely created
  route cache; same case later hit cache at 3.662s.
- Segment files are similar: visible 33,484,669 lines, blind 33,411,704 lines,
  both 1.4G. Segment size does not explain visible/blind Xplace build_rc gap.

## Current Conclusions

- The 125s Xplace visible build_rc outlier is policy/cache, not visible route
  size: it used skip=0 against skip_fanout300 segments; mempool_group should use
  auto/300 unless debugging exact missing high-fanout behavior.
- Route-segment cache is enabled by default in `OpenroadGrRc.cpp`; cache key
  includes source size/mtime, design signature, and `missing_high_fanout_skip`.
  Current mempool_group v5 caches are skip=300 and about 1.2G each.
- OpenROAD spends about 190s in read_route+build_rc on mempool_group; Xplace
  cache-hit RC is about 4s and cold auto/300 is about 48s, but Xplace `read_input`
  is slower than OpenROAD by about 16-24s.

## Optimization Priority

- First guard the skip policy: avoid accidental `--missing-fanout-skip 0` for
  mempool_group/cluster saved skip_fanout300 segments.
- Add sub-stage profiling around Xplace `init_dmp_rc_route_segments` so
  parse/cache-load, graph finalize, GPU upload, DMP calc, and DMP propagation
  are not all reported as one `build_rc` number.
- Next speed target is Xplace `read_input`: it reparses Liberty/LEF/DEF and
  builds rawdb/gpdb every run; cache/narrower timing-only read path would help.

## 2026-06-03 No-Cache Read/Input And Route RC Speed Work

All accepted changes below were built from `build/`, followed by `make install`,
then checked with no route cache (`GPUTIMER_ROUTE_SEG_DISABLE_CACHE=1`). Smoke
case was `visible_ariane`; profile case was `visible_mempool_group` with
`XPLACE_IO_PROFILE=1`, `XPLACE_TIMER_PROFILE=1`, `DMP_RC_PROFILE=1`, and
`GPUTIMER_ROUTE_SEG_PROFILE=1`.

Accepted read-input / GPDB commits:

- `711db59 Avoid GPDB port map name copies`: changed `GPNode::portMap` to
  `std::unordered_map<std::string_view, int>` and borrowed macro pin names.
  Targeted `setup_nets` sub-stages improved despite noisy total `read_def`.
- `79d4246 Avoid bounds checks in GPDB net setup`: replaced hot-loop
  `nodes.at(...)` with `nodes[...]` inside validated `setupNets` ids.
  `visible_ariane` passed; `visible_mempool_group` profile showed
  `setup_nets=6.986`, `read_input=48.473`.
- `884dff7 Build GPDB pin index maps during net setup`: filled
  `pin_id2node_id`, `pin_id2net_id`, and `pin_names` during the existing
  parallel pin fill so `setupIndexMap` skips its redundant 12M-pin walk.
  Compared with `79d4246` profile: `setup_index_map 0.698 -> 0.344`,
  `read_input 48.473 -> 48.136`; `pin_fill` increased slightly as expected.

Accepted route-RC commits:

- `fc7460b Speed up route pin location lookup`: replaced per-pin map/vote
  structures with stack-backed one-box fast paths and precomputed routing
  level-to-layer lookup. This reduced targeted pin-location work, with total
  `build_rc` still noisy.
- `f98a651 Use flat adjacency for route RC finalization`: replaced large-net
  `vector<vector<int>>` repair adjacency with a flat adjacency and used the
  same large-net path in prune. On `visible_mempool_group`, compared with the
  prior committed profile: `repair_work 3.366 -> 2.337`,
  `prune_work 2.520 -> 1.732`, finalize wall `0.942 -> 0.817`, and
  `build_route_segments_graph 5.941 -> 5.678`.
- `11fd6d4 Copy route RC graph ranges during materialization`: copied
  contiguous `edge_res`, `node2pin`, and `node_cap` ranges directly during
  HostRcGraph materialization. Compared with `f98a651` profile:
  `append 1.432 -> 1.186`, `build_route_segments_graph 5.678 -> 5.338`.
  `initialize_dmp_rc_explicit` varied widely run-to-run and should not be used
  alone to judge this host graph construction change.

Rejected/reverted experiments:

- Route attach duplicate detection with small-net linear scans (thresholds 32
  and 8) was correct but did not improve `attach_work`; reverted.
- `CellType` pin-name index in `common/db/Cell` regressed `read_def`; reverted.
- Moving GPNode names/cell-type strings into `node_id2*` arrays reduced
  `setup_index_map 0.344 -> 0.188` but enlarged/touched GPNode enough that the
  same run regressed `setup_nets 7.083 -> 9.981` and `read_input 48.136 ->
  54.305`; reverted and reinstalled `884dff7` before continuing.
- Earlier attempts that should stay rejected: robin-hood route net-name map,
  raw DB secondary cell-name index, DEF connection marker check caching, and
  direct route trailing-integer parser.

Current no-cache `visible_mempool_group` profile after `11fd6d4`:

```text
read_input 49.115  (read_def noise: 38.266)
setup_nets 6.997, setup_index_map 0.334
build_net_name_map 1.010, map_pins_by_gpdb 1.154, scan_route_blocks 1.691
parse_segments 2.700, finalize_done 5.055, append 1.186
build_route_segments_graph 5.338
build_rc 7.501 (dominated by noisy initialize_dmp_rc_explicit=2.052 in this run)
```


Additional accepted read-input profiling and optimization:

- Added env-gated `XPLACE_DEF_PROFILE` section timing inside the DEF reader
  callbacks. It also turns on under `XPLACE_IO_PROFILE`, prints only when the
  env flag is enabled, and counts rows/tracks/gcells/vias/components/pins/
  blockages/specialnets/nets/net-connections/regions.
- Baseline profiler-only `visible_mempool_group` run:
  `read_def=37.218`, DEF setup `0.083`, components `11.787`, pins `2.118`,
  nets `23.223`, total DEF profile `37.211`, with `3,077,989` components,
  `3,503,992` nets, and `12,026,191` net connections.
- Skipped raw `Cell:Pin` string construction in `Pin::Pin(Cell*, int)` when
  `setting.SkipDefNetWires` is true. In route-segment direct-timing mode the
  Python driver already sets this flag, and GPDB later builds the exported pin
  names used by the timer. `Net::getPin()` now has a lazy fallback for the rare
  raw-DB caller that searches by full pin name.
- After skipping raw cell-pin names, `visible_ariane` no-cache still passed
  timing/power/group comparisons. `visible_mempool_group` no-cache profile:
  components `11.787 -> 9.445`, `read_def 37.218 -> 35.524`, and
  `read_input 47.983 -> 47.112`. The remaining DEF reader hotspot is the
  serial nets section at about `23.7s`.


Additional accepted DEF connection lookup optimization:

- Rejected a lazy `CellType` pin-name hash map experiment: it passed
  correctness, but the large no-cache profile was not a clear total win
  (`read_def=34.868`, `read_input=47.632`) and it added persistent map storage
  to every `CellType`.
- Kept the simpler indexed linear `Cell::pin(const char*)` lookup instead:
  scan the fixed `CellType::pins` vector by index, compare first character
  before `strcmp`, and touch the per-cell `_pins` vector only after the pin type
  name matches. This keeps the code readable and avoids per-cell/per-type maps.
- Validation after the simplified scan: `visible_ariane` no-cache passed timing,
  power, and group checks. `visible_mempool_group` no-cache passed with
  `read_input=45.579`, `read_def=35.652`, DEF components `9.382`, DEF nets
  `23.975`, `setup_nets=6.632`, `setup_index_map=0.354`,
  `build_route_segments_graph=4.659`, and `build_rc=6.642`. The route-RC
  numbers are noted as run-to-run noise for this read-input-only code change.


Accepted route-RC CUDA scaling optimization:

- Moved explicit route-segment `rc_time_factor` scaling out of the host loop
  and into `DmpRc.cu` as `scale_explicit_edge_res_kernel`. The host now copies
  raw edge resistance values to the GPU and scales `h_dmp_db->edge_res` in
  place before copying the `DmpModel` descriptor to device memory.
- `visible_ariane` no-cache passed timing/power/group checks after the change.
  `visible_mempool_group` no-cache also passed. The measured local RC init
  substage improved from `initialize_dmp_rc_explicit=1.848` in the prior
  profile to `0.271`; total `build_rc=6.770` in that run was still dominated
  by route graph construction noise (`build_route_segments_graph=6.387`).
