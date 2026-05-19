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
