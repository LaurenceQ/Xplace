# Missing-High-Fanout Policy

Saved OpenROAD segments are under `openroad_gr_segments_skip_fanout300`.
Those files omit very high-fanout nets from OpenROAD routing. Xplace controls
missing-net fallback with `GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP`.

## Current Rule

```text
visible/NV_NVDLA_partition_c -> 0
blind/ariane                 -> 0
all other current cases      -> 300
```

Encoded in `tools/run_ispd25_all_case_matrix.py::missing_fanout_skip_value()`.

## Why

- `mempool_group/cluster`: skip `300` prevents huge fallback RC on absent
  clock nets with hundreds of thousands to over a million pins.
- `visible/NV_NVDLA_partition_c`: skip `300` removes needed fallback nodes for
  `nvdla_core_clk`/`nvdla_core_rstn`/`u_NV_NVDLA_cdma/nvdla_op_gated_clk_buffer`
  and falsely drifts to about `-0.430/-3213.686`.
- `blind/ariane`: skip `300` removes `clk_i`/`rst_ni` fallback nodes and
  falsely drifts to about `-0.460/-83.427`.
- `bsg_chip`: skip `300` is correct for timing; do not special-case to `0`.

## OpenROAD Missing Net Model

- `read_global_route_segments` only inserts nets present in the segment file
  into `GlobalRouter::routes_`; absent nets do not get fallback segments.
- `estimate_parasitics -global_routing` deletes old parasitics and builds RC
  only for non-empty `getRoutes()` entries; missing nets keep no GRoute RC.
- OpenSTA still sees logical net connectivity/pin caps. Xplace `skip=0`
  preserves receiver pin nodes with 0-ohm repair; `skip=300` lumps high-fanout
  missing sink caps at the driver and removes per-sink RC nodes.

## Evidence To Check

In route profile logs, check:

```text
missing_high_fanout_skip=<0|300>
skipped_missing_high_fanout_nets=<N>
skipped_missing_high_fanout_pins=<N>
```
