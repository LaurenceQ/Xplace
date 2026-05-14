# Missing-High-Fanout Policy

Saved OpenROAD segments are in:

```text
openroad_gr_segments_skip_fanout300
```

Those files omit very high-fanout nets from OpenROAD routing. Xplace has a
separate policy for missing nets:

```text
GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP
```

## Current Rule

```text
visible/NV_NVDLA_partition_c -> 0
blind/ariane                 -> 0
all other current cases      -> 300
```

Encoded in:

```text
tools/run_ispd25_all_case_matrix.py::missing_fanout_skip_value()
```

## Why

- `mempool_group/cluster`: skip `300` prevents huge fallback RC on absent
  clock nets with hundreds of thousands to over a million pins.
- `visible/NV_NVDLA_partition_c`: skip `300` removes needed reset/clock-style
  fallback RC and falsely drifts to about `-0.430/-3213.686`.
- `blind/ariane`: skip `300` falsely drifts to about `-0.460/-83.427`.
- `bsg_chip`: skip `300` is correct for timing; do not special-case to `0`.

## Evidence To Check

In route profile logs, check:

```text
missing_high_fanout_skip=<0|300>
skipped_missing_high_fanout_nets=<N>
skipped_missing_high_fanout_pins=<N>
```
