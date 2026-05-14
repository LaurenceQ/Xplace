# OpenROAD GR RC Alignment

## Target

Mainline timing input is saved OpenROAD global-route segment files:

```text
run_timer.py --route_segments <design>.route_segments
```

Do not use `--gr_rc` TSV as main timing input. It is only a debug oracle for
RC graph comparison.

## Reference Data

Benchmark root:

```text
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks
```

Current reference directories:

```text
openroad_gr_segments_skip_fanout300
openroad_gr_logs_skip_fanout300
openroad_gr_eval_crpr_off_skip_fanout300
```

Reference semantics:

- OpenROAD reads saved segments with `read_global_route_segments`.
- OpenROAD runs `estimate_parasitics -global_routing`.
- CRPR is disabled.
- Segment generation used `global_route -skip_large_fanout_nets 300`.

## Current Timing Status

The current all-case matrix says all 12 visible+blind cases pass the 1% timing
gate on the same saved skip-fanout300 segment input.

Matrix files:

```text
result/ispd25_direct_route_latest/xplace_openroad_all_case_matrix.md
result/ispd25_direct_route_latest/xplace_openroad_all_case_matrix.csv
```

Alignment means WNS/TNS and relevant endpoint evidence agree on the same route
segment input. RC graph construction alone is not enough.
