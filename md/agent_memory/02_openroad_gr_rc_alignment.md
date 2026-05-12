# ISPD2025 OpenROAD GR Segment Timing Alignment

## Active Target

New target as of 2026-05-12:

Use OpenROAD global-route segment dumps from
`/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks`, read the same
segments into the Xplace/GPUTimer flow, and check whether no-CRPR GPUTimer
slack, WNS, and TNS can align with OpenROAD.

CRPR/CPPR is out of scope for this target. Disable CRPR in OpenROAD reference
runs before timing reports:

```tcl
sta::set_crpr_enabled 0
set ::sta_crpr_enabled 0
```

## Primary Data

Benchmark root:

```text
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks
```

Visible and blind design names:

```text
ariane
bsg_chip
NV_NVDLA_partition_c
mempool_tile_wrap
mempool_group
mempool_cluster
```

Important local files already observed:

- `ariane.route_segments_noclk`: existing OpenROAD-style global-route segment
  file at the benchmark root.
- `GR.tcl`: loads a design, runs `global_route`, and writes
  `${DESIGN_NAME}.route_segments`.
- `evaluate.tcl`: reads `${DESIGN_NAME}.route_segments`, runs
  `estimate_parasitics -global_routing`, then reports TNS/WNS/power.
- `openroad_evaluation.tcl`: official-style route evaluation script.
- `NanGate45/setRC.tcl`: RC setup used before global-route parasitic
  estimation.

Current contest route-file RC graph dumps also exist under:

```text
/research/d7/ascstd/qkduan25/contest25/rc_graph_dumps
```

Those dumps are route-file-derived CSVs, not the OpenROAD GR segment reference
itself. The active timing target is to match OpenROAD timing on the same
`read_global_route_segments` input.

## OpenROAD Reference

Prefer the standalone local OpenROAD binary for ISPD2025 reference runs:

```text
/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad
```

Observed version:

```text
26Q2-414-g5f65c482de
```

The GNNTimer OpenROAD fork also exists and supports the CRPR command:

```text
/research/d7/ascstd/qkduan25/GNNTimer/openroad/build/bin/openroad
31fc25440f489b2587a9b801c9cfcd06344a0178
```

Do not rely on `openroad` from `PATH`; it was not found in the current shell.
Use an absolute binary path and record which binary produced each result.

## First Validation Scope

Start with the smallest stable scope:

1. Visible `ariane`.
2. OpenROAD reference using the benchmark scripts and
   `ariane.route_segments_noclk` or a freshly generated `ariane.route_segments`.
3. CRPR disabled.
4. Report OpenROAD WNS/TNS and critical endpoint slacks.
5. Read the same segment file into GPUTimer/DMP global-route RC flow.
6. Compare no-CRPR slack/WNS/TNS before touching broader cases.

Required comparison fields:

- OpenROAD `report_wns`
- OpenROAD `report_tns`
- OpenROAD critical `report_checks` endpoint slack
- GPUTimer/Xplace endpoint slack
- GPUTimer/Xplace WNS/TNS computed from the same endpoint set when possible

## Current Known State

Verified from local inspection:

- `ISPD2025_benchmarks/GR.tcl` uses `write_global_route_segments`.
- `ISPD2025_benchmarks/evaluate.tcl` uses `read_global_route_segments` and
  `estimate_parasitics -global_routing`.
- `contest25/evaluation/openroad_evaluation.tcl` reads
  `${OUTPUT_DIR}/${DESIGN_NAME}.route` with `read_global_route_segments`.
- Both local OpenROAD binaries support `sta::set_crpr_enabled`.
- No current active memory yet records a completed no-CRPR ISPD2025
  GPUTimer-vs-OpenROAD WNS/TNS comparison.

## Success Criteria

For each checked design, record:

- exact OpenROAD binary path and version,
- exact benchmark design path,
- exact segment/route file path,
- whether the segment file includes clocks,
- whether CRPR was disabled,
- OpenROAD WNS/TNS,
- GPUTimer WNS/TNS,
- max endpoint slack delta and representative endpoint examples.

Do not claim alignment from route-file parsing or RC graph loading alone.
Alignment means timing metrics agree on the same segment input with the same
CRPR-off semantics.
