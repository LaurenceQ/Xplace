# Local Timing Alignment Tools

## Active ISPD2025 Inputs

Benchmark root:

```text
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks
```

Contest repo:

```text
/research/d7/ascstd/qkduan25/contest25
```

Visible design inputs:

```text
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/visible/<design>/<design>.def
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/visible/<design>/<design>.v.gz
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/visible/<design>/<design>.sdc
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/visible/<design>/<design>.cap
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/visible/<design>/<design>.net
```

Existing route/segment artifacts observed:

```text
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/ariane.route_segments_noclk
/research/d7/ascstd/qkduan25/contest25/outputs/ariane_minimal_dump.route
/research/d7/ascstd/qkduan25/contest25/rc_graph_dumps/ariane_minimal/rc_segments.csv
/research/d7/ascstd/qkduan25/contest25/rc_graph_dumps/ariane_minimal/rc_edges.csv
```

## OpenROAD Binaries

Preferred for ISPD2025 reference runs:

```text
/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad
```

Observed version:

```text
26Q2-414-g5f65c482de
```

GNNTimer fork, available if custom dump commands are needed:

```text
/research/d7/ascstd/qkduan25/GNNTimer/openroad/build/bin/openroad
```

Observed version:

```text
31fc25440f489b2587a9b801c9cfcd06344a0178
```

Both support:

```tcl
sta::set_crpr_enabled 0
```

## OpenROAD Segment Reference Scripts

Existing benchmark scripts:

- `/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/GR.tcl`
  - loads one design,
  - runs placement parasitics,
  - runs `global_route`,
  - writes `${DESIGN_NAME}.route_segments`,
  - reports global-route timing.
- `/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/evaluate.tcl`
  - loads one design,
  - reads `${DESIGN_NAME}.route_segments`,
  - runs `estimate_parasitics -global_routing`,
  - reports TNS/WNS/power.
- `/research/d7/ascstd/qkduan25/contest25/evaluation/openroad_evaluation.tcl`
  - official-style route evaluation,
  - reads `${OUTPUT_DIR}/${DESIGN_NAME}.route` with
    `read_global_route_segments`,
  - reports TNS/WNS/power and critical checks.

Before repeated reference runs, create a stable CRPR-off wrapper instead of
generating throwaway Tcl in `/tmp`. The wrapper should:

1. accept design name, visible/blind split, and route/segment path,
2. source a single consistent Liberty/LEF setup,
3. load DEF and SDC,
4. source `NanGate45/setRC.tcl`,
5. disable CRPR,
6. run `read_global_route_segments`,
7. run `estimate_parasitics -global_routing`,
8. write `report_tns`, `report_wns`, and `report_checks` to stable logs.

Required CRPR-off Tcl lines:

```tcl
sta::set_crpr_enabled 0
set ::sta_crpr_enabled 0
```

## Xplace/GPUTimer Build

Use the normal Xplace build/install flow before running Python comparisons:

```bash
cd /research/d7/ascstd/qkduan25/Xplace/build
source ~/.bashrc
conda activate gnn
make -j8
make install
```

Then run the relevant Python entry point from Xplace. If a new ISPD2025 segment
comparison script is added, put it under:

```text
/research/d7/ascstd/qkduan25/Xplace/tools
```

## Legacy Sky130 Tools

The previous active sky130 no-CRPR CSV task is archived at:

```text
/research/d7/ascstd/qkduan25/Xplace/md/agent_memory/archive/sky130_no_crpr_2026-05-12
```

Do not use the sky130 CSV comparison tool as evidence for the active ISPD2025
GR segment target. It remains available only for returning to sky130 work:

```text
/research/d7/ascstd/qkduan25/Xplace/tools/compare_dmp_openroad_csv.py
```
