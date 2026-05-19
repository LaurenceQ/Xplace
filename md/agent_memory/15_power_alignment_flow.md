# Power Alignment Flow
Scope: OpenROAD/OpenSTA power oracle and Xplace CUDA power compare.

## Verified 2441 Paths
- OpenROAD repo: `/research/d7/ascstd/qkduan25/GNNTimer/openroad`.
- OpenROAD bin: `.../openroad/build-check/bin/openroad`.
- Deps env: `/research/d7/ascstd/qkduan25/app/openroad-deps/env.sh`.
- Verified hash `31fc25440f489b2587a9b801c9cfcd06344a0178`; commands exist:
  `my_dump_graph`, `my_dump_pins`, `my_dump_power`.
- Xplace python: `/home/qkduan25/.conda/envs/gnn/bin/python` (`Python 3.10.19`).

## Current ISPD2025 Driver
- Use `tools/compare_ispd25_route_power_timing.py` for ISPD route-segment compare.
- Resolve golden cache to absolute path; OpenROAD cwd is benchmark root.
- Golden paths: `<cache>/openroad_dump/<split>_<design>_power*.csv` plus manifest.
- Valid ISPD golden requires all four power CSVs plus manifest; reject instance-only dumps.
- Do not use `report_power -format json`; use `my_dump_power` four CSVs:
  instance, pins, internal arcs, leakage.
- Use `--openroad-bin /research/d7/ascstd/qkduan25/GNNTimer/openroad/build-check/bin/openroad` and `--gpu 0`.
- Reuse golden unless DEF, SDC, libs, route_segments, OpenROAD bin, Tcl, or
  dump code changed; use `--force-openroad-golden` only then.

## Required Compare Semantics

- OpenROAD CSV is golden oracle only; Xplace must compute power itself.
- Compare by instance name, never by internal id.
- Report internal/switching/leakage/total plus worst instance/component.
- Accept only if internal, switching, and leakage are each within 1% of OpenROAD.
- Activity debug target: each pin density/duty ideally within 5% of OpenROAD.
- Mandatory debug order: first check all four OpenROAD CSVs; if missing, dump
  them with `my_dump_power`. If present, diff instance CSV first, then inspect
  worst pin/internal-arc/leakage CSV rows before changing code.
- Only after CSV diff identifies the failing component should source be read:
  switching -> load/activity code; internal -> arc/table/duty code; leakage ->
  when/PG/default code. Do not guess from total-only numbers.
- For activity mismatch, first use final CSVs and OpenROAD source semantics. If
  still unclear, instrument/probe `ensureActivities()` pass-by-pass and record
  target pin, pending register output, density/duty, and enqueue/update changes.
- Preserve OpenSTA semantics: switching load cap not Ceff; activity follows
  `ensureActivities()`; internal/leakage use Liberty table/when rules.

## Root Debug
- OR roots: `Power::seedActivities()`/`levelize_->roots()`; dump with
  `OR_POWER_DUMP_ROOTS_FILE`, optional `OR_POWER_ROOT_PROBE_PINS_FILE`.
- X dump: `XPLACE_POWER_DUMP_ROOTS_FILE`, optional
  `XPLACE_POWER_ROOT_PROBE_PINS_FILE`; compare via
  `tools/power_alignment/compare_power_roots.py`.
- Judge: OR root not X candidate => graph/fanin; X candidate not seed => seed
  selection; X seed but zero => seed/enqueue.
