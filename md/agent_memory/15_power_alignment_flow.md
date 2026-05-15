# Power Alignment Flow
Scope: OpenROAD/OpenSTA power oracle and Xplace CUDA power compare.

## Verified 2441 Paths
- OpenROAD repo: `/research/d7/ascstd/qkduan25/GNNTimer/openroad`.
- OpenROAD bin: `.../openroad/build-check/bin/openroad`.
- Deps env: `/research/d7/ascstd/qkduan25/app/openroad-deps/env.sh`.
- Verified hash `31fc25440f489b2587a9b801c9cfcd06344a0178`; commands exist:
  `my_dump_graph`, `my_dump_pins`, `my_dump_power`.
- Xplace python: `/home/qkduan25/.conda/envs/gnn/bin/python` (`Python 3.10.19`).

## Legacy Bundle
- Reference: `/research/d7/ascstd/qkduan25/openroad_power_alignment_scripts_20260515_194001`.
- Prompt typo without underscores is wrong; old package is flow reference only.
- Stale paths found: `/data/Xplace`, `/data/GNNTimer`,
  `/home/lawrenced/anaconda3`, `sky130hd`, `/data/Xplace/netlists`.

## Current ISPD2025 Driver
- Use `tools/compare_ispd25_route_power_timing.py` for contest route-segment
  power/timing compare.
- Resolve golden cache to absolute path; OpenROAD cwd is benchmark root.
- Golden paths: `<cache>/openroad_dump/<split>_<design>_power*.csv`
  plus `<split>_<design>_manifest.json`.
- Valid ISPD golden requires all four power CSVs plus manifest; do not accept
  instance-only dumps as complete.
- Do not use `report_power -format json`; use `my_dump_power` four CSVs:
  instance, pins, internal arcs, leakage.
- Use `--openroad-bin /research/d7/ascstd/qkduan25/GNNTimer/openroad/build-check/bin/openroad`
  and `--gpu 0`.
- Reuse golden unless DEF, SDC, libs, route_segments, OpenROAD bin, Tcl, or
  dump code changed; use `--force-openroad-golden` only then.

## Legacy Script Porting Rules

- If using old GNNTimer benchmark scripts, copy to a work dir before edits.
- Correct env names are `OUT_ROOT`, `DESIGN_LIST`, `OPENROAD_BIN`,
  `GNNTIMER_DIR`, `GPU`; not `OUTROOT`, `DESIGNLIST`, `OPENROADBIN`.
- `run_power_benchmark_compare.sh` must call its local
  `power_benchmark_compare_one.py`, not `/data/Xplace/logs/...`.
- Replace `sys.path.insert('/data/Xplace')`, `platformPath`, and `designPath`
  with `XPLACE_DIR`, `PLATFORM_PATH`, and `DESIGN_PATH` inputs.

## Required Compare Semantics

- OpenROAD CSV is golden oracle only; Xplace must compute power itself.
- Compare by instance name, never by internal id.
- Report internal, switching, leakage, total, worst instance/component, and
  1% total-power status.
- Preserve OpenSTA semantics: switching load cap not Ceff; activity follows
  `ensureActivities()`; internal/leakage use Liberty table/when rules.
