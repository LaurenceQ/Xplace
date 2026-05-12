# Phase13: SPEF Coupling Parser Speed Plan

Goal: reduce large SPEF parse time and memory while matching OpenROAD's default
`read_spef` behavior.

## OpenROAD Behavior

- `read_spef` defaults to not keeping capacitive coupling.
- Unless `-keep_capacitive_coupling` is set, `SpefReader::makeCapacitor`
  folds each coupling cap into ground cap on nodes belonging to the current net
  using `coupling_reduction_factor`, default `1.0`.

## Implementation

1. Add `SpefParseOptions` with `keep_coupling_caps` and
   `coupling_reduction_factor`.
2. Default standalone timer to OpenROAD behavior:
   `keep_coupling_caps=false`, `coupling_reduction_factor=1.0`.
3. In the fast SPEF parser, avoid creating external coupled nodes when
   coupling caps are not kept.
4. Preserve a debug switch, `--keep-coupling-caps`, to reproduce the old full
   coupling graph.
5. Preserve `--coupling-reduction-factor <value>` for OpenROAD-style tuning.

## Bison Fallback

If a real SPEF fails the fast parser because it depends on rarely used grammar
features, add a build option `STIMER_USE_BISON_SPEF` and implement a strict
flex/bison SPEF reader modeled after OpenROAD's `SpefLex.ll` and
`SpefParse.yy`. Keep it as an optional correctness fallback, not the default
hot path.

## Validation

- Build in conda `gnn`.
- Run `ctest --output-on-failure`.
- Run `make install`.
- Run ASAP7 `des` and Xplace `blabla`.
- Run `blabla --keep-coupling-caps` once to confirm the old graph is still
  available for debug.
