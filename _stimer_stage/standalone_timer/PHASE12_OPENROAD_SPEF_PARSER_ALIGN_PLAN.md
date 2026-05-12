# Phase12: OpenROAD SPEF Parser Alignment Plan

Goal: keep the fast line-based SPEF parser, but match OpenROAD/OpenSTA SPEF
semantics for the cases used by timing RC graph construction.

## OpenROAD Reference

- `src/sta/parasitics/SpefLex.ll` tokenizes SPEF with flex.
- `src/sta/parasitics/SpefParse.yy` accepts `*NAME_MAP` entries as
  `INDEX mapped_item`.
- `src/sta/parasitics/SpefReader.cc` stores name-map entries by index,
  looks names up by the numeric prefix, splits pin/node suffixes with the SPEF
  delimiter, and reduces coupling caps onto local net nodes when coupling caps
  are not kept.

## Parser Fixes

1. Stop treating lines after `*NAME_MAP` as name-map entries once another SPEF
   control token such as `*PORTS` is seen.
2. Accept fast-path name-map entries only when the line has exactly
   `INDEX mapped_item`, matching the OpenROAD grammar.
3. Keep dense-vector name-map storage for speed, but preserve OpenROAD's
   overwrite behavior for true duplicate `INDEX` entries.
4. Keep numeric-prefix lookup for `*id:suffix` nodes so RC subnodes expand to
   the mapped net/pin base plus suffix.

## Validation

- Build in conda `gnn`.
- Run `ctest --output-on-failure`.
- Run `make install`.
- Run Xplace `blabla` through the path resolver.
- Run Xplace_dmp ASAP7 `des` and confirm the first RC nets are
  `clk`, `reset`, `load_i`, not `I`.
- Confirm the standalone binary still has no Python/Torch linkage.
