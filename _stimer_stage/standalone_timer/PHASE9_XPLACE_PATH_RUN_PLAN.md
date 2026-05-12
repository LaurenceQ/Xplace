# Phase9: Xplace-Style Path Runner Plan

Goal: run real Xplace/Xplace_dmp designs by passing paths only. No JSON
manifest, no Python, no Torch.

## Interface

- Add `--platform <path>` for a PDK/platform directory.
- Add `--design-dir <path>` for one design directory.
- Add `--design-root <path> --design <name>` as the parent-directory form.
- Keep explicit `--liberty/--def/--verilog/--spef/--sdc` overrides.

## Path Rules

- Liberty files are discovered under `<platform>/LIB`, `<platform>/lib`, and
  `<platform>`.
- Default behavior follows Xplace and skips filenames containing `ram`
  case-insensitively.
- `--include-ram-lib` keeps RAM Liberty files when needed.
- Design files are discovered from `<design-dir>` using these priorities:
  - DEF: `<name>.def`, `20-<name>.def`, then other `.def` files excluding
    `.cells.def`, `.fp.def`, and `.ref.def`.
  - Verilog: `<name>.v`, then other `.v` files.
  - SPEF: `<name>.spef`, `20-<name>.spef`, then other `.spef` files.
  - SDC: `<name>.sdc`, `<name>.cts_1.sdc`, then other `.sdc` files.

## Implementation

- Add `stimer::PathResolver` in C++ only.
- Resolve path inputs before constructing `stimer::Timer`.
- Print the resolved paths before probing inputs, so bad path choices are
  visible immediately.
- Add a CTest path-smoke using existing minimal test data.

## Validation

- Build in conda `gnn` from `standalone_timer/build`.
- Run `ctest --output-on-failure`.
- Run at least one real Xplace_dmp design with:
  `--platform /research/d7/ascstd/qkduan25/Xplace_dmp/platform/ASAP7`
  and `--design-dir /research/d7/ascstd/qkduan25/Xplace_dmp/design/<name>`.
- Verify static rules:
  - no libtorch/libpython dependency,
  - no CUDA runtime calls in `.cpp`/`.h`.
