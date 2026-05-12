# Phase10: Parser Speed Plan

Goal: keep the standalone timer C++/CUDA-only while making real Xplace-style
design loading faster.

## Current Measurements

On Xplace `blabla`:

- DEF: about 1.0-1.6 s.
- Verilog: about 1.3-2.2 s.
- SPEF before fast parser: about 8.0 s.
- SPEF after line parser: about 1.5-1.7 s.

## Next Optimizations

1. Keep the SPEF fast line parser as the default path.
2. Replace per-line `std::vector<std::string_view>` token storage with a small
   fixed token array because SPEF lines have a small bounded token count.
3. Add a raw-token node cache keyed by `std::string_view` for the current net.
   This avoids repeatedly allocating expanded node names for repeated SPEF node
   tokens.
4. Use the raw-token cache as the hot path. Do not keep a second expanded-name
   hash map because it duplicates every RC node name and slows large SPEFs.
5. Fix name-map suffix expansion:
   - `*3` maps to `clk`.
   - `*3:51` maps to `clk:51`, not `clk`.
6. Store name-map entries in a dense vector indexed by SPEF id instead of an
   unordered map.
7. Reserve approximate capacities for name map and net vectors from file size.

## Validation

- Build in conda `gnn`.
- Run `ctest --output-on-failure`.
- Run `make install`.
- Run Xplace `blabla` CPU and CUDA path.
- Check that RC counts and CPU/CUDA results remain stable.
- Check no Python/Torch link and no CUDA runtime usage in `.cpp/.h`.
