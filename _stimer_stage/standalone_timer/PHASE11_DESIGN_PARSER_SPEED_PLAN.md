# Phase11: Design Parser Speed Plan

Goal: reduce real-design load time after SPEF parser optimization.

## Current Bottleneck

On Xplace `blabla`, after SPEF optimization:

- Liberty read: about 0.4-0.6 s.
- DEF read: about 1.0-1.7 s.
- Verilog read: about 1.3-2.2 s.
- Design count rebuild: about 0.4-0.6 s.
- SPEF RC graph: about 1.3-1.4 s.

## Plan

1. Treat DEF as the primary physical/timing design source when it has both
   components and nets.
2. Skip auto-discovered Verilog after a complete DEF because it usually
   rebuilds duplicate connectivity.
3. Keep a `--force-verilog` CLI switch for debugging or designs whose DEF is
   incomplete.
4. Reserve per-net connection containers in DEF callbacks using
   `defiNet::numConnections()` to reduce vector growth and hash rehash.
5. Avoid repeated connection-key rebuilds after `read_def_design()` when the
   parser has already maintained connection keys incrementally.

## Validation

- Build in conda `gnn`.
- Run `ctest --output-on-failure`.
- Run `make install`.
- Run Xplace `blabla` CPU and CUDA.
- Run a real Xplace_dmp ASAP7 design (`des`) CPU/CUDA.
- Confirm no Python/Torch link and no CUDA runtime calls in `.cpp/.h`.
