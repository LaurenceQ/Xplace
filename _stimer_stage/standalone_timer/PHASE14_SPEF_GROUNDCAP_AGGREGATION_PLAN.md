# Phase14: SPEF Ground Cap Aggregation Plan

Goal: continue parser speed work after OpenROAD-style coupling reduction by
removing unnecessary per-capacitor allocations for ground caps.

## Motivation

OpenROAD's default `read_spef` path does not create a separate object for every
ground capacitance contribution. It calls `incrCap(node, cap)` and accumulates
capacitance on the parasitic node. After Phase13, standalone timer still pushed
one `RcCapacitor` per reduced ground-cap contribution, which kept parse time and
memory higher than necessary.

## Implementation

1. Add `RcNode::ground_capacitance`.
2. Change the fast SPEF parser so all ground caps increment the node's
   accumulated capacitance.
3. Keep `RcNet::capacitors` for explicit coupling caps only when
   `--keep-coupling-caps` is enabled.
4. Track `rc_ground_cap_nodes` in summaries and reports.
5. Count `rc_capacitors` as `ground_cap_nodes + explicit coupling caps`, which
   reflects the compact RC graph rather than raw SPEF cap-line count.

## Validation

- Build in conda `gnn`.
- Run `ctest --output-on-failure`.
- Run `make install`.
- Run ASAP7 `des`.
- Run Xplace `blabla` default mode and `--keep-coupling-caps` mode.
- Confirm CUDA/CPU max absolute error remains zero in CUDA runs.
