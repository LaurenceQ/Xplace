# DMP Fallback Branch

Do not confuse compile-time `DMP_FORWARD_ARC_LEVEL` with runtime full
arc-level timing.

## Correct Current Branch

For direct route timing:

```text
DMP_FORWARD_ARC_LEVEL=true
DMP_FORCE_PIN_FALLBACK=1
runtime use_arc_level=0
runtime use_fused_fallback=1 or historical use_hybrid_arc_slots=1
gate_net_pairs=0
```

This is not full arc-level pair materialization.

## Work Partition

The fused fallback kernel launches one work item per gate/cell arc lane:

```text
arc_pos = idx >> 3
lane    = idx & 0b111
```

It computes local gate state for that cell arc/lane, then enumerates the
output pin's net sink arcs and submits `updateLoadWinner()` candidates.

## Historical Passing Logs

Examples:

```text
fallback_ab/logs/*/*.direct_route.log: use_arc_level=0, use_hybrid_arc_slots=1
fallback_ab_fused/logs/*/*.direct_route.log: use_arc_level=0, use_fused_fallback=1
fallback_ab_full_debug: use_arc_level=1 and dense gate_net_pairs
```

If blind ariane reports `-0.460/-83.427`, first check skip-fanout policy, not
this branch.
