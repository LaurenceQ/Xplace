# DMP Timing Path

Current code has one direct-route DMP timing propagation path.

## Single Path

The retained implementation is the fused gate/net path:

```text
build_forward_arc_levels(): gate_arc_list, net_arc_list, direct_net_arc_list
propagateFusedGateNetDelaySlewAndAT_dmp()
propagateNetArcSlewDelay_dmp() for direct net arcs
finalizeNetDelayWinnersAndPropagateAT_dmp()
finalizePinWinners_dmp()
propagatePinTests_dmp()
propagatePinBack_dmp()
```

`set_driving_cell` source handling now uses the same local-state engine:
`applyDrivingCellSourceSlewKernel` stores only source timing metadata and
source slew; `propagateLoadSlewDelay()` recomputes the driving-cell local
state in the direct-net thread and adds the virtual extra delay there.

Removed alternate paths:

```text
DMP_FORCE_PIN_FALLBACK env switch
full arc-level gate/net-pair materialization
hybrid arc-slot path
old pin-level propagation fallback
pin lock fallback arrays
```

Reason: timing kernels are not the end-to-end bottleneck, and full arc-level
OOMs on `mempool_group`. Keep this path simple unless correctness fails.
