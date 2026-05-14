# DMP Timing Path

Current code has one direct-route DMP timing propagation path.

## Single Path

The retained implementation is the direct gate/net path:

```text
build_forward_arc_levels(): gate_arc_list, net_arc_list, direct_net_arc_list
dmpGateKernel()
dmpDirectNetKernel() for direct net arcs
dmpNetWinnerKernel()
dmpPinWinnerKernel()
dmpTestKernel()
dmpBackwardKernel()
```

`set_driving_cell` source handling now uses the same local-state engine:
`applyDrivingCellSourceSlewKernel` stores only source timing metadata and
source slew; `propagateLoadSlewDelay()` recomputes the driving-cell local
state in the direct-net thread and adds the virtual extra delay there.

Threshold handling follows OpenROAD: timing arcs and pins map to DMP library
ids; library arrays hold input/output/slew thresholds. Threshold adjust is
skipped when driver/load library ids match.
Because `CellLib` merges multiple `.lib` files, DMP ids are deduped by the
threshold set copied into each `LibertyCell`; Nangate/fakeram stay separate.

Forward/backward timing propagation lives in `DmpTiming.cu`.
Gate files are `Common`, `CellModel`, `Propagation`, and `Direct`; host
driving-cell wrapper is merged into `Direct`, and prototypes/waveform helpers
live in `DmpCudaUtils.cuh` after deleting `DmpKernels.cuh`/`DmpWaveform.cuh`.
`Direct` uses `DmpLocalGateState` for DMP waveform/crossing helpers; do not
reintroduce per-slot waveform arrays or long helper arg lists.
Keep the copied `gt::exp2/y/dy` formulas unchanged; moving/renaming that path
once changed ariane from `-0.510/-1456.030` to `-0.546/-1756.678`.

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
