# Speed And Memory Status

## Current Direction

Optimize end-to-end direct `--route_segments` Xplace time. Avoid rerunning
OpenROAD; use saved matrix runtimes as reference.

## Known Bottlenecks

- Small/mid cases are dominated by Python import, design load, timing DB setup,
  GPUTimer init, and route segment/cache overhead.
- Large cases are dominated by route segment parse/finalize and timing memory.
- GPU allocation can appear early, but that does not mean CUDA kernels are the
  bottleneck.

## Large-Case Memory Facts

All-case matrix completed visible/blind `mempool_cluster`:

```text
Xplace wall about 678s/683s
peak sampled CPU RSS about 169.7/171.5 GiB
peak sampled GPU memory about 77.3/77.7 GiB
```

## Important Memory Fixes

- Removed per-slot threshold arrays.
- Compact endpoint pin slack storage.
- Skip direct-route ref/ratio and state-backup tensors.
- Defer timing scratch until after RC propagation.
- Release transient RC graph arrays after GPU upload.
- Do not re-enable default debug/profile output.

## Thread Note

Automatic direct-route thread lift was removed. `run_timer.py` now leaves
`--num_threads` at the normal CLI/default value unless explicitly set.

## 2026-05-14 Blind Timer-Only Rerun

All blind mid cases passed timing, but none hit 4x speedup:

```text
ariane 1.07x, bsg_chip 2.55x, NV_NVDLA_partition_c 1.61x
mempool_tile_wrap 1.11x, mempool_group 3.51x
```
