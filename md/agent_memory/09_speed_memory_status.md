# Speed And Memory Status

Optimize end-to-end direct `--route_segments` Xplace time. Avoid rerunning
OpenROAD; use saved matrix runtimes as reference.

## Known Bottlenecks

- Small/mid cases are dominated by Python import, design load, timing DB setup,
  GPUTimer init, and route segment/cache overhead.
- Large cases are dominated by route segment parse/finalize and timing memory.
- GPU allocation can appear early, but that does not mean CUDA kernels are the
  bottleneck.

All-case matrix completed visible/blind `mempool_cluster`:

```text
Xplace wall about 678s/683s
peak sampled CPU RSS about 169.7/171.5 GiB
peak sampled GPU memory about 77.3/77.7 GiB
```

- Removed dead per-slot thresholds and waveform scratch arrays
  (`k*`, `p*`, `A/B/D`, `rd/t0/dt`, `ceff`, `vo_*`, `dmp_alg_kind`).
- Replaced per-pin/per-timing threshold copies with pin/timing library-id
  maps plus compact per-library threshold arrays.
- `set_driving_cell` now stores only source metadata; direct-net computes
  virtual local state on the fly.
- Compact endpoint pin slack storage.
- Skip direct-route ref/ratio and state-backup tensors.
- Defer timing scratch until after RC propagation.
- Release transient RC graph arrays after GPU upload.
- Do not re-enable default debug/profile output.

Automatic direct-route thread lift was removed. `run_timer.py` now leaves
`--num_threads` at the normal CLI/default value unless explicitly set.

All blind mid cases passed timing, but none hit 4x speedup:

```text
ariane 1.07x, bsg_chip 2.55x, NV_NVDLA_partition_c 1.61x
mempool_tile_wrap 1.11x, mempool_group 3.51x
```

2026-05-14 DMP file cleanup smoke passed after build/install:
visible ariane -0.510/-1456.030, 11.58s, RSS 1856028 KB;
visible bsg_chip -0.447/-10281.189, 33.34s, RSS 7846836 KB;
prior blind ariane -0.956/-237.935.
