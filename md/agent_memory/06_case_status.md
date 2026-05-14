# Case Status

Current timing source:

```text
result/ispd25_direct_route_latest/xplace_openroad_all_case_matrix.csv
```

## 1% Timing Gate

All 12 current visible+blind ISPD2025 cases pass direct `--route_segments`
against CRPR-off OpenROAD on the same saved skip-fanout300 segment input.

Small/mid examples:

```text
visible/ariane               -0.510/-1456.030
visible/bsg_chip             -0.447/-10281.188
visible/NV_NVDLA_partition_c -48.101/-447159.05
visible/mempool_tile_wrap    -0.674/-3411.857
blind/ariane                 -0.956/-237.935
blind/mempool_tile_wrap      -0.516/-2179.185
```

Large cases also pass in the matrix:

```text
visible/mempool_group
visible/mempool_cluster
blind/mempool_group
blind/mempool_cluster
```

## 4x Speed Gate

Timing pass does not imply 4x speed pass.

From `quiet_4x_all_case_20260514_gpu0`:

```text
visible/bsg_chip: OpenROAD 97.53s, Xplace 53.74s, speedup 1.81x, fail
visible/NV_NVDLA_partition_c: OpenROAD 31.13s, Xplace 23.53s, speedup 1.32x, fail
```

From earlier speed checkpoints:

```text
visible/mempool_group warm cache: 101.08s vs OpenROAD 447.51s, 4.43x
visible/ariane warm cache: about 8-9s, still below 4x target
```
