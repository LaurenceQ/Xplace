# DMP Arc-Level Vs Fallback Profile

Date: 2026-05-14. Scope: visible route-segment runs only.
Logs: `result/dmp_arc_vs_fallback_20260514/logs/`.

Definitions:
- `load_s`: `loading from original benchmark` to `Design loaded successfully`.
- `build_s`: `Design loaded successfully` to `Running DMP timing`.
- `route_*_s`: `[ROUTE_SEG_PROFILE]` elapsed time.
- `rc_ms`: `calc_dmp_rc + propagate_rc_dmp` CUDA event time.
- `timing_ms`: forward + backward DMP timing CUDA event time.
- `kernel_ms`: `[DMP KERNEL PROFILE] total_kernel_ms`.

```text
case mode load build parse finalize rc_ms timing_ms kernel_ms wall rssGiB WNS/TNS
ariane fallback 4.846 3.217 0.396 0.863 3.964 199.085 192.616 10.75 1.81 -0.510/-1456.030
ariane arc 4.846 3.217 0.396 0.863 3.995 102.037 89.189 9.47 1.87 -0.510/-1456.030
mempool_tile fallback 2.311 3.250 0.495 1.043 5.193 217.156 206.887 10.87 2.06 -0.674/-3411.857
mempool_tile arc 2.311 3.250 0.495 1.043 5.101 101.189 91.758 11.01 2.15 -0.674/-3411.857
bsg_chip fallback 16.244 7.316 2.612 5.420 14.578 669.529 663.221 41.78 7.57 -0.447/-10281.189
bsg_chip arc 16.244 7.316 2.612 5.420 14.526 399.820 392.640 29.90 7.84 -0.447/-10281.189
mempool_group fallback 58.738 32.831 13.995 28.917 62.416 2376.090 2363.878 150.11 33.35 -0.912/-45811.569
mempool_group arc OOM after RC/timing scratch allocation; no WNS/TNS.
```

Conclusion: arc-level is faster for timing kernels on passing mid cases, but
uses much more scratch. `mempool_group` arc-level failed at
`driving_cell_extra_delay_` allocation after 58.748 GiB already allocated.
