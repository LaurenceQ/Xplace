# Current Case Debugging

## bsg_chip

- Timing passes with missing-high-fanout skip `300`.
- Visible pass: Xplace `-0.447/-10281.188` or `-10281.189` vs OpenROAD
  `-0.44652295/-10282.74414`.
- Old timing bug was per-capture-clock period plus `set_clock_uncertainty`.
- Current remaining issue is 4x speed, not 1% timing.

## visible/NV_NVDLA_partition_c

- Timing needs `GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP=0`.
- Pass: Xplace `-48.101/-447159.05` vs OpenROAD
  `-48.10162735/-447169.3125`.
- False-fail with skip `300`: about `-0.430/-3213.686`.
- Historical fix: use SDC `set_clock_transition 0.05` as ideal clock slew for
  register clock pins and ideal-clock setup checks.
- Current remaining issue is 4x speed, not 1% timing.

## blind/NV_NVDLA_partition_c

- Current matrix uses skip `300`.
- OpenROAD `0/0`; Xplace has non-negative WNS and TNS `0`.
- Timing passes.

## blind/ariane

- Timing needs `GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP=0`.
- Pass: about `-0.956/-237.935`.
- False-fail with skip `300`: about `-0.460/-83.427`.
- If this false value appears, check skip policy before changing DMP kernels.

## mempool_tile_wrap

- Timer-only import and placement import were checked after the split.
- Visible both reported `-0.674/-3411.857`.
- Blind both reported `-0.516/-2179.185`.
- The timer-only Python import split is not changing this case's timing.
