# Power Alignment Scripts

Migrated 2441-compatible wrapper for the old OpenROAD/GNNTimer power dump and
Xplace/GPUTimer CUDA power compare flow.

Smoke run:

```bash
GPU=0 tools/power_alignment/run_power_benchmark_compare.sh
```

Useful overrides:

```bash
DESIGN_LIST=/path/to/designs.txt
OUT_ROOT=/research/d7/ascstd/qkduan25/Xplace/result/sky130_power_alignment_run
OPENROAD_DUMP_DIR=/path/to/cached/openroad_dump
```

Defaults point to:

- OpenROAD: `/research/d7/ascstd/qkduan25/GNNTimer/openroad/build-check/bin/openroad`
- GNNTimer: `/research/d7/ascstd/qkduan25/GNNTimer`
- Xplace: repository root
- Design path: `/research/d7/ascstd/qkduan25/TimingPredict/data/netlists`
- Platform path: `sky130hd`
