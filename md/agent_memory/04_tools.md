# Tools And Commands

## Build

```bash
cd /research/d7/ascstd/qkduan25/Xplace/build
source ~/.bashrc
conda activate gnn
cmake ..
make -j8
make install
```

Always run `make install`; Python imports from `cpp_to_py/cpybin`.

## Direct Timer Smoke

```bash
python run_timer.py \
  --platformPath /research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/NanGate45 \
  --designPath /research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/<split> \
  --designName <design> \
  --route_segments /research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/openroad_gr_segments_skip_fanout300/<split>/<design>.route_segments \
  --global_placement False --legalization False --detail_placement False \
  --write_placement False
```

## Batch Matrix

```text
result/ispd25_direct_route_latest/xplace_openroad_all_case_matrix.md
result/ispd25_direct_route_latest/xplace_openroad_all_case_matrix.csv
```

All-case runner:

```text
tools/run_ispd25_all_case_matrix.py
```

Its `missing_fanout_skip_value()` has the current skip policy.

## Key Environment

```text
DMP_FORCE_PIN_FALLBACK=1
GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP=<0|300>
```

Use route/memory profile env vars only when debugging, not by default.
