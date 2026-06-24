# Build Timing Graph 审查索引

Last reviewed: 2026-06-23

原来的 build timing graph 审查笔记已经按阶段拆分。入口链路仍然是：

```text
timer_only.timing_opt.GPUTimer(...)
  gputimer.create_timing_rawdb(...)
  gputimer.create_gputimer(...)
    C++ create_gputimer(...)
      GTDatabase(rawdb, gpdb, timing_raw_db)
      SDC::read(...)
      GTDatabase::preparePinNameMapForSdc(...)
      GTDatabase::ExtractTimingGraph()
      GTDatabase::readSdc(...)
      GTDatabase::RunSdcConstantSimulation()
      gt::GPUTimer(gtdb, timing_raw_db)
  gt::GPUTimer::initialize()
  gt::GPUTimer::levelize()
```

## 拆分文件

| 文件 | 覆盖内容 |
|---|---|
| [02A_BUILD_TIMING_ENTRY_RAWDB.md](02A_BUILD_TIMING_ENTRY_RAWDB.md) | Python wrapper、`create_timing_rawdb`、`create_gputimer`、`GTDatabase` 基础对象 |
| [02B_BUILD_TIMING_EXTRACT_GRAPH.md](02B_BUILD_TIMING_EXTRACT_GRAPH.md) | `preparePinNameMapForSdc`、`ExtractTimingGraph()`、Liberty flatten、pin map/tag、net/cell arcs、pin arc lists、frontiers、endpoint compaction、tensor materialization |
| [02C_BUILD_TIMING_SDC.md](02C_BUILD_TIMING_SDC.md) | `readSdc()`、SDC Tcl/JSON bridge、timing/clock/exception command visitor、sparse clock tables、constant simulation placeholder |
| [02D_BUILD_TIMING_TIMER_INIT.md](02D_BUILD_TIMING_TIMER_INIT.md) | `gt::GPUTimer` constructor、`initialize()`、`levelize()`、`levelize_power` 复用、downstream consumers、审查清单、高风险点和调试开关 |

## 阅读顺序

1. 先看 `02A_BUILD_TIMING_ENTRY_RAWDB.md`，确认入口和 raw tensor 生命周期。
2. 再看 `02B_BUILD_TIMING_EXTRACT_GRAPH.md`，审查 host-side canonical timing graph 怎么建。
3. 然后看 `02C_BUILD_TIMING_SDC.md`，审查 SDC values 和 sparse clock state 怎么落到 graph/state 上。
4. 最后看 `02D_BUILD_TIMING_TIMER_INIT.md`，审查 graph 怎么传到 CUDA，并被 timing/power 下游使用。

## 当前重点

- `ExtractTimingGraph()` 仍是 CPU host-side canonical graph build；它不应该因为 SDC exception 直接物理删除 canonical arc。
- `readSdc()` 现在在 graph build 之后执行；clock state 是 sparse `clock_id + clock_* table`，不是旧的 per-pin/per-test dense clock tensor。
- `release_dmp_timing_scratch_for_power()` 不能释放 sparse clock device tables，因为 power CUDA 会复用 `h_dmp_db->pin_clock_ids` 和 `h_dmp_db->clock_slews`。
- `SetFalsePath` 当前不是 OpenROAD-style path-tag timing exception；它只生成 optional debug-only power mask。
