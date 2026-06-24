# Struct / Profile Cleanup Audit

本文记录扫描结果、已实现清理和剩余可改点。扫描范围沿现有审查文档调用链走：

```text
parser/data loader
  -> GTDatabase timing graph
  -> SDC / sparse clock
  -> route-segment RC
  -> DMP timing
  -> power cuda input / activity / report
```

核心审查标准：

- 如果一组数据是明确阶段产物，可以保留 `struct`，但下游函数应该直接吃这个 `struct`，不要把字段拆成十几个参数继续传。
- 如果 `struct` 是 CUDA/device view，可以保留 data bag，但不要配一个 20+ 参数构造函数；优先默认构造后按字段赋值，或者由单一 builder 函数填充。
- profile/log 不能污染主流程。重复的 `if (profile) fprintf/fflush`、局部 `profile_log` lambda、阶段 profiler class 应统一成一个短接口或宏。

说明：下面 `源码位置` 指当前源码里已经存在的函数、成员、struct 或 log 点；`可改` 里的示例接口和宏如果没有列源码位置，就是建议新增/改造的形态，不是当前已有 symbol。

## 0. 阅读分组

这份审查现在按两类问题读，不要混在一起看：

### A. Profile / Debug 输出治理

目标是减少主流程里的噪声和样板代码。

- 已经统一的 profile class / profile lambda：见 `A.1 Profile 清理状态`。
- 已经统一或仍残留的 debug/log 输出：见 `A.2 Debug / Log Cleanup Audit`。
- 判断标准：默认路径只保留短 summary；env-gated debug 用 `XPLACE_DEBUGF(...)`；阶段耗时用 `StageProfiler`；device-side `printf` 单独按 CUDA 边界处理。

### B. Struct / Class / API 设计治理

目标是让阶段产物和业务逻辑边界清楚。

- 阶段产物可以是 `struct`，例如 `HostRcGraph`、`DrivingCellSource`。
- 临时 thread-local cell timing arc 不应该暴露在 public header；当前源码名已经改成 `CellTimingArc`，并已移到 `GTDatabase.cpp` anonymous namespace。
- 下游函数应该直接吃阶段产物，不要把 `graph.x / graph.y / graph.z` 拆成十几个参数继续传。
- CUDA/device view 可以保留 data bag，但避免长构造函数；优先默认构造后由 builder 填字段。
- 命名上避免 `State`、`Entry`、`Scratch` 这种没有业务含义的词。只有真正描述状态机/生命周期状态时才用 `State`。
- 如果 struct 只是累加计数器，优先叫 `Counts`；如果它表示某个 worker 或阶段的完整产出，优先叫 `Result`。

当前集中待改的 struct/class/API 点：

| 模块 | 位置 | 类型 | 处理意见 |
| --- | --- | --- | --- |
| GTDatabase timing graph | `cpp_to_py/gputimer/db/GTDatabase.cpp:42` | `CellTimingArc` 临时 cell timing arc | 已放到 anonymous namespace，不再暴露到 header。 |
| GTDatabase timing graph | `cpp_to_py/gputimer/db/GTDatabase.cpp:225`, `cpp_to_py/gputimer/db/GTDatabase.cpp:434` | `BuildNetCellArcAndTest(...)` / `WriteCellArcListAndTest(...)` 局部 helper | 已从 `GTDatabase.h` public API 移除；仍可后续合并成明确 build result。 |
| GTDatabase SDC/DMP | `cpp_to_py/gputimer/db/GTDatabase.h:88`, `cpp_to_py/gputimer/db/GTDatabase.h:214` | `DrivingCellSource` 阶段产物 | 保留，它描述 `set_driving_cell` 到 DMP source slew 的跨阶段数据。 |
| Route RC / DMP | `cpp_to_py/gputimer/core/GPUTimer.h:131` | `HostRcGraph` 阶段产物 | 保留，并让最底层 DMP 初始化直接读 `graph.*` 字段，不再拆 vector 参数。 |
| OpenROAD route builder | `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:110`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:129` | `Openroad*BuildCounts` counter struct | 已从 `*BuildStats` 改名，表达这是 counter bundle。 |
| OpenROAD route builder | `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:671` | `RouteFinalizeWorkerResult` 大局部 struct | 已从 `RouteFinalizeThreadStats` 改名，字段 `counts` 和 `merge(...)` 已补齐；仍可后续移出函数中间。 |
| RC CUDA structs | `cpp_to_py/gputimer/core/rc/RcModels.h:8`, `cpp_to_py/gputimer/core/rc/RcModels.h:30`, `cpp_to_py/gputimer/core/rc/RcModels.h:42`, `cpp_to_py/gputimer/core/rc/RcModels.h:50`, `cpp_to_py/gputimer/core/rc/RcModels.h:126` | `RcStarNet` / `RcTreeHost` / `RcTreeDeviceGraph` / `RcTreePropagation` / `RcTreeDevice` | 已去掉 `Model` / `View` / `Scratch` / `Explicit` 命名；`RcTreeDeviceGraph` 是 `RcTreeDevice` 的拓扑子对象。 |
| Power CUDA model | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:10`, `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:114`, `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:228` | device view data bag | `Power*DeviceView` 可保留 view，但长构造函数改成默认构造 + builder 填字段。 |
| Power activity model | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:143`, `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:313`, `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:365` | `State` / `ScratchView` / queue view | 建议改成业务名：`PowerActivitySeeds`、`PowerActivityBuffers`、`PowerActivityLevelQueue`。 |
| Power host trace | `cpp_to_py/gputimer/core/power/common/PowerHostCommon.h:29` | `PowerTracePathWriter` | 不是状态机，建议改成 `PowerTracePathFilter` 或 `PowerTracePathWriter`。 |
| Route grad | `cpp_to_py/gputimer/core/route_grad/DmpRouteGradHost.h:20` 等 | RAII/device buffer 和 host result bundle | 保留；统一使用 `HostRcGraph::num_nets`，不要从 vector size 反推。 |

下面的详细条目仍按调用链展开，但每个条目会标明它属于 profile/debug 还是 struct/class/API。

## A. Profile / Debug 输出治理

### A.1 Profile 清理状态

状态：已实现并编译通过。

新增统一接口：

- `StageProfiler`：`cpp_to_py/common/StageProfiler.h:19`
- `xplace_env_enabled(...)`：`cpp_to_py/common/StageProfiler.h:8`
- `StageProfiler::mark(...)`：`cpp_to_py/common/StageProfiler.h:28`
- `StageProfiler::markSeconds(...)`：`cpp_to_py/common/StageProfiler.h:48`
- `StageProfiler::markf(...)`：`cpp_to_py/common/StageProfiler.h:68`
- `XPLACE_DEBUGF(...)` / `XPLACE_PROFILEF(...)` / `XPLACE_ERRORF(...)`：`cpp_to_py/common/XplaceLog.h:23`, `cpp_to_py/common/XplaceLog.h:26`, `cpp_to_py/common/XplaceLog.h:32`

已替换的 profile 样板：

- 删除 rawdb `XplaceIoProfileTimer`，`Database::load()` / `Database::setup()` 直接使用 `StageProfiler`。
- 删除 DEF materialize、GPDB setup、pybind create、route segment builder、route cache loader 的局部 `profile_log` / `cache_profile_log` / `log_profile` lambda。
- 删除 timing graph build 的 `ExtractProfileTimer`。
- 删除 DMP RC wrapper/model init 的 `DmpRcPhaseProfile`。
- `PowerStageProfiler` 保留类型名，但内部改为持有 `StageProfiler`，避免 power 侧继续维护独立 chrono/fprintf 实现。
- `PowerReport.cpp` 的 power stage summary 输出改成统一 `phase=... elapsed=... total=...` 格式。
- `file_lefdef_db.cpp::DefReadProfile` 不再手写 chrono/fprintf，内部改为 `StageProfiler`，只保留 DEF callback 计数器职责。
- `profileDefBufferScan(...)` / fast DEF stage timing 改为 `StageProfiler::mark/markf(...)`。
- SDC verbose units / missing-object warning / driving-cell verbose warning 改为 `XPLACE_DEBUGF(...)`。
- inference CSV 和 `apply_infer_data(...)` 的逐行/sample/D2H/H2D 输出改为 `XPLACE_INFER_DEBUG`，默认只保留 summary。
- power row/root/upload/CPU activity trace 的 host-side debug 输出改为 `XPLACE_DEBUGF(...)` 或 `XPLACE_ERRORF(...)`。
- `DMP_PROGRESS_PRINT(...)` 内部改为 `XPLACE_LOGF("DMP_PROGRESS", ...)`，`dmp_debug_on` 的 host trace 改为 `XPLACE_ERRORF("DMP_FLUTE_DEBUG", ...)`。

仍保留但不属于本轮 stage-profile 样板的输出：

- `DmpTiming.cu` 的 CUDA event kernel profile lambda：这是 kernel 事件计时，不是阶段 mark。
- `DmpRc.cu::print_dmp_rc_kernel_profile(...)` 和 `print_dmp_rc_parallel_stats(...)`：这是 kernel/work summary 输出，后续可以再封装成 debug/profile helper。
- `GPUTimer.cu` 的 `DMP_LUT_PROFILE` 输出、`DmpDebug.cu` 的 root/driving-cell kernel profile 输出：这些也是 CUDA/debug profile，不是普通 stage mark。
- device-side `printf`、CUDA error helper、显式 dump API (`DMP RC DUMP` / `SPEF RC DUMP` / `GR RC DUMP`) 仍保留。原因是它们不是普通 host logger 路径，后续要按 CUDA/device 或 dump API 单独处理。
- `PowerCudaActivity.cu` frontier/trace 和 `DmpRc.cu` CUDA-side debug 仍有裸 `fprintf/printf`，这是下一轮 CUDA/debug helper 统一的范围。

#### Route Segment RC profile

源码位置：

- route `StageProfiler`：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:243`
- cache `StageProfiler`：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:263`
- cache profile marks：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:269`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:271`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:273`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:275`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:283`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:288`
- route profile marks：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:292`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:300`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:309`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:321`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:326`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:331`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:333`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:488`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:497`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:638`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:655`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:706`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:1027`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:1105`
- 旧 route/cache profile lambda 和 route-profile `if (profile) fprintf(...)`：已删除。

Route segment finalize 里仍有一类 profile-adjacent 字段要注意：

```cpp
double pinloc_seconds = 0.0;
double attach_seconds = 0.0;
double reorder_seconds = 0.0;
double repair_seconds = 0.0;
double prune_seconds = 0.0;
```

它们当前在 `OpenroadRouteSegmentsBuilder.cpp` 的 `RouteFinalizeWorkerResult` 里，每个 worker 局部累加，最后通过 `RouteFinalizeWorkerResult::merge(...)` 合并到 `finalize_total`，再通过：

```cpp
route_profile.markf("finalize_done", ...);
```

输出。这个是 profile 归属，不应该和 RC graph 业务 struct 混在一起讨论。后续重构时可以保留在 `RouteFinalizeWorkerResult`，但要把它明确标成 `timing` 或 `profile` 字段，例如：

```cpp
struct RouteFinalizeProfileTimes {
    double pinloc_seconds = 0.0;
    double attach_seconds = 0.0;
    double reorder_seconds = 0.0;
    double repair_seconds = 0.0;
    double prune_seconds = 0.0;
    void merge(const RouteFinalizeProfileTimes& other);
};
```

验证：

- `source /research/d7/ascstd/qkduan25/app/miniconda3/etc/profile.d/conda.sh && conda activate gnn && make -j8`
- `source /research/d7/ascstd/qkduan25/app/miniconda3/etc/profile.d/conda.sh && conda activate gnn && make install`
- `source /research/d7/ascstd/qkduan25/app/miniconda3/etc/profile.d/conda.sh && conda activate gnn && python -c "from cpp_to_py import gputimer, io_parser; print('ok')"`

### A.1.1 Parser / Data Loader

#### `cpp_to_py/common/db/Database.cpp`

源码位置：

- `StageProfiler` 使用点 `Database::load()`：`cpp_to_py/common/db/Database.cpp:85`
- `StageProfiler` 使用点 `Database::setup()`：`cpp_to_py/common/db/Database.cpp:785`
- 旧 `XplaceIoProfileTimer`：已删除。

发现：

- 旧 `XplaceIoProfileTimer` 是一个独立 profile class，只用于 rawdb load/setup。
- `Database::load()` 和 `Database::setup()` 里有大量：

```cpp
io_profile.mark("read_lef");
io_profile.mark("read_def");
io_profile.mark("setup_regions");
```

评价：

- 这里的 profile class 比散落 `fprintf` 好，主流程还能读。
- 但它和后面的 `ExtractProfileTimer`、`DmpRcPhaseProfile`、`PowerStageProfiler` 重复实现同一件事。

可改：

- 已完成：提取通用 `StageProfiler`，输出统一为 `phase=... elapsed=... total=...`。

#### `cpp_to_py/common/io/file_lefdef_db.cpp`

源码位置：

- `StageProfiler` 使用点：`cpp_to_py/common/io/file_lefdef_db.cpp:1250`
- profile marks：`cpp_to_py/common/io/file_lefdef_db.cpp:1327`, `cpp_to_py/common/io/file_lefdef_db.cpp:1332`, `cpp_to_py/common/io/file_lefdef_db.cpp:1343`, `cpp_to_py/common/io/file_lefdef_db.cpp:1354`, `cpp_to_py/common/io/file_lefdef_db.cpp:1421`, `cpp_to_py/common/io/file_lefdef_db.cpp:1433`
- 旧 local `profile_log` lambda：已删除。

发现：

- 旧 LEF/DEF loader 里也有一套局部 `profile_log` lambda，和 GPDB / pybind 的 profile lambda 形态一致。

可改：

- 已完成：和 `Database.cpp` / `GPDatabase.cpp` 一起收敛到 `StageProfiler`。

#### `cpp_to_py/io_parser/gp/GPDatabase.cpp`

源码位置：

- `GPDatabase::setupNets()` 内 `StageProfiler`：`cpp_to_py/io_parser/gp/GPDatabase.cpp:326`
- `setupNets()` profile marks：`cpp_to_py/io_parser/gp/GPDatabase.cpp:336`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:344`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:394`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:400`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:406`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:414`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:428`
- `GPDatabase::setup()` 内 `StageProfiler`：`cpp_to_py/io_parser/gp/GPDatabase.cpp:721`
- `setup()` profile marks：`cpp_to_py/io_parser/gp/GPDatabase.cpp:726`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:729`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:731`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:733`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:735`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:737`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:739`, `cpp_to_py/io_parser/gp/GPDatabase.cpp:741`
- 旧局部 `profile_log` lambda：已删除。

发现：

- 旧实现中，`setupNets()` 内部定义局部 `profile_log` lambda。
- 旧实现中，`setup()` 又定义几乎一样的 `profile_log` lambda。
- 旧实现中，两段都手写 `chrono + fprintf + fflush`。

问题：

- 这是 profile 样板复制，和主逻辑混在一起。
- `setupNets()` 已经很长，profile lambda 加重阅读负担。

可改：

```cpp
XPLACE_PROFILE_MARK(gpdb_profile, "node_pin_prefix");
XPLACE_PROFILE_MARK(gpdb_profile, "node_pin_flatten");
```

或者：

```cpp
StageProfiler profile("XPLACE_GPDB_PROFILE", gpdbProfileEnabled());
profile.mark("setup_num");
```

状态：已完成。`setupNets()` 的 `threads=%d` 信息通过 `markf(...)` 保留。

#### `cpp_to_py/gputimer/PyBindCppMain.cpp`

源码位置：

- `create_gputimer(...)` 内 `StageProfiler`：`cpp_to_py/gputimer/PyBindCppMain.cpp:26`
- pybind create profile marks：`cpp_to_py/gputimer/PyBindCppMain.cpp:40`, `cpp_to_py/gputimer/PyBindCppMain.cpp:48`, `cpp_to_py/gputimer/PyBindCppMain.cpp:51`, `cpp_to_py/gputimer/PyBindCppMain.cpp:53`, `cpp_to_py/gputimer/PyBindCppMain.cpp:55`, `cpp_to_py/gputimer/PyBindCppMain.cpp:57`, `cpp_to_py/gputimer/PyBindCppMain.cpp:60`, `cpp_to_py/gputimer/PyBindCppMain.cpp:64`
- pybind DMP RC APIs：`cpp_to_py/gputimer/PyBindCppMain.cpp:170`, `cpp_to_py/gputimer/PyBindCppMain.cpp:171`, `cpp_to_py/gputimer/PyBindCppMain.cpp:173`
- 旧 local `profile_log` lambda：已删除。

发现：

- 旧 pybind entry 也手写了 `profile_log`，并且处在 `construct_gtdb -> read_sdc_json -> ExtractTimingGraph -> readSdc -> RunSdcConstantSimulation -> construct_gputimer` 主调用链上。

可改：

- 已完成：主入口已改为 `StageProfiler`，和 GPDB / route segment / DMP RC 使用同一套实现。

## A.2 Debug / Log Cleanup Audit

本节专门记录 `parser -> timer -> power` 路径上的 debug/log 可读性问题。这里不把显式 dump API 当成默认问题；例如 `debug_dump_*` / `dump_timing_graph(...)` 是用户主动调用的输出，可以继续面向 stdout。需要优先清理的是默认路径、env-gated debug path 和重复 profiler path 里的裸 `printf/fprintf/cerr`。

### 当前 logger 基础

源码位置：

- `utils::PrintfLogger`：`cpp_to_py/common/utils/log.h:62`
- `logger.log(...)` 当前底层仍使用 `std::cout + printf`：`cpp_to_py/common/utils/log.h:93`, `cpp_to_py/common/utils/log.h:102`
- `logger.debug/verbose/info`：`cpp_to_py/common/utils/log.h:136`, `cpp_to_py/common/utils/log.h:144`, `cpp_to_py/common/utils/log.h:152`

结论：

- 不需要引入新日志库。当前已经有统一 `logger`，应先把业务代码里的裸输出收敛到它。
- 现有 logger 本身还可以后续再改内部实现，但调用侧应该先统一成 `logger.info/warning/error/debug`。

当前已新增轻量 helper：

```cpp
XPLACE_DEBUGF(env, fmt, ...)
XPLACE_PROFILEF(env, fmt, ...)
XPLACE_LOGF(tag, fmt, ...)
XPLACE_ERRORF(tag, fmt, ...)
```

源码位置：`cpp_to_py/common/XplaceLog.h:8`, `cpp_to_py/common/XplaceLog.h:23`

规则：

- 默认 summary 用 `logger.info(...)`，必须短。
- 异常、fallback、数据不匹配用 `logger.warning(...)`。
- 真错误用 `logger.error(...)` 或抛异常。
- env-gated host debug 一律用 `XPLACE_DEBUGF(...)`，业务代码不要再直接写裸 `printf/fprintf` 或默认 `logger.info`。
- 已经由调用点布尔值控制的 host trace/dump helper 用 `XPLACE_ERRORF(...)` / `XPLACE_LOGF(...)`，避免再套第二层 env。
- 大文件 dump 继续写 `std::ofstream`，但“开始/完成/失败”消息走 `logger`。
- `XPLACE_*` 是当前统一 debug/profile/log 入口；如果后续要让它底层完全接入 `utils::logger`，只改 `cpp_to_py/common/XplaceLog.h`，不要在各业务文件重复发明输出路径。

### Parser / DEF Fast Path

源码位置：

- `Database::save(...)` 失败直接 `std::cout`：`cpp_to_py/common/db/Database.cpp:293`
- `DefReadProfile` 当前仅保留 DEF callback 计数器，阶段计时走 `StageProfiler`：`cpp_to_py/common/io/file_lefdef_db.cpp:65`, `cpp_to_py/common/io/file_lefdef_db.cpp:77`, `cpp_to_py/common/io/file_lefdef_db.cpp:93`, `cpp_to_py/common/io/file_lefdef_db.cpp:105`
- `DefMappedFile` open/stat/mmap failure 已改为 `XPLACE_DEBUGF("XPLACE_DEF_BUFFER_PROFILE", ...)`：`cpp_to_py/common/io/file_lefdef_db.cpp:137`, `cpp_to_py/common/io/file_lefdef_db.cpp:145`, `cpp_to_py/common/io/file_lefdef_db.cpp:158`
- `profileDefBufferScan(...)` 已改为 `StageProfiler::mark/markf(...)`：`cpp_to_py/common/io/file_lefdef_db.cpp:369`, `cpp_to_py/common/io/file_lefdef_db.cpp:377`, `cpp_to_py/common/io/file_lefdef_db.cpp:385`, `cpp_to_py/common/io/file_lefdef_db.cpp:398`
- fast DEF fallback/error 已改为 `XPLACE_DEBUGF(...)`：`cpp_to_py/common/io/file_lefdef_db.cpp:991`, `cpp_to_py/common/io/file_lefdef_db.cpp:1190`, `cpp_to_py/common/io/file_lefdef_db.cpp:1491`, `cpp_to_py/common/io/file_lefdef_db.cpp:1505`, `cpp_to_py/common/io/file_lefdef_db.cpp:1516`
- fast DEF stage 输出已改为 `StageProfiler`：`cpp_to_py/common/io/file_lefdef_db.cpp:1482`, `cpp_to_py/common/io/file_lefdef_db.cpp:1508`, `cpp_to_py/common/io/file_lefdef_db.cpp:1520`, `cpp_to_py/common/io/file_lefdef_db.cpp:1526`, `cpp_to_py/common/io/file_lefdef_db.cpp:1535`, `cpp_to_py/common/io/file_lefdef_db.cpp:1589`, `cpp_to_py/common/io/file_lefdef_db.cpp:1592`
- `file_lefdef_db.cpp` rectangle partition error 已改为 `logger.warning(...)`：`cpp_to_py/common/io/file_lefdef_db.cpp:2638`

问题：

- 这里残留一套 `DefReadProfile`，功能和 `StageProfiler` 重复。
- fast DEF fallback 是有价值 warning，但现在绕过 logger。
- `XPLACE_DEF_BUFFER_PROFILE` 和 `XPLACE_FAST_DEF_*` 输出格式不统一，读 log 时很难按 stage 聚合。
- parser/common 里的裸 `std::cout` 错误输出没有 log level，也不会受全局 logger 控制。

状态：

- 已完成 `DefReadProfile` / DEF buffer / fast DEF stage profile 收敛。
- 已完成 fast DEF debug fallback 和 rectangle partition warning 收敛。
- `Database::save(...)` 的 `std::cout` 仍未处理；它不在本轮 GPUTimer parser-to-power 主调用链上。

### SDC / Sparse Clock

源码位置：

- `warn_missing_sdc_object(...)` 已改为 `XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_WARNINGS", ...)`：`cpp_to_py/gputimer/db/sdc/SdcUtils.cpp:23`
- `SetUnits` verbose units 已改为 `XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_UNITS", ...)`：`cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:53`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:56`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:59`
- `SetDrivingCell` 找不到 pin/arc 已改为 `XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_WARNINGS", ...)`：`cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:191`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:224`
- per-clock debug 已改为 `XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_CLOCKS", ...)`：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:273`
- clock-test / propagation summary：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:329`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:468`

问题：

- 默认路径只保留 summary；per-command / per-clock 细节必须 env-gated。
- SDC warning/debug 输出统一走 `XPLACE_DEBUGF(...)` 这类集中 helper，避免业务代码里继续混用裸 `printf/fprintf` 和默认 `logger.info`。

状态：

- 已完成 SDC verbose units / missing object / driving-cell verbose warning 的 host-side 裸输出清理。
- 已完成 per-clock 默认 info 清理；详细 clock 行现在只在 `GPUTIMER_VERBOSE_SDC_CLOCKS=1` 时输出。
- 默认保留 `Mapped %d/%d timing tests...` 和 `Clock propagation...` 两条 summary。

### Inference CSV / Apply

源码位置：

- `read_infer(...)` 入口和最终 summary 保留 `logger.info(...)`，section/sample/time-scale 已改为 `XPLACE_INFER_DEBUG`：`cpp_to_py/gputimer/core/infer/InferCsv.cpp:64`, `cpp_to_py/gputimer/core/infer/InferCsv.cpp:71`, `cpp_to_py/gputimer/core/infer/InferCsv.cpp:85`, `cpp_to_py/gputimer/core/infer/InferCsv.cpp:105`, `cpp_to_py/gputimer/core/infer/InferCsv.cpp:124`
- `read_opr_gt_infer(...)` 同类输出已收敛：`cpp_to_py/gputimer/core/infer/OpenroadInferCsv.cpp:64`, `cpp_to_py/gputimer/core/infer/OpenroadInferCsv.cpp:81`, `cpp_to_py/gputimer/core/infer/OpenroadInferCsv.cpp:98`, `cpp_to_py/gputimer/core/infer/OpenroadInferCsv.cpp:143`, `cpp_to_py/gputimer/core/infer/OpenroadInferCsv.cpp:170`
- `apply_infer_data(...)` 同步、D2H/H2D、sample 更新输出已改为 `XPLACE_INFER_DEBUG`，默认只保留一条 update summary：`cpp_to_py/gputimer/core/infer/InferApply.cu:19`, `cpp_to_py/gputimer/core/infer/InferApply.cu:23`, `cpp_to_py/gputimer/core/infer/InferApply.cu:28`, `cpp_to_py/gputimer/core/infer/InferApply.cu:50`, `cpp_to_py/gputimer/core/infer/InferApply.cu:91`, `cpp_to_py/gputimer/core/infer/InferApply.cu:152`, `cpp_to_py/gputimer/core/infer/InferApply.cu:174`

问题：

- 这些是 debug/probe 路径，但用了默认 `logger.info`，会在正常跑 inference 时输出大量细节。
- `logger.info` 字符串里自带 `\n`，而 logger 自己会再追加换行，格式不一致。

状态：

- 已完成：默认只保留 opened/loaded/update summary。
- 已完成：section、sample row、D2H/H2D、GPU sync 全部改到 `XPLACE_INFER_DEBUG`。
- 已完成：这些 logger format 里的手写 `\n` 已去掉。

### DMP RC / Route RC

源码位置：

- `DMP_PROGRESS_PRINT(...)` 内部已改为 `XPLACE_LOGF("DMP_PROGRESS", ...)`：`cpp_to_py/gputimer/core/DmpModel.cpp:34`
- `GPUTimer::print_pin_id_name()` 和 `GPUTimer::get_units()` 已改为 `XPLACE_LOGF(...)`：`cpp_to_py/gputimer/core/DmpModel.cpp:78`, `cpp_to_py/gputimer/core/DmpModel.cpp:130`
- `FluteRCTreeDMP(...)` 的 `dmp_debug_on` 已改为 `XPLACE_ERRORF("DMP_FLUTE_DEBUG", ...)`：`cpp_to_py/gputimer/core/DmpModel.cpp:201`, `cpp_to_py/gputimer/core/DmpModel.cpp:210`, `cpp_to_py/gputimer/core/DmpModel.cpp:228`, `cpp_to_py/gputimer/core/DmpModel.cpp:235`
- 通用 CUDA debug print kernels：`cpp_to_py/gputimer/core/utils.cuh:18`, `cpp_to_py/gputimer/core/utils.cuh:31`, `cpp_to_py/gputimer/core/utils.cuh:48`
- `DmpCudaUtils.cuh::gpuErrchk` / `gpuAssert` helper 直接 `fprintf(stderr, ...)`：`cpp_to_py/gputimer/core/DmpCudaUtils.cuh:100`
- DMP RC CUDA error helper 直接 `fprintf(stderr, ...)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:16`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:25`
- DMP RC progress/debug 直接 `fprintf(stderr, ...)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:66`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:88`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:104`
- DMP RC stats/profile 直接 `printf(...)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:235`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:263`
- DMP RC `debug_on` 路径直接 `printf(...)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:471`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:497`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:575`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:615`
- DMP direct-clock / driving-cell device debug 直接 `printf(...)`：`cpp_to_py/gputimer/core/DmpDebug.cu:194`, `cpp_to_py/gputimer/core/DmpDebug.cu:223`
- DMP driving-cell counts/profile 直接 `printf(...)`：`cpp_to_py/gputimer/core/DmpDebug.cu:243`, `cpp_to_py/gputimer/core/DmpDebug.cu:256`
- DMP debug counts/sample/parallel stats 直接 `fprintf/printf(...)`：`cpp_to_py/gputimer/core/DmpDebug.cu:302`, `cpp_to_py/gputimer/core/DmpDebug.cu:332`, `cpp_to_py/gputimer/core/DmpDebug.cu:386`, `cpp_to_py/gputimer/core/DmpDebug.cu:398`, `cpp_to_py/gputimer/core/DmpDebug.cu:420`, `cpp_to_py/gputimer/core/DmpDebug.cu:542`
- OpenROAD route builder debug/fallback 直接 `fprintf(stderr, ...)`：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:657`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:801`
- OpenROAD RC compare/debug dump 直接 `printf(...)`：`cpp_to_py/gputimer/core/openroad/OpenroadRcDebug.cpp:437`, `cpp_to_py/gputimer/core/openroad/OpenroadRcDebug.cpp:452`, `cpp_to_py/gputimer/core/openroad/OpenroadRcDebug.cpp:473`, `cpp_to_py/gputimer/core/openroad/OpenroadRcDebug.cpp:486`
- SPEF read errors 已改为 `logger.error(...)`：`cpp_to_py/gputimer/core/rc/SpefRc.cpp:20`, `cpp_to_py/gputimer/core/rc/SpefRc.cpp:27`

问题：

- DMP RC 有 progress、kernel profile、debug dump、CUDA error 四类输出混在一起。
- CUDA error helper 在 `.cu` 里可以保留，但格式应统一。
- `debug_on` 的 printf 是典型历史 debug 输出，默认不应该污染 stdout。
- `DMP_PROGRESS_PRINT` 和 `dmp_debug_on` 是两套并行 debug/progress 入口，名字和 env 控制不统一。
- `utils.cuh` 的 generic debug kernels 如果仍被使用，应明确只作为临时 debug 工具；否则容易被误用到默认路径。

状态：

- 已完成 host-side `DmpModel.cpp` progress/debug/unit 输出收敛。
- 已完成 SPEF read error 收敛。
- 未完成：`.cu/.cuh` 内 CUDA error helper、kernel profile、device debug `printf` 仍是下一轮范围。
- 显式 dump API，例如 `[DMP RC DUMP]` / `[SPEF RC DUMP]` / `[GR RC DUMP]`，继续保留 stdout，因为这是用户主动请求的数据输出。

### Power Input / Activity

源码位置：

- direct expression mismatch device debug 直接 `printf(...)`：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:1083`
- `printPowerRowStats(...)` 已改为 `XPLACE_DEBUGF("XPLACE_POWER_ROW_STATS", ...)`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRows.cpp:124`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRows.cpp:131`
- `XPLACE_POWER_DEBUG_NODE` 已改为 `XPLACE_DEBUGF(...)`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRows.cpp:285`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRows.cpp:351`
- power seq/root/debug/upload 已改为 `XPLACE_DEBUGF(...)` 或 `XPLACE_ERRORF(...)`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:341`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:347`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:666`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:677`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:752`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:890`
- power cuda input chunk debug 已改为 `XPLACE_DEBUGF("XPLACE_POWER_DEBUG", ...)`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:383`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:405`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:455`
- power activity frontier / trace 直接 `fprintf(stderr, ...)`：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:407`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:487`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:574`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:595`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:601`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:867`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:874`
- CPU activity trace 已改为 `XPLACE_ERRORF(...)`：`cpp_to_py/gputimer/core/power/activity_cpu/PowerActivityCpu.cpp:252`, `cpp_to_py/gputimer/core/power/activity_cpu/PowerActivityCpu.cpp:735`, `cpp_to_py/gputimer/core/power/activity_cpu/PowerActivityCpu.cpp:869`, `cpp_to_py/gputimer/core/power/activity_cpu/PowerActivityCpu.cpp:890`
- `PowerReport.cpp` direct power profile summary 已改为 `XPLACE_PROFILEF("XPLACE_POWER_PROFILE_STAGES", ...)`：`cpp_to_py/gputimer/core/power/report/PowerReport.cpp:101`, `cpp_to_py/gputimer/core/power/report/PowerReport.cpp:108`

问题：

- power debug 输出很多，但 tag、env、输出流不统一。
- 有些 debug 是文件 dump，有些是 stderr trace，有些是 summary；现在都散在业务逻辑里。
- `printPowerRowStats(...)` 这种 summary 如果默认或常用打开，应该走 logger；如果只给 profile/debug，就走 env-gated debug helper。
- device-side `printf` 不能直接走 host logger，但应该集中在明确的 debug compile/runtime flag 后面，并限制输出数量。

状态：

- 已完成 power input rows/roots/build/report 和 CPU activity trace 的 host-side debug/profile 输出收敛。
- 未完成：`PowerCudaActivity.cu` frontier/trace 和 `PowerCudaActivityDevice.cu` device mismatch `printf` 仍是 CUDA-side 下一轮范围。
- 文件 dump 继续保留 `std::ofstream`；打开失败应走 `logger.warning(...)`，这类小点后续可顺手处理。

### Recommended Order

1. 下一轮先改 `PowerCudaActivity.cu` frontier/trace host-side CUDA wrapper 输出。
2. 再改 `DmpRc.cu` / `DmpDebug.cu` / `DmpTiming.cu` 的 CUDA profile/debug helper，保留 device-side 边界。
3. 最后清理显式 dump API 的打开失败路径，dump 内容本身继续输出到 stdout/file。

## B. Struct / Class / API 设计详情

### B.1 Timing Graph Build

#### `cpp_to_py/gputimer/db/GTDatabase.h`

源码位置：

- `DrivingCellSource`：`cpp_to_py/gputimer/db/GTDatabase.h:88`
- `GTDatabase::extract_profile` 成员：`cpp_to_py/gputimer/db/GTDatabase.h:105`
- `GTDatabase::MarkExtractProfile(...)` 声明：`cpp_to_py/gputimer/db/GTDatabase.h:106`
- `GTDatabase::ExtractTimingGraph()` 声明：`cpp_to_py/gputimer/db/GTDatabase.h:111`
- `GTDatabase::SetupThresholdAndFlattenLib()` 声明：`cpp_to_py/gputimer/db/GTDatabase.h:112`
- `GTDatabase::SetPinMapAndTag(int)` 声明：`cpp_to_py/gputimer/db/GTDatabase.h:113`
- `GTDatabase::AllocatePinArcListStorage(...)` 声明：`cpp_to_py/gputimer/db/GTDatabase.h:114`
- `GTDatabase::WriteNetArcList(...)` 声明：`cpp_to_py/gputimer/db/GTDatabase.h:117`
- `GTDatabase::RunSdcConstantSimulation()` 声明：`cpp_to_py/gputimer/db/GTDatabase.h:128`
- `driving_cell_sources` 成员：`cpp_to_py/gputimer/db/GTDatabase.h:214`
- sparse clock 成员 `clock_periods`：`cpp_to_py/gputimer/db/GTDatabase.h:235`
- sparse clock 成员 `pin_clock_ids` / `test_clock_ids`：`cpp_to_py/gputimer/db/GTDatabase.h:243`, `cpp_to_py/gputimer/db/GTDatabase.h:244`

分类：

- `DrivingCellSource`：header-visible 阶段产物，可以保留。
- `CellTimingArc` / `BuildNetCellArcAndTest(...)` / `WriteCellArcListAndTest(...)`：已从 header API 移出，属于 `GTDatabase.cpp` 内部实现细节。
- `extract_profile` / `MarkExtractProfile(...)`：profile/debug 输出治理问题，已经收敛到 `StageProfiler`，不再作为 struct 设计问题展开。
- sparse clock 成员：跨 timing / DMP / power 的状态数据，属于阶段状态成员，不是临时 tuple。

发现：

```cpp
struct DrivingCellSource { ... };
```

评价：

- `DrivingCellSource` 是 SDC `set_driving_cell` 到 DMP source slew 的阶段产物，可以保留。
- `CellTimingArc` 不再在 `GTDatabase.h` 中出现；header 只保留真实需要跨文件调用的 member function。
- `BuildNetCellArcAndTest(...)`、`WriteCellArcListAndTest(...)` 也不再是 `GTDatabase` public method，而是 `GTDatabase.cpp` anonymous namespace 中的局部 helper。
- `extract_profile` 放在 `GTDatabase` private 成员里是可以接受的：它解决的是“不要给每个业务子函数传 profiler 参数”，不是新增业务状态。

可改：

- 后续如果继续清理，可以把这两个局部 helper 的多 vector 输出合并成明确阶段产物：

```cpp
BuildNetCellArcAndTest(
  graph_threads,
  net_arc_start,
  local_cell_timing_arcs,
  thread_cell_arc_start,
  thread_cell_test_start)
```

可以定义一个明确阶段产物：

```cpp
struct BuiltTimingArcs {
    vector<int> net_arc_start;
    vector<vector<CellTimingArc>> local_cell_timing_arcs;
    vector<int> thread_cell_arc_start;
    vector<int> thread_cell_test_start;
    int num_arcs = 0;
    int num_tests = 0;
    int num_net_arcs = 0;
};
```

然后 `BuildNetCellArcAndTest(...)` 返回这个阶段产物。后续 `WriteNetArcList(...)` / `WriteCellArcListAndTest(...)` 直接吃这个对象，不拆四五个 vector。

不建议：

- 不要把 `CellTimingArc` 放回 `GTDatabase.h`；它只是 build arc 阶段的局部缓存。
- 不要为了减少参数再包一层无业务 wrapper，例如 `foo(result)` 进去后马上拆成 `result.a/result.b` 调另一个旧接口；底层逻辑函数应该直接消费阶段产物。

#### `cpp_to_py/gputimer/db/GTDatabase.cpp`

源码位置：

- `CellTimingArc` 定义：`cpp_to_py/gputimer/db/GTDatabase.cpp:42`
- `BuildNetCellArcAndTest(...)` 局部 helper：`cpp_to_py/gputimer/db/GTDatabase.cpp:225`
- `CellTimingArc` push：`cpp_to_py/gputimer/db/GTDatabase.cpp:326`
- `GTDatabase::AllocatePinArcListStorage(...)` 定义：`cpp_to_py/gputimer/db/GTDatabase.cpp:367`
- `GTDatabase::WriteNetArcList(...)` 定义：`cpp_to_py/gputimer/db/GTDatabase.cpp:391`
- `WriteCellArcListAndTest(...)` 局部 helper：`cpp_to_py/gputimer/db/GTDatabase.cpp:434`
- `GTDatabase::SetupThresholdAndFlattenLib()` 定义：`cpp_to_py/gputimer/db/GTDatabase.cpp:652`
- `GTDatabase::SetPinMapAndTag(int)` 定义：`cpp_to_py/gputimer/db/GTDatabase.cpp:823`
- `GTDatabase::ExtractTimingGraph()` 定义：`cpp_to_py/gputimer/db/GTDatabase.cpp:938`
- `ExtractTimingGraph()` 主线调用：`cpp_to_py/gputimer/db/GTDatabase.cpp:944`, `cpp_to_py/gputimer/db/GTDatabase.cpp:945`, `cpp_to_py/gputimer/db/GTDatabase.cpp:951`, `cpp_to_py/gputimer/db/GTDatabase.cpp:964`, `cpp_to_py/gputimer/db/GTDatabase.cpp:969`, `cpp_to_py/gputimer/db/GTDatabase.cpp:976`
- 旧 `ExtractProfileTimer`：已删除。

发现：

- 旧 `ExtractProfileTimer` 是第三套 profile class，现已删除。
- `ExtractTimingGraph()` 已经被拆成多个函数，主线比以前清楚：

```cpp
SetupThresholdAndFlattenLib(...)
SetPinMapAndTag(...)
BuildNetCellArcAndTest(...)
AllocatePinArcListStorage(...)
WriteNetArcList(...)
WriteCellArcListAndTest(...)
```

状态：

- 已完成：`SetupThresholdAndFlattenLib()` / `SetPinMapAndTag(int)` 不再为了 profile 多传 `std::function<void(const char*)>`。
- 已完成：`CellTimingArc`、`BuildNetCellArcAndTest(...)`、`WriteCellArcListAndTest(...)` 已从 `GTDatabase.h` public API 移到 `GTDatabase.cpp` anonymous namespace。
- 当前做法是 `GTDatabase` 持有 `extract_profile`，子函数内部只调用 `MarkExtractProfile("phase")`。
- 这个方式比把 profiler 参数塞进业务函数签名干净，后续如果再拆 `BuildNetCellArcAndTest(...)`，也应该沿用同样模式。

可改：

- `BuildNetCellArcAndTest` 的返回/输出参数仍可进一步合并成阶段产物，避免“函数看起来像搬运容器”。

状态：

- profile class 合并已完成。
- profile callback 传参已删除。

### B.2 SDC / Sparse Clock

#### `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp`

源码位置：

- `GTDatabase::preparePinNameMapForSdc(...)` 中 `SetClockLatency` target 收集：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:59`
- `GTDatabase::RunSdcConstantSimulation()`：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:72`
- sparse clock 查询 helper 使用 `pin_clock_ids` / `test_clock_ids`：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:83`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:91`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:109`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:127`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:140`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:148`
- `GTDatabase::SetClockLatencyHasUnsupportedMask(...)`：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:159`
- `clock_periods` 清空/填充：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:177`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:201`
- `pin_clock_ids` 分配和传播：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:235`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:270`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:292`
- `test_clock_ids` 分配和映射：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:309`, `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:326`
- `GTDatabase::readSdc(...)`：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:332`
- `for (auto& command : sdc.commands)` visitor：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:348`

发现：

- sparse clock 的后半段逻辑已经是比较清楚的阶段：

```text
read command visitor
  -> build clock id tables
  -> map clock nets/pins/tests
  -> upload sparse clock tensors
```

评价：

- 这段属于 SDC clock 数据流，不是 struct/class 设计问题。
- per-clock debug 输出已经归到 A.2 `SDC / Sparse Clock`，这里不再重复讨论 log。
- `logger.warning("Applied %d pin set_clock_latency overrides...")` 是有价值 summary，可以保留。

#### `cpp_to_py/gputimer/db/sdc/*.cpp`

源码位置：

- `warn_missing_sdc_object(...)` 声明：`cpp_to_py/gputimer/db/sdc/SdcUtils.h:12`
- `warn_missing_sdc_object(...)` 定义，当前已走 `XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_WARNINGS", ...)`：`cpp_to_py/gputimer/db/sdc/SdcUtils.cpp:22`, `cpp_to_py/gputimer/db/sdc/SdcUtils.cpp:24`
- clock SDC handlers：`cpp_to_py/gputimer/db/sdc/SdcClockConstraints.cpp:28`, `cpp_to_py/gputimer/db/sdc/SdcClockConstraints.cpp:58`, `cpp_to_py/gputimer/db/sdc/SdcClockConstraints.cpp:140`, `cpp_to_py/gputimer/db/sdc/SdcClockConstraints.cpp:184`, `cpp_to_py/gputimer/db/sdc/SdcClockConstraints.cpp:239`, `cpp_to_py/gputimer/db/sdc/SdcClockConstraints.cpp:305`
- timing SDC handlers：`cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:29`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:61`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:104`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:137`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:264`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:315`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:349`
- `SetUnits` verbose units 当前已走 `XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_UNITS", ...)`：`cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:53`
- `SetDrivingCell` verbose warning 当前已走 `XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_WARNINGS", ...)`：`cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:188`, `cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp:222`
- exception handlers：`cpp_to_py/gputimer/db/sdc/SdcExceptions.cpp:28`, `cpp_to_py/gputimer/db/sdc/SdcExceptions.cpp:66`

状态：

- SDC verbose units、missing object、driving-cell verbose warning 已归到 A.2 的集中 debug helper。
- 这里保留 handler 文件位置，方便后续审查命令语义；不再把 log 输出问题放在 struct/class 章节里。

### B.3 Route Segment RC

#### `cpp_to_py/gputimer/core/GPUTimer.h::HostRcGraph`

源码位置：

- `HostRcGraph` 定义：`cpp_to_py/gputimer/core/GPUTimer.h:131`
- graph builder 声明 `build_spef_rc()` / `build_openroad_gr_rc(...)` / `build_openroad_route_segments_rc(...)`：`cpp_to_py/gputimer/core/GPUTimer.h:180`, `cpp_to_py/gputimer/core/GPUTimer.h:181`, `cpp_to_py/gputimer/core/GPUTimer.h:182`
- debug dump API 声明：`cpp_to_py/gputimer/core/GPUTimer.h:183`, `cpp_to_py/gputimer/core/GPUTimer.h:184`, `cpp_to_py/gputimer/core/GPUTimer.h:185`
- `GPUTimer::initialize_dmp_rc_explicit(HostRcGraph& graph)` 声明：`cpp_to_py/gputimer/core/GPUTimer.h:395`
- `init_dmp_rc_spef()` / `init_dmp_rc_gr(...)` / `init_dmp_rc_route_segments(...)` 声明：`cpp_to_py/gputimer/core/GPUTimer.h:400`, `cpp_to_py/gputimer/core/GPUTimer.h:401`, `cpp_to_py/gputimer/core/GPUTimer.h:402`
- `DmpModel::initialize_rc_explicit(const HostRcGraph&, float*)` 声明：`cpp_to_py/gputimer/core/DmpModel.h:318`

当前定位：

```cpp
struct HostRcGraph {
    edge_from / edge_to / edge_res
    node_cap
    net2node_start / net2edge_start
    node2pin
    includes_pin_caps
    num_nets / num_nodes / num_edges
};
```

评价：

- 这个 struct 有存在理由：`build_spef_rc()`、`build_openroad_gr_rc()`、`build_openroad_route_segments_rc()` 都产出同一种 host-side RC graph。
- 它应该是阶段产物，不应该伪装成业务 class。
- 关键接口应当是：

```cpp
void GPUTimer::initialize_dmp_rc_explicit(HostRcGraph& graph);
void DmpModel::initialize_rc_explicit(const HostRcGraph& graph, float* pinCap);
```

不应该再出现：

```cpp
initialize_rc_explicit(graph.edge_from,
                       graph.edge_to,
                       graph.net2node_start,
                       ...)
```

已发现/应保留的方向：

- `num_nets` 应属于 `HostRcGraph`，否则 `net2node_start/net2edge_start` 不是自解释数据。
- DMP 最底层执行函数应直接读 `graph.edge_from` 等字段，不要通过 overload 转调旧 vector 接口。

#### `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h`

源码位置：

- `OpenroadRoutePt`：`cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:43`
- `OpenroadRoutePtKey`：`cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:50`
- `OpenroadRoutePtKey::operator==(...)`：`cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:55`
- `OpenroadRoutePtKeyHash`：`cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:61`
- `OpenroadPinRouteLoc`：`cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:71`
- `LocalRcNetGraph`：`cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:88`
- local RC helpers touching `LocalRcNetGraph`：`cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:244`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:247`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:248`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:249`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:250`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:255`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:261`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:264`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:265`
- Host graph cache/load helpers：`cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:190`, `cpp_to_py/gputimer/core/openroad/OpenroadRcInternal.h:197`

发现：

- `OpenroadRoutePt`、`OpenroadRoutePtKey`、`OpenroadPinRouteLoc` 是合理的小数据结构。
- `LocalRcNetGraph` 和 `HostRcGraph` 形态接近：

```cpp
struct LocalRcNetGraph {
    vector<int> edge_from;
    vector<int> edge_to;
    vector<float> edge_res;
    vector<float> node_cap;
    vector<int> node2pin;
    ...
};
```

评价：

- `LocalRcNetGraph` 是 per-net RC graph build 结果，可以保留。
- 原名 `LocalSpefNetRc` 的 `Spef` 不准确，因为它也被 OpenROAD GR / route segment builder 使用；当前已统一改名为 `LocalRcNetGraph`。

可改：

- 和 `HostRcGraph` 的字段命名保持一致。
- 已完成：`append_blank_node(...)` 改成 `append_local_rc_node(...)`。它实际是在 local RC graph 里追加一个 RC node，`pin_id == -1` 时表示没有 pin binding。
- `append_route_node(...)` 不只是 append：它会确保 `local_nets[net_idx]` 存在、查 `route_node_maps`、必要时追加 route node。建议改成 `get_or_append_route_node(...)`，或者拆成 `ensure_local_rc_net(...)` + `get_or_append_route_node(...)`。
- `add_edge(...)` 太泛，建议改成 `add_local_rc_edge(...)`，让调用点直接看出它在改 `LocalRcNetGraph`。
- `count_tree_edges_from_root(...)` / `prune_to_rooted_tree(...)` 默认 root 是 local node 0，这依赖前面的 `reorder_root(local, driver_node)`。建议在命名或注释里写清楚这个前置条件，否则单看 helper 不知道 root 语义。
- `ensure_local_node(...)` 只扩 `node2pin/node_cap` 等数组，不维护 `route_points`；`append_local_rc_node(...)` 会维护 `route_points`。这个 invariant 要写清楚，否则后续容易把两个路径混用。

#### `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp`

源码位置：

- `GPUTimer::build_openroad_route_segments_rc(...)`：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:238`
- `OpenroadRouteSegmentsBuildCounts counts` 主计数器：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:325`
- `scan_stats` / `parse_stats`：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:366`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:490`
- `counts.mergeParseCounts(...)`：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:621`
- `RouteFinalizeWorkerResult` 局部 struct：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:671`
- `finalize_results` / `finalize_one_net`：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:712`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:715`
- `thread_result.counts` 使用点：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:717`
- `finalize_total.merge(...)` / `counts.mergeFinalizeCounts(...)`：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:997`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:1000`
- `final_append_seconds`：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:995`, `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp:1080`

发现：

- `RouteFinalizeWorkerResult` 定义在函数中间，字段多，同时承载 finalize counter、repair summary 和 profile 秒数。
- 已完成：`OpenroadRouteSegmentsBuildStats` / `OpenroadGrRcBuildStats` / `SpefRcBuildStats` 已改成 `*BuildCounts`。

问题：

- `RouteFinalizeWorkerResult` 不是坏 struct。它的存在合理：parallel finalize 时每个 worker 需要局部 accumulator，避免 atomic 更新全局统计。
- 现在的问题是命名和合并规则：`ThreadStats` 太窄，里面混了 build counts、分阶段耗时、repair counters、repair max product；这更像一个 worker 的 finalize result。
- 已完成：`RouteFinalizeWorkerResult` 字段名使用 `counts`，调用点是 `thread_result.counts`。
- 已完成：parse/finalize counter 合并逻辑分别收进 `mergeParseCounts(...)` / `mergeFinalizeCounts(...)` / `RouteFinalizeWorkerResult::merge(...)`。
- `final_append_seconds` 名字不准确；它最后等于 `seconds_since(materialize_start)`，描述的是 materialize/copy 阶段，不是 thread finalize 的 append work。
- 不建议为了“看起来统一”改成 `RouteFinalizeState`。这里没有状态机，也没有生命周期状态；`State` 没有提供额外语义。

当前实现：

```cpp
struct OpenroadRouteSegmentsBuildCounts {
    int parsed_nets = 0;
    int missing_nets = 0;
    int unknown_nets = 0;
    int segment_rows = 0;
    int wire_segments = 0;
    int via_segments = 0;
    int malformed_rows = 0;
    int unknown_layers = 0;
    int non_manhattan_segments = 0;
    int skipped_self_segments = 0;
    int missing_driver_nodes = 0;
    int missing_net_pins = 0;
    int fallback_net_pins = 0;
    int pin_stub_edges = 0;
    int skipped_missing_unconnected_nets = 0;
    long long skipped_missing_unconnected_pins = 0;
    int repaired_edges = 0;
    int skipped_loop_edges = 0;
    int skipped_missing_high_fanout_nets = 0;
    long long skipped_missing_high_fanout_pins = 0;

    void mergeParseCounts(const OpenroadRouteSegmentsBuildCounts& other);
    void mergeFinalizeCounts(const OpenroadRouteSegmentsBuildCounts& other);
};
```

`mergeParseCounts(...)` 替代原来的 `add_route_segment_parse_stats(...)`，只合并 route file parse 阶段字段：

```cpp
segment_rows
wire_segments
via_segments
malformed_rows
unknown_layers
non_manhattan_segments
skipped_self_segments
```

`mergeFinalizeCounts(...)` 对应当前 finalize 完成后的主流程手动累加，合并字段：

```cpp
missing_nets
missing_driver_nodes
missing_net_pins
fallback_net_pins
pin_stub_edges
skipped_missing_unconnected_nets
skipped_missing_unconnected_pins
skipped_missing_high_fanout_nets
skipped_missing_high_fanout_pins
repaired_edges
skipped_loop_edges
```

`RouteFinalizeWorkerResult` 当前完整字段：

```cpp
struct RouteFinalizeWorkerResult {
    OpenroadRouteSegmentsBuildCounts counts;
    double pinloc_seconds = 0.0;   // profile: find/attach pin locations
    double attach_seconds = 0.0;   // profile: attach pin stubs
    double reorder_seconds = 0.0;  // profile: move driver/root to local node 0
    double repair_seconds = 0.0;   // profile: repair/prune preparation
    double prune_seconds = 0.0;    // profile: prune to rooted tree
    int repair_adjacency_nets = 0;
    int repair_scan_nets = 0;
    long long repair_node_edge_product_max = 0;

    void merge(const RouteFinalizeWorkerResult& other);
};
```

`RouteFinalizeWorkerResult::merge(...)` 应该做两件事：

```cpp
counts.mergeFinalizeCounts(other.counts);
pinloc_seconds += other.pinloc_seconds;
attach_seconds += other.attach_seconds;
reorder_seconds += other.reorder_seconds;
repair_seconds += other.repair_seconds;
prune_seconds += other.prune_seconds;
repair_adjacency_nets += other.repair_adjacency_nets;
repair_scan_nets += other.repair_scan_nets;
repair_node_edge_product_max = std::max(repair_node_edge_product_max,
                                        other.repair_node_edge_product_max);
```

调用位置按当前源码对应如下：

```cpp
// current: OpenroadRouteSegmentsBuilder.cpp:712
std::vector<RouteFinalizeWorkerResult> finalize_results(finalize_threads);

// current: OpenroadRouteSegmentsBuilder.cpp:715 and parallel caller below it
finalize_one_net(net_idx, finalize_results[tid]);

// current: OpenroadRouteSegmentsBuilder.cpp:996
RouteFinalizeWorkerResult finalize_total;
for (const RouteFinalizeWorkerResult& worker : finalize_results) {
    finalize_total.merge(worker);
}

// global build counters
counts.mergeFinalizeCounts(finalize_total.counts);

// profile output still belongs to A section
route_profile.markf("finalize_done", ... finalize_total.pinloc_seconds ...);
```

状态：

- profile 输出统一已完成：route/cache/finalize profile 都走 `StageProfiler::mark(...)` / `markf(...)`。
- 已完成：`OpenroadRouteSegmentsBuildStats` -> `OpenroadRouteSegmentsBuildCounts`。
- 已完成：`OpenroadGrRcBuildStats` -> `OpenroadGrRcBuildCounts`。
- 已完成：`SpefRcBuildStats` -> `SpefRcBuildCounts`。
- 已完成：`RouteFinalizeThreadStats` -> `RouteFinalizeWorkerResult`，字段 `stats` 改为 `counts`。
- 已完成：`add_route_segment_parse_stats(...)` 删除，逻辑进入 `OpenroadRouteSegmentsBuildCounts::mergeParseCounts(...)`。
- `RouteFinalizeWorkerResult` 仍在函数中间，属于 struct/组织问题，本轮未移动。

#### `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp`

源码位置：

- `RouteSegmentCacheHeader`：`cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:17`
- `load_route_segment_cache(...)`：`cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:131`
- cache load `StageProfiler`：`cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:140`
- cache load profile marks：`cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:146`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:163`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:178`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:182`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:186`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:190`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:194`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:198`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:202`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:206`, `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:209`
- `save_route_segment_cache(...)`：`cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp:230`
- 旧 cache load `log_profile` lambda：已删除。

发现：

- 旧 cache load 里又定义一套 `log_profile` lambda。
- 旧实现中，它和 route segment builder 的 `cache_profile_log` 输出 tag 类似但实现独立。

可改：

- 已完成：cache 文件读使用统一 `StageProfiler`。
- `RouteSegmentCacheHeader` 与 `HostRcGraph` 字段强相关，加载时必须设置 `graph.num_nets`。

#### `cpp_to_py/gputimer/core/rc/SpefRc.cpp`

源码位置：

- `SpefRcBuildCounts`：`cpp_to_py/gputimer/core/rc/SpefRc.cpp:61`
- `LocalRcNetGraph`：`cpp_to_py/gputimer/core/rc/SpefRc.cpp:77`
- `count_tree_edges_from_root(...)`：`cpp_to_py/gputimer/core/rc/SpefRc.cpp:141`
- `GPUTimer::build_spef_rc()`：`cpp_to_py/gputimer/core/rc/SpefRc.cpp:172`
- `local_nets` 构建：`cpp_to_py/gputimer/core/rc/SpefRc.cpp:197`
- `GPUTimer::debug_dump_spef_rc_net(...)`：`cpp_to_py/gputimer/core/rc/SpefRc.cpp:427`
- debug dump `printf(...)`：`cpp_to_py/gputimer/core/rc/SpefRc.cpp:445`, `cpp_to_py/gputimer/core/rc/SpefRc.cpp:452`, `cpp_to_py/gputimer/core/rc/SpefRc.cpp:460`

发现：

- `SpefRcBuildCounts` 是构建统计，`LocalRcNetGraph` 是 per-net RC graph build result。
- 构图结果也应填完整 `HostRcGraph::num_nets`。
- debug dump 用 `printf`，正常 dump API 可以保留，但错误/warning 应走 logger。

可改：

- 已完成：SPEF parser 内部结构也改成 `LocalRcNetGraph`，与 OpenROAD local RC graph 命名一致。
- 已完成：`SpefRcBuildStats` 改成 `SpefRcBuildCounts`。这里没有状态机语义，不应该改成 `State`。
- `debug_dump_*` 输出可以继续 `printf`，因为它是显式 debug API。

### B.4 DMP RC / Timing

#### `cpp_to_py/gputimer/core/rc/DmpRc.cu`

源码位置：

- DMP RC env helper 使用 `xplace_env_enabled(...)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:33`
- model init `StageProfiler`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:343`
- wrapper init `StageProfiler`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:432`
- 旧 `DmpRcPhaseProfile`：已删除。
- DMP RC progress/error `std::fprintf(stderr, ...)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:66`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:88`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:104`
- `print_dmp_rc_parallel_stats(...)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:205`
- `print_dmp_rc_kernel_profile(...)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:244`
- `DmpModel::initialize_rc_explicit(const HostRcGraph&, float*)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:341`
- wrapper-side graph call and profile marks：`cpp_to_py/gputimer/core/rc/DmpRc.cu:437`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:438`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:447`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:450`
- kernel profile call sites：`cpp_to_py/gputimer/core/rc/DmpRc.cu:657`, `cpp_to_py/gputimer/core/rc/DmpRc.cu:706`
- `debug_dump_dmp_rc_net_cuda(...)`：`cpp_to_py/gputimer/core/rc/DmpRc.cu:720`
- DMP timing kernel profile struct / print：`cpp_to_py/gputimer/core/DmpTiming.cu:47`, `cpp_to_py/gputimer/core/DmpTiming.cu:113`
- DMP timing profile print sites：`cpp_to_py/gputimer/core/DmpTiming.cu:693`, `cpp_to_py/gputimer/core/DmpTiming.cu:705`, `cpp_to_py/gputimer/core/DmpTiming.cu:779`
- DMP LUT profile print：`cpp_to_py/gputimer/core/GPUTimer.cu:90`, `cpp_to_py/gputimer/core/GPUTimer.cu:235`
- DMP root solve profile print：`cpp_to_py/gputimer/core/DmpDebug.cu:101`, `cpp_to_py/gputimer/core/DmpDebug.cu:118`, `cpp_to_py/gputimer/core/DmpDebug.cu:125`, `cpp_to_py/gputimer/core/DmpDebug.cu:130`, `cpp_to_py/gputimer/core/DmpDebug.cu:136`
- DMP driving-cell kernel profile print：`cpp_to_py/gputimer/core/DmpDebug.cu:255`, `cpp_to_py/gputimer/core/DmpGateEval.cu:1256`

发现：

- 旧 `DmpRcPhaseProfile` 是第四套 profile class，现已删除。
- `DmpModel::initialize_rc_explicit(...)` 之前是十几个 vector 参数，是本轮最需要清理的接口。
- `print_dmp_rc_parallel_stats(...)` 和 `print_dmp_rc_kernel_profile(...)` 直接 `printf`。
- `DmpTiming.cu`、`DmpDebug.cu`、`GPUTimer.cu` 仍有 CUDA/kernel/profile 输出，当前不是 class/lambda 样板，但输出格式和开关分散。

可改目标：

```cpp
__host__ void DmpModel::initialize_rc_explicit(const HostRcGraph& graph,
                                               float* pinCap);
```

函数体中直接：

```cpp
graph.edge_from
graph.edge_to
graph.net2node_start
graph.net2edge_start
graph.node2pin
graph.edge_res
graph.node_cap
graph.includes_pin_caps
graph.num_nets
graph.num_nodes
graph.num_edges
```

不要保留 “graph overload -> vector overload”。

profile 可改：

- 已完成：`DmpRcPhaseProfile` 和 `ExtractProfileTimer` / `PowerStageProfiler` 合并到 `StageProfiler`。
- kernel/LUT/root/driving-cell profile 输出如果保留，至少用短宏封装：

```cpp
DMP_RC_PROFILEF("calc_dmp_rc name=%s ...", ...);
```

#### `cpp_to_py/gputimer/core/DmpModel.cpp`

源码位置：

- `debug_dump_dmp_rc_net_cuda(...)` 前置声明：`cpp_to_py/gputimer/core/DmpModel.cpp:93`
- `GPUTimer::init_dmp_rc_spef()`：`cpp_to_py/gputimer/core/DmpModel.cpp:340`
- `GPUTimer::init_dmp_rc_gr(...)`：`cpp_to_py/gputimer/core/DmpModel.cpp:356`
- `GPUTimer::init_dmp_rc_route_segments(...)`：`cpp_to_py/gputimer/core/DmpModel.cpp:373`
- `GPUTimer::debug_dump_dmp_rc_net(...)`：`cpp_to_py/gputimer/core/DmpModel.cpp:390`
- `debug_dump_dmp_rc_net_cuda(...)` 调用：`cpp_to_py/gputimer/core/DmpModel.cpp:407`

发现：

- `init_dmp_rc_spef()`、`init_dmp_rc_gr()`、`init_dmp_rc_route_segments()` 三段高度重复。
- 以前 route segment 入口里有 `DmpRcStageProfile rc_profile` 和多处 `rc_profile.log(...)`，这类写法不应回到主流程。

可改：

```cpp
void GPUTimer::run_dmp_rc(int num_nets, bool update_timing_after_rc);
```

`run_dmp_rc(...)` 只负责跑 DMP RC 计算和可选 timing 准备，不负责打印入口类型。`SPEF` / `GR` / `route-segment` 这些来源说明应留在外层入口里打 log，不要作为 `tag` 传进去。

内部逻辑：

```cpp
void GPUTimer::run_dmp_rc(int num_nets, bool update_timing_after_rc) {
    calc_res_cap_dmp(dmp_db, num_nets);
    propagate_rc_tree_dmp(dmp_db, num_nets);
    if (update_timing_after_rc) {
        dmp_prepare_timing_after_rc(h_dmp_db, dmp_db);
    }
    apply_dmp_driving_cell_source_slew(*this);
}
```

入口函数保留来源相关 log：

```cpp
HostRcGraph graph = build_openroad_route_segments_rc(file);
const int graph_num_nets = graph.num_nets;
const int graph_num_nodes = graph.num_nodes;
const int graph_num_edges = graph.num_edges;
initialize_dmp_rc_explicit(graph);
graph.release_storage();
DMP_PROGRESS_PRINT("DMP route-segment RC calculation starting, num_nets: %d num_nodes: %d num_edges: %d",
                   graph_num_nets, graph_num_nodes, graph_num_edges);
run_dmp_rc(graph_num_nets, true);
DMP_PROGRESS_PRINT("DMP route-segment RC propagation done.");
```

`update_timing_after_rc` 的含义：

- `false`：只把 RC 结果算完，保持当前 `init_dmp_rc_spef()` 行为。
- `true`：RC propagation 之后调用 `dmp_prepare_timing_after_rc(h_dmp_db, dmp_db)`，保持当前 `init_dmp_rc_gr(...)` 和 `init_dmp_rc_route_segments(...)` 行为。

`dmp_prepare_timing_after_rc(...)` 做的不是 timing propagation。它是 RC 阶段和 timing 阶段之间的 DMP model 内存切换：

```cpp
cudaMemcpy(&device_state, dmp_db, sizeof(DmpModel), cudaMemcpyDeviceToHost);
std::memcpy(h_dmp_db, &device_state, sizeof(DmpModel));
h_dmp_db->release_rc_transient();
h_dmp_db->allocate_timing_scratch();
cudaMemcpy(dmp_db, h_dmp_db, sizeof(DmpModel), cudaMemcpyHostToDevice);
```

具体效果：

- 从 device 拷回最新 `DmpModel` descriptor，让 host 端 `h_dmp_db` 拿到 RC kernel 更新后的 device pointer / metadata。
- `release_rc_transient()` 释放 RC 构图和 RC propagation 阶段才需要的显存：`edge_from`、`edge_to`、`flat_net2node_start_map`、`flat_net2edge_start_map`、`node2pin_map`、`edge_res`、`node_cap`、`includes_pin_caps`、`root_dist`、`cnts`、`node_order`、`parent_node`、`res_parent`、`node_delay`、`y1/y2/y3/down_cap` 等。
- `allocate_timing_scratch()` 分配 timing 阶段需要的 scratch，目前主要是 `pin_at_winner`。
- 再把更新后的 descriptor 拷回 device，让后续 timing kernel 用释放/重分配后的正确指针。

### B.5 Power CUDA Input / Activity

#### View / Scratch / State 命名扫描结论

本轮只扫描 `cpp_to_py/gputimer` 范围内的声明级 `struct/class` 名字，命中如下：

```text
RcTreeDeviceGraph
RcTreePropagation
RcStarNet
RcTreeHost
RcTreeDevice
PowerGraphDevice
PowerExprDevice
PowerActivitySeedDevice
PowerComponentDevice
PowerActivityPropDevice
PowerActivityLevelQueueDevice
PowerTracePathWriter
PowerExprEval
```

判断：

- RC 侧已按业务边界改名为 `RcStarNet` / `RcTreeHost` / `RcTreeDeviceGraph` / `RcTreePropagation` / `RcTreeDevice`，不再使用 `Model` / `View` / `Scratch` / `Explicit`。
- `*DeviceView` / `PowerExprEval` 仍是 power 侧待清理对象；它们目前是 CUDA kernel 读 device pointer 的轻量 view，不拥有数据。
- `State` / `Scratch` / `ScratchView` 需要警惕。这里大多不是状态机，也不是“临时变量”这么简单，而是有明确业务用途的 device buffer bundle。
- 替代命名应表达用途，不要只说容器形态。

已完成命名表：

| 当前名字 | 位置 | 当前用途 | 状态 | 理由 |
| --- | --- | --- | --- | --- |
| `RcStarNet` | `cpp_to_py/gputimer/core/rc/RcModels.h:8` | star RC kernel 的 net/pin coordinate、load、delay、unit 参数包 | 已完成 | 强调这是 star-net RC 计算输入，不是抽象模型。 |
| `RcTreeHost` | `cpp_to_py/gputimer/core/rc/RcModels.h:50` | host wrapper 输入，引用 host vectors 和 torch/device output pointers | 已完成 | 表达 CPU 侧传入的 RC tree 数据。 |
| `RcTreeDeviceGraph` | `cpp_to_py/gputimer/core/rc/RcModels.h:30` | 已 copy/malloc 到 GPU 的 RC tree 拓扑数组，被 `RcTreeDevice::graph` 持有 | 已完成 | 它不是 view，而是 device graph arrays。 |
| `RcTreePropagation` | `cpp_to_py/gputimer/core/rc/RcModels.h:42` | RC propagation kernel 的 node load/delay/impulse/beta 中间数组，被 `RcTreeDevice::propagation` 持有 | 已完成 | 这些数组是传播阶段数据，不是泛 scratch。 |
| `RcTreeDevice` | `cpp_to_py/gputimer/core/rc/RcModels.h:126` | CUDA kernel 实际接收的完整 RC tree 参数包，包含 graph、propagation 和输出数组 | 已完成 | 表达它是 device-side RC tree kernel data。 |
| `PowerGraphDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:10` | power activity kernel 访问 timing graph、netlist、clock slew、DMP load 的 device pointer bundle | 已完成 | 核心内容是 power activity 使用的 graph/device 数据；去掉 `View` 泛词。 |
| `PowerExprDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:114` | Liberty function expression op table、expr range、node-port-pin map | 已完成 | 这是 expression 相关 device tables；名字保持短，和 `PowerExprEval` 配套。 |
| `PowerActivitySeedDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:143` | activity seed 输入：PI、case value、clock pins、seqs、feedback seeds | 已完成 | 它不是状态机；它提供 activity propagation 的 seed/input 集合。 |
| `PowerComponentDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:228` | internal/switching/leakage rows、allocator 和输出数组 | 已完成 | 主要是 power component row tables 和对应输出指针；去掉 `View`。 |
| `PowerActivityDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:284` | activity CUDA launcher 的顶层参数包，持有 graph/expr/seed/config/component/out | 已完成 | 顶层 device 参数包，不是算法 model。`Cuda` 和 `Device` 二选一，保留 `Device` 和其它 power 参数包统一。 |
| `PowerActivityPropDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:313` | activity propagation 输出/工作数组：density、duty、origin、active flags、pending seq 等 | 已完成 | 这是 activity propagation 的 device 工作区和传播结果，和 seed 输入要分开。 |
| `PowerActivityLevelQueueDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:365` | level queue、queued bitmap、overflow、pending seq queue | 已完成 | 这是 levelized activity propagation queue，`Level` 是算法约束，不是废词。 |
| `PowerInternalDenomDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:391` | internal power denominator chunk kernel 的输入/输出指针包 | 已完成 | 这是 internal denom kernel 的 device 参数包，不是抽象 model。 |
| `PowerInternalInstDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:432` | internal power contribution chunk kernel 的 activity、slew/load、denom、allocator 和输出指针包 | 已完成 | 这一阶段最终写 `inst_internal` / `internal_row_power`，名字直接表达 instance internal power 输出。 |
| `PowerLeakageCondDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:518` | leakage row condition kernel 的输入/输出指针包 | 已完成 | 这一阶段逐条计算 leakage `when` 条件 duty，并累加到 group condition buffers。 |
| `PowerLeakageInstDevice` | `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:568` | leakage group summary kernel 的输入/输出指针包 | 已完成 | 这一阶段把 leakage group condition 结果累加到 `inst_leakage`。 |
| `PowerTracePathWriter` | `cpp_to_py/gputimer/core/power/common/PowerHostCommon.h:29` | host-side trace path filter/output：目标 pins/arcs/nodes 和输出文件 | 已完成 | 里面带 `ofstream out`，核心职责是过滤 trace target 并写 trace path，不是状态机。 |
| `PowerExprEval` | `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cuh:112` | device expression evaluator，绑定 expr table、pin activity、node context | 已完成 | 它有 `evalBool/activity/duty/diffDuty` 行为，不只是 pointer view；名字和 `PowerExprDevice` 配套。 |

调用/使用关系：

- `RcTreeDeviceGraph` / `RcTreePropagation` 被 `RcTreeDevice` 持有：`cpp_to_py/gputimer/core/rc/RcModels.h:126`。kernel 里通过 `m.graph` / `m.propagation` 访问：`cpp_to_py/gputimer/core/rc/ExplicitRcTree.cu:14`, `cpp_to_py/gputimer/core/rc/ExplicitRcTree.cu:107`, `cpp_to_py/gputimer/core/rc/ExplicitRcTree.cu:165`, `cpp_to_py/gputimer/core/rc/ExplicitRcTree.cu:315`。
- `PowerGraphDevice` / `PowerExprDevice` / `PowerActivitySeedDevice` / `PowerComponentDevice` 在 `PowerCudaInputBuild.cpp` 里组装：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:808`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:841`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:850`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:877`，然后塞进 `PowerActivityDevice`。计划改名后，顶层包叫 `PowerActivityDevice`。
- `PowerActivityPropDevice` 在 main activity launch 和 chunk launch 里作为 density/duty/active/pending-seq 工作区：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:222`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:238`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaChunkLaunchers.cu:66`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaChunkLaunchers.cu:100`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaChunkLaunchers.cu:129`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaChunkLaunchers.cu:160`。
- `PowerActivityLevelQueueDevice` 在 frontier queue 模式里使用：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:542`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityQueue.cu:152`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityQueue.cu:164`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityQueue.cu:177`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityQueue.cu:195`。
- `PowerTracePathWriter` 由 `loadPowerTracePathWriter(...)` 构造：`cpp_to_py/gputimer/core/power/common/PowerHostCommon.h:253`，CPU activity trace 入口使用：`cpp_to_py/gputimer/core/power/activity_cpu/PowerActivityCpu.cpp:77`。
- `PowerExprEval` 由 seed/queue/component kernels 临时构造，用来 eval/duty/diffDuty：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivitySeeds.cu:123`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityQueue.cu:41`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityComponents.cu:86`。

完成状态：

1. Activity propagation 相关类型已改完：`PowerActivityDevice`、`PowerActivitySeedDevice`、`PowerActivityPropDevice`、`PowerActivityLevelQueueDevice`。
2. Power CUDA 顶层数据包已改完：`PowerGraphDevice`、`PowerExprDevice`、`PowerComponentDevice`。
3. Chunk component kernel 参数包已改完：`PowerInternalDenomDevice`、`PowerInternalInstDevice`、`PowerLeakageCondDevice`、`PowerLeakageInstDevice`。
4. Device-side expression evaluator 已改为 `PowerExprEval`。
5. Host trace path 输出对象已改为 `PowerTracePathWriter`。

#### `cpp_to_py/gputimer/core/power/common/PowerCudaModel.h`

源码位置：

- `PowerGraphDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:10`
- `PowerGraphDevice()` default constructor：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:44`
- `PowerGraphDevice(...)` 长参数构造函数：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:45`
- `PowerExprDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:114`
- `PowerExprDevice::pin_expr_id` 字段：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:120`
- `PowerExprDevice(...)` 构造参数 `pin_expr_id_`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:130`
- `PowerExprDevice(...)` 初始化 `pin_expr_id(...)`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:138`
- `PowerActivitySeedDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:143`
- `PowerActivityConfig`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:196`
- `PowerActivityConfig()` default constructor：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:207`
- `PowerActivityConfig(...)` 构造函数：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:208`
- `PowerComponentDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:228`
- `PowerActivityDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:284`
- `PowerActivityPropDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:313`
- `PowerActivityPropDevice()` default constructor：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:330`
- `PowerActivityLevelQueueDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:365`
- `PowerInternalDenomDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:391`
- `PowerInternalInstDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:432`
- `PowerLeakageCondDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:518`
- `PowerLeakageInstDevice`：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:568`
- `run_power_internal_denom_chunk_cuda_launcher(...)` 参数：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:610`
- `run_power_internal_contrib_chunk_cuda_launcher(...)` 参数：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:611`
- `run_power_leakage_rows_chunk_cuda_launcher(...)` 参数：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:612`
- `run_power_leakage_summary_chunk_cuda_launcher(...)` 参数：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h:613`

当前剩余可改点：

1. `PowerGraphDevice` 构造函数参数非常长，和旧 `initialize_rc_explicit(vector...)` 是同一种问题。
2. `PowerExprDevice::pin_func_expr_id` 已改为 `pin_expr_id`。
3. `Device` 后缀当前保留。原因是这些类型由 host builder 组装、拷贝或传入 CUDA kernel，和 host-side vectors/tensors 混在同一文件里；`Device` 能明确“字段是 device pointer / CUDA kernel 参数”。如果后续把这些类型都收进 `cuda_activity` namespace 且没有 host 对应类型，再考虑删除 `Device`。
4. 旧审查里提到的 `PowerActivityConfig` 重复初始化、`PowerActivityPropDevice` 重复 default constructor、`PowerInternalDenomDevice` 多余 `};`，当前源码未复现；当前保留关注点是这些 device struct 的长构造函数风格。

建议模式：

- Device view struct 可以保留，因为 kernel 入口需要一个 compact view。
- 但不要写超长构造函数。用默认构造 + builder 逐字段填：

```cpp
PowerGraphDevice graph_view;
graph_view.level_list = ...;
graph_view.pin_forward_arc_list = ...;
...
```

或者集中到一个 `buildPowerGraphDevice(...)`，函数体里直接填字段，不把 30 个参数暴露在调用点。

#### `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h`

源码位置：

- `PowerStageProfiler`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h:26`
- `PowerStageProfiler::PowerStageProfiler(...)` 声明：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h:28`
- `PowerStageProfiler` 持有统一 profiler：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h:32`
- `PowerCudaRunBuffers`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h:55`
- `PowerDmpLoadPointers`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h:100`
- `PowerClockPinActivity`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h:116`
- `PowerCudaRootInputs`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h:170`
- `PowerStageProfiler` 定义/输出：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:174`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:177`

发现：

- `PowerCudaRunBuffers`、`PowerDmpLoadPointers`、`PowerClockPinActivity`、`PowerCudaRootInputs` 等是阶段产物，可以保留。
- `PowerStageProfiler` 保留类型名和 `profile.mark(...)` 调用方式，但内部已经改成 `StageProfiler`，不再自己维护 chrono/fprintf。

可改：

- 已完成：全局通用 profiler 已移到 `cpp_to_py/common/StageProfiler.h`，`PowerStageProfiler` 变成薄包装。
- `PowerCudaRootInputs` 字段很多，但它是真正的 root selection 结果。可保留，后续按“输入/输出/统计”分段整理字段顺序。

#### `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp`

源码位置：

- `preparePowerCudaRunBuffers(...)` 定义：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:194`
- `finishPowerActivityOutputs(...)` 定义：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:496`
- `GPUTimer::compute_power_activity_cuda(...)` 定义：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:526`
- `PowerStageProfiler profile(...)` 创建：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:545`
- `preparePowerCudaRunBuffers(...)` 调用点：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:769`
- `finishPowerActivityOutputs(...)` 调用点：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp:937`
- report 入口调用：`cpp_to_py/gputimer/core/power/report/PowerReport.cpp:40`, `cpp_to_py/gputimer/core/power/report/PowerReport.cpp:46`, `cpp_to_py/gputimer/core/power/report/PowerReport.cpp:52`, `cpp_to_py/gputimer/core/power/report/PowerReport.cpp:60`, `cpp_to_py/gputimer/core/power/report/PowerReport.cpp:66`, `cpp_to_py/gputimer/core/power/report/PowerReport.cpp:74`, `cpp_to_py/gputimer/core/power/report/PowerReport.cpp:84`
- direct power profile output in report path 已改为 `XPLACE_PROFILEF(...)`：`cpp_to_py/gputimer/core/power/report/PowerReport.cpp:102`, `cpp_to_py/gputimer/core/power/report/PowerReport.cpp:109`
- power input host-side debug 已改为 `XPLACE_DEBUGF(...)` / `XPLACE_ERRORF(...)`：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputExpr.cpp:176`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRows.cpp:124`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRows.cpp:285`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:342`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:666`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:753`, `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputRoots.cpp:890`

发现：

- `compute_power_activity_cuda(...)` 主干已经大量使用 `profile.mark("...")`，这比 route segment builder 好。
- power stage profile 输出已统一成 `phase=... elapsed=... total=...`。
- `preparePowerCudaRunBuffers(...)` / `finishPowerActivityOutputs(...)` 已经不再传 `PowerStageProfiler& profile`，profile mark 留在上层 orchestration。
- build/rows/roots 子文件的 host-side debug/error 输出已经收敛；CUDA activity 文件里仍有 CUDA-side `fprintf/printf`。

可改：

- 保留 `profile.mark` 模式。
- 后续只处理 CUDA-side debug helper，不再在 power input host 侧新加裸 `fprintf/cerr`。

状态：

- stage profile 已完成：`PowerStageProfiler` 使用 `StageProfiler`，`PowerReport.cpp` summary 也统一格式。
- host-side debug/error 输出宏化已完成。

#### `cpp_to_py/gputimer/core/power/cuda_activity/*`

源码位置：

- `PowerActivityOps` device methods：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:24`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:29`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:39`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:78`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:110`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:115`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:143`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:193`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:206`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:222`
- `PowerExprEval` device methods：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:238`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:1042`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:1052`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:1098`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:1115`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:1131`
- `PowerBddContextCuda`：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:348`
- `PowerBddContextCuda` methods：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:378`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:379`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:380`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:382`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:409`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:414`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:421`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:458`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:474`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:491`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:523`
- `PowerBddExprEval` / `PowerDirectExprEval`：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:531`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:535`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:673`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivityDevice.cu:675`
- frontier/debug `fprintf(stderr, ...)` examples：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:407`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:487`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:574`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:595`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:601`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:867`, `cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu:874`

发现：

- `PowerActivityOps`、`PowerExprEval`、BDD/direct evaluator structs 是 device-side behavior structs，不是单纯 data bag。保留。
- `PowerBddContextCuda` 等是 device evaluator 状态，也可保留。
- 多处 `fprintf(stderr, "[power_frontier] ...")` 是 debug/unsupported path 输出，建议统一宏。

### B.6 Route Gradient

#### `cpp_to_py/gputimer/core/route_grad/*`

源码位置：

- `RouteGradDeviceFloatBuffer`：`cpp_to_py/gputimer/core/route_grad/DmpRouteGradHost.h:20`
- `RouteGradDeviceIntBuffer`：`cpp_to_py/gputimer/core/route_grad/DmpRouteGradHost.h:31`
- `RouteGradDeviceU64Buffer`：`cpp_to_py/gputimer/core/route_grad/DmpRouteGradHost.h:42`
- `RouteGradNetSlopesHost`：`cpp_to_py/gputimer/core/route_grad/DmpRouteGradHost.h:53`
- `RouteGradActiveGateSlopesHost`：`cpp_to_py/gputimer/core/route_grad/DmpRouteGradHost.h:70`
- `RouteGradGateSlewWinnerSlopesHost`：`cpp_to_py/gputimer/core/route_grad/DmpRouteGradHost.h:82`
- `run_route_segment_dmp_for_route_grad(...)` 声明：`cpp_to_py/gputimer/core/route_grad/DmpRouteGradInternal.h:26`
- `run_route_segment_dmp_for_route_grad(...)` 定义：`cpp_to_py/gputimer/core/route_grad/DmpRouteGrad.cu:107`
- route-grad graph size checks using `graph.num_nets` / `graph.net2edge_start`：`cpp_to_py/gputimer/core/route_grad/DmpRouteGrad.cu:154`, `cpp_to_py/gputimer/core/route_grad/DmpRouteGrad.cu:308`, `cpp_to_py/gputimer/core/route_grad/DmpRouteGrad.cu:368`
- route-grad calls into DMP run：`cpp_to_py/gputimer/core/route_grad/DmpRouteGrad.cu:158`, `cpp_to_py/gputimer/core/route_grad/DmpRouteGrad.cu:312`

发现：

- `RouteGradDeviceFloatBuffer`、`RouteGradDeviceIntBuffer`、`RouteGradDeviceU64Buffer` 有 RAII-ish 行为，比裸 device pointer 好。
- `RouteGrad*SlopesHost` 是 host result bundle，可以保留。
- `run_route_segment_dmp_for_route_grad(...)` 也应直接 `timer.initialize_dmp_rc_explicit(graph)`，不拆 graph fields。

可改：

- route grad helper 中不要再用 `graph.net2edge_start.size() - 1` 推 `num_nets`；既然 `HostRcGraph` 有 `num_nets`，统一使用 `graph.num_nets`。

## C. High Priority Cleanup List

### C.1 Struct / Class / API

1. `HostRcGraph` / `DmpModel::initialize_rc_explicit`：保留 graph 阶段产物，底层函数直接消费 `graph`，删除 explicit vector 参数接口。
2. `PowerCudaModel.h`：清理重复字段/重复初始化/重复 default constructor；长构造函数改成默认构造后填字段或 builder。
3. `OpenroadRouteSegmentsBuilder.cpp`：route/cache/finalize profile 输出已统一为 `StageProfiler`；剩余是移走函数中间的大 `RouteFinalizeWorkerResult`。
4. `GTDatabase` cell timing arc 中间结果：把 `net_arc_start/local_cell_timing_arcs/thread_*` 合并为一个明确阶段产物，减少函数参数。

### C.2 Profile / Debug

1. 全链路 profile class 收敛：已完成。`XplaceIoProfileTimer`、`ExtractProfileTimer`、`DmpRcPhaseProfile`、GPDB local lambda、pybind local lambda 已删除；`PowerStageProfiler` 改为 `StageProfiler` 包装。
2. host-side debug/log 输出收敛：SDC / infer / parser DEF fast path / power input host path 已完成。
3. 下一步先统一 `PowerCudaActivity.cu` frontier/trace，再处理 `DmpRc.cu` / `DmpDebug.cu` / `DmpTiming.cu` 的 CUDA profile/debug helper。
4. 显式 dump API 最后处理：dump 内容本身可以继续 stdout/file，但打开失败、fallback、summary 应走 logger。

## D. Recommended Naming / Interface Rules

阶段产物：

```cpp
HostRcGraph graph = build_openroad_route_segments_rc(file);
initialize_dmp_rc_explicit(graph);
```

底层执行：

```cpp
void DmpModel::initialize_rc_explicit(const HostRcGraph& graph, float* pinCap);
```

不要：

```cpp
initialize_rc_explicit(graph.edge_from,
                       graph.edge_to,
                       graph.net2node_start,
                       ...);
```

device view：

```cpp
PowerGraphDevice view;
view.level_list = ...;
view.pin_forward_arc_list = ...;
```

不要：

```cpp
PowerGraphDevice view(a, b, c, d, e, f, ..., z);
```

profile：

```cpp
PROFILE_MARK(profile, "phase");
PROFILE_PRINTF(profile, "phase=%s nodes=%d", phase, nodes);
```

不要：

```cpp
if (profile) {
    std::fprintf(stderr, ...);
    std::fflush(stderr);
}
```
