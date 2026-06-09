# GPUTimer Power 总架构

Last reviewed: 2026-06-08

本文只保留全局脉络、模块边界和 power 主干。Data loader 的逐层端口展开放在 `01_DATA_LOADER.md`；`build_rc` 的 route-segment RC/CUDA 端口展开放在 `03_BUILD_ROUTING_RC.md`。

## 1. 端到端脉络

```text
tools/compare_ispd25_route_power_timing.py
  main()
    case_iter(args)
    run_openroad(...)
      OpenROAD Tcl runs timing + report_power
      parse_openroad_log(...) reads OpenROAD report_power total/group rows
    run_xplace_parent(...)
      starts the same script as --worker xplace
        run_xplace_worker(args)
          run_timer.getArgs()
          load_design(...)
          data.to_timing_device(...).preprocess_timing()
          timer_only.timing_opt.GPUTimer(...)
            gputimer.create_timing_rawdb(...)
            gputimer.create_gputimer(...)
              C++ create_gputimer(...)
                GTDatabase(rawdb, gpdb, timing_raw_db)
                SDC::read(...)
                GTDatabase::preparePinNameMapForSdc(...)
                GTDatabase::ExtractTimingGraph()
                GTDatabase::readSdc(...)
                gt::GPUTimer(gtdb, timing_raw_db)
            gt::GPUTimer::initialize()
            gt::GPUTimer::levelize()
          build_rc stage
            gt::GPUTimer::update_states()
            gt::GPUTimer::init_dmp_rc_route_segments(route_segments)
          timer stage
            gt::GPUTimer::update_timing_dmp()
            timer_only.GPUTimer.report_timing_slack()
              gt::GPUTimer::update_endpoints()
              gt::GPUTimer::report_wns_and_tns()
          optional release stage
            gt::GPUTimer::release_dmp_timing_scratch_for_power()
          power stage
            gt::GPUTimer::report_power_total_cuda()
          power_total_summary stage
            summarize_xplace_power_total(tensors)
          power_group_codes stage
            gt::GPUTimer::report_power_group_codes()
          power_group_summary stage
            summarize_xplace_power_groups(gpdb, tensors, group_codes)
          optional write_power_csv stage
            write_xplace_power_csv(gpdb, tensors, group_codes)
          writes summaries/<split>_<design>.xplace.json
    flatten_row(...)
      compares OpenROAD report_power rows against Xplace summary
    write_outputs(...)
      writes summary.csv, summary.json, SUMMARY.md
```

这张图是审查入口。`load_design(...)` 到 `preprocess_timing()` 的细节不要塞在本文件，见 `01_DATA_LOADER.md`。

## 2. 模块职责

- `tools/compare_ispd25_route_power_timing.py`: 验收驱动。负责 OpenROAD reference、Xplace worker 子进程、阶段计时、summary/CSV 输出。
- `timer_only/read_platform.py` + `timer_only/database.py` + `timer_only/io_parser.py`: data loader。负责 DEF/LEF/LIB/rawdb/gpdb/tensor 化和 timing-only preprocess。
- `timer_only/timing_opt.py::GPUTimer`: Python wrapper。负责把 `PlaceData` tensor 转成 `TimingTorchRawDB`，再创建 C++ `gt::GPUTimer`。
- `cpp_to_py/gputimer/PyBindCppMain.cpp`: pybind 边界。暴露 `create_timing_rawdb`、`create_gputimer` 和 C++ timer methods。
- `gt::GPUTimer`: timing/RC/power 的 C++ 主对象。
- `cpp_to_py/gputimer/core/power/report`: power report API 和 group classification。
- `cpp_to_py/gputimer/core/power/cuda_input`: 把 `GTDatabase`、Liberty、SDC、timing labels 转成 CUDA power model。
- `cpp_to_py/gputimer/core/power/cuda_activity`: CUDA activity propagation 和 component power kernels。

## 3. Data Loader 边界

本文件只保留 data loader 在总链路里的位置：

```text
run_xplace_worker(args)
  run_timer.getArgs()
  load_design(...)
  data.to_timing_device(...).preprocess_timing()
```

这段的详细端口和最底层 C++ 调用见 `01_DATA_LOADER.md`。本文件从 `timer_only.timing_opt.GPUTimer(...)` 开始继续描述 GPUTimer/power 主干。

## 4. Python Wrapper 到 C++ Timer

```text
timer_only.timing_opt.GPUTimer(data, rawdb, gpdb, params, args)
  输入:
    data: `PlaceData`，已经完成 data loader 和 timing preprocess
    rawdb: `std::shared_ptr<db::Database>`
    gpdb: `std::shared_ptr<gp::GPDatabase>`
    params: 包含 sdc、route_segments/gr_rc、num_threads 等
  关键计算:
    node_lpos = data.node_pos - data.node_size / 2
    pin_rel_lpos = data.pin_rel_lpos + data.pin_size / 2
    conn_node_lpos = concat(movable node_lpos, fixed-connected node_lpos)
    num_movable_nodes = movable_index[1] - movable_index[0]
    scale_factor = 1.0 / data.site_width
  下一跳:
    gputimer.create_timing_rawdb(...)
    gputimer.create_gputimer(params, rawdb, gpdb, timing_raw_db)
    self.timer.init()
    self.timer.levelize()
```

`create_timing_rawdb(...)` 端口：

```text
create_timing_rawdb(
  conn_node_lpos,
  node_size,
  pin_rel_lpos,
  pin_id2node_id,
  pin_id2net_id.int(),
  node2pin_list,
  node2pin_list_end,
  hyperedge_list.int(),
  hyperedge_list_end.int(),
  net_mask,
  num_movable_nodes,
  scale_factor,
  microns,
  wire_resistance_per_micron,
  wire_capacitance_per_micron)
  -> std::shared_ptr<gt::TimingTorchRawDB>
```

`create_gputimer(...)` 端口：

```text
create_gputimer(kwargs, rawdb, gpdb, timing_raw_db)
  文件: cpp_to_py/gputimer/PyBindCppMain.cpp
  做什么:
    - 检查 Liberty 是否已读入 rawdb
    - 根据 kwargs["num_threads"] 设置 timing_raw_db->num_threads
    - 创建 GTDatabase(rawdb, gpdb, timing_raw_db)
    - direct_rc_mode 下跳过 legacy RC tensors
    - SDC::read(kwargs["sdc"])
    - GTDatabase::preparePinNameMapForSdc(*sdc)
    - GTDatabase::ExtractTimingGraph()
    - GTDatabase::readSdc(*sdc)
    - 创建 gt::GPUTimer(gtdb, timing_raw_db)
  返回: std::shared_ptr<gt::GPUTimer>
```

## 5. Timing/RC 前置主干

Power report 依赖 timing graph、RC 和 timing labels。

```text
gt::GPUTimer::initialize()
  初始化 C++/GPU timing 数据结构。

gt::GPUTimer::levelize()
  生成 timing level list，power 可复用或生成 power level。

gt::GPUTimer::update_states()
  同步当前位置/状态到 timing 模型。

gt::GPUTimer::init_dmp_rc_route_segments(route_segments)
  解析 OpenROAD route-segment RC，构建 DMP RC graph/model。
  逐层 C++/CUDA 端口见 `03_BUILD_ROUTING_RC.md`。

gt::GPUTimer::update_timing_dmp()
  更新 arrival/required/slew/load/delay 等 timing arrays。

gt::GPUTimer::release_dmp_timing_scratch_for_power()
  power 前释放非必要 DMP timing scratch，降低峰值显存。
```

Power CUDA 主要消费：`pinLoad`、`pinSlew`、DMP `C1/C2`、SDC clock 信息、`pin_is_clk`、`net_is_clock`。

## 6. Power Report 分发表

所有 CUDA component report 基本都进入 `compute_power_activity_cuda(...)`。非空指针参数就是请求开关。

```text
report_power_activity_cuda()
  -> compute_power_activity_cuda(nullptr, nullptr)
  -> CPU tensor [num_pins, 3] = density, duty, origin

report_power_switching_cuda()
  -> compute_power_activity_cuda(&inst_switching_cpu, &pin_switching_cpu)
  -> (inst_switching [num_nodes], pin_switching [num_pins])

report_power_internal_cuda()
  -> compute_power_activity_cuda(nullptr, nullptr, &inst_internal_cpu)
  -> inst_internal [num_nodes]

report_power_internal_arcs_cuda()
  -> compute_power_activity_cuda(nullptr, nullptr,
                                 &inst_internal_cpu,
                                 &internal_row_power_cpu,
                                 &internal_row_meta_cpu)
  -> (inst_internal, internal_row_power, internal_row_meta)

report_power_leakage_cuda()
  -> compute_power_activity_cuda(nullptr, nullptr, nullptr, nullptr, nullptr,
                                 &inst_leakage_cpu)
  -> inst_leakage [num_nodes]

report_power_leakage_rows_cuda()
  -> compute_power_activity_cuda(nullptr, nullptr, nullptr, nullptr, nullptr,
                                 &inst_leakage_cpu,
                                 &leakage_row_power_cpu,
                                 &leakage_row_meta_cpu)
  -> (inst_leakage, leakage_row_power, leakage_row_meta)

report_power_total_cuda()
  -> compute_power_activity_cuda(&inst_switching_gpu, nullptr,
                                 &inst_internal_gpu, nullptr, nullptr,
                                 &inst_leakage_gpu, nullptr, nullptr,
                                 true)
  -> inst_total_gpu = internal + switching + leakage
  -> (inst_internal_gpu, inst_switching_gpu, inst_leakage_gpu, inst_total_gpu)

report_power_group_codes()
  不调用 compute_power_activity_cuda。
  CPU 分类并返回 int64 [num_nodes]:
    0=sequential, 1=combinational, 2=clock, 3=macro, 4=pad
```

## 7. `compute_power_activity_cuda` 主干

```text
torch::Tensor GPUTimer::compute_power_activity_cuda(
    torch::Tensor* inst_switching_cpu,
    torch::Tensor* pin_switching_cpu,
    torch::Tensor* inst_internal_cpu = nullptr,
    torch::Tensor* internal_row_power_cpu = nullptr,
    torch::Tensor* internal_row_meta_cpu = nullptr,
    torch::Tensor* inst_leakage_cpu = nullptr,
    torch::Tensor* leakage_row_power_cpu = nullptr,
    torch::Tensor* leakage_row_meta_cpu = nullptr,
    bool output_power_tensors_cuda = false)
```

稳定管线：

```text
compute_power_activity_cuda(...)
  1. 读取 n=num_pins、num_nodes、time unit、voltage/cap unit、min clock period
  2. 根据输出指针推导 need_switching/internal/leakage/activity
  3. buildPowerPinNodeNetMaps
  4. classifyPowerPins
  5. buildPowerNetDriverPins
  6. buildPowerClockGateMaps
  7. buildPowerClockPinActivity
  8. buildPowerCudaExprInputs
  9. buildPowerCudaSeqInputs
 10. buildPowerClockSlews
 11. buildPowerCudaRootInputs
 12. buildPowerCudaInternalRows
 13. buildPowerCudaLeakageRows
 14. writePowerRowMetaOutputs
 15. buildPowerNodePortPinMap
 16. PowerCudaUploader 上传 host vectors 到 CUDA tensors
 17. buildPowerCudaArcSkipInputs
 18. levelize_power(...)
 19. finalizePowerCudaRootInputs(...)
 20. choosePowerActivityLevels(...)
 21. preparePowerCudaRunBuffers(...)
 22. choosePowerDmpLoadPointers(...)
 23. 组装 PowerGraphDeviceView / PowerExprDeviceView / PowerActivityState /
     PowerActivityConfig / PowerComponentDeviceView / PowerActivityCudaModel
 24. run_power_activity_cuda_launcher(activity_model)
 25. runPowerChunkedComponents(...)
 26. finishPowerActivityOutputs(...)
```

## 8. CUDA 主干

```text
run_power_activity_cuda_launcher(const PowerActivityCudaModel& model)
  文件: cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu
  做什么:
    - 解包 model
    - 分配 density/duty/origin/active/seq scratch
    - cudaMemcpy model/scratch view 到 device
    - seed roots: case/PI/clock/seq feedback
    - 按 level 或 level queue 传播 activity
    - 多 pass 处理 sequential feedback
    - pack activity output
    - inline 或 chunked 计算 switching/internal/leakage
    - 释放 scratch 并检查 CUDA error
```

最底层 kernel 主类：

```text
Activity seed:
  power_seed_case_kernel
  power_seed_pi_kernel
  power_seed_clock_active_kernel
  power_seed_seq_feedback_state_kernel

Activity propagation:
  power_visit_level_kernel
  power_visit_level_serial_kernel
  power_visit_active_list_serial_kernel
  power_activity_level_queue_persistent_kernel
  power_activity_level_queue_ordered_kernel

Sequential feedback:
  power_mark_pending_seq_changes_kernel
  power_seed_seq_kernel
  power_seed_seq_ordered_kernel
  power_seed_seq_id_list_ordered_kernel

Output/copy:
  power_pack_output_kernel
  power_copy_precomputed_activity_output_kernel
  power_unpack_precomputed_activity_kernel
  power_unpack_activity_density_duty_kernel

Component power:
  power_switching_kernel
  power_internal_denom_fast_kernel
  power_internal_denom_kernel
  power_internal_contrib_fast_kernel
  power_internal_contrib_kernel
  power_leakage_row_fast_kernel
  power_leakage_row_kernel
  power_leakage_summary_kernel
```

## 9. 核心 Struct 合约

- `GpuPowerExprOpHost`: Liberty function/when expression 的单个 op，device expression eval 使用。
- `GpuPowerSeqHost`: sequential data/clock/Q/QN activity 状态。
- `GpuPowerInternalHost`: 一个 Liberty `internal_power` row 的 GPU 输入。
- `GpuPowerLeakageRowHost`: 一个 Liberty `leakage_power` 条件行。
- `GpuPowerLeakageGroupHost`: 每个 node 的 leakage group 汇总基础。
- `PowerCudaExprInputs`: host expression table，由 expr/seq/internal/leakage 构建阶段共同维护。
- `PowerCudaSeqInputs`: host sequential table。
- `PowerCudaRootInputs`: root seed、loop、feedback 控制。
- `PowerCudaArcSkipInputs`: power propagation arc skip table。
- `PowerCudaRunBuffers`: 本次 run 的 output tensors 和 raw pointers。
- `PowerGraphDeviceView`: graph/timing/load/clock/level device view。
- `PowerExprDeviceView`: expression device view。
- `PowerActivityState`: seed 和 sequential state。
- `PowerActivityConfig`: runtime knobs。
- `PowerComponentDeviceView`: switching/internal/leakage row table 和 output pointers。
- `PowerActivityCudaModel`: launcher 和 kernels 的总模型。
- `PowerActivityScratchView`: launcher 内部可变 scratch。

## 10. 不变量

- `pin_id` 必须全链路一致：gpdb pin id、gtdb pin id、timing arc pin id、power expression physical pin id 是同一编号空间。
- `node_id` 必须对应 gpdb node 顺序；所有 `[num_nodes]` power tensor 直接用 node_id 索引。
- `net_id` 必须对应 gpdb net 顺序；`pin_to_net`、`net_driver_pin`、flat net2pin map 不能混用编号。
- `TimingTorchRawDB` 接收的坐标必须和 `PlaceData.preprocess_timing()` 后的 scale/shift 语义一致。
- Direct RC route-segment 模式不能回退到 legacy SPEF/flute RC 语义。
- `report_power_total_cuda()` 返回顺序固定为 internal、switching、leakage、total。
- `report_power_group_codes()` 编码固定为 0 sequential、1 combinational、2 clock、3 macro、4 pad。
- `.cpp` 文件不放 CUDA runtime/kernels；CUDA launch wrapper 必须在 `.cu`。

## 11. 新 Feature 放置位置

- data loader 入口和端口文档：`01_DATA_LOADER.md`。
- Python benchmark/验收：`tools/compare_ispd25_route_power_timing.py`。
- Python wrapper 到 C++ timer：`timer_only/timing_opt.py`。
- pybind：`cpp_to_py/gputimer/PyBindCppMain.cpp`。
- C++ power report：`cpp_to_py/gputimer/core/power/report/PowerReport.cpp`。
- C++ power groups：`cpp_to_py/gputimer/core/power/report/PowerGroups.cpp`。
- Host CUDA input：`cpp_to_py/gputimer/core/power/cuda_input/`。
- CUDA model/view：`cpp_to_py/gputimer/core/power/common/PowerCudaModel.h`。
- CUDA activity launcher/kernels：`cpp_to_py/gputimer/core/power/cuda_activity/`。

## 12. 常用验收命令

文档修改不需要 build。改 C++/CUDA 后再跑：

```bash
cd /research/d7/ascstd/qkduan25/Xplace/build
conda activate gnn
make -j8
make install
cd /research/d7/ascstd/qkduan25/Xplace
python tools/compare_ispd25_route_power_timing.py --split visible --design ariane --reuse-openroad
```
