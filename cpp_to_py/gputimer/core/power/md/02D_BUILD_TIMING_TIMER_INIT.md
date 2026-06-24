# Build Timing Graph: Timer Init 和 Levelize

Last reviewed: 2026-06-23

拆自 `02_BUILD_TIMING_GRAPH.md`。本文覆盖 `gt::GPUTimer` constructor、CUDA `initialize()`、`levelize()`、downstream consumers、审查清单和调试开关。

## 16. gt::GPUTimer 构造函数

文件：`cpp_to_py/gputimer/core/GPUTimer.cpp:14`。

这个 constructor 不做复杂计算，核心是从 `TimingTorchRawDB` 的 Torch tensors 取裸指针：

```text
x / y / init_x / init_y
node_size_x / node_size_y
pin_offset_x / pin_offset_y

pinSlew / pinLoad / pinRAT / pinAT
pinImpulse / pinRootDelay  # nullable
arcDelay

at_prefix_pin / at_prefix_arc / at_prefix_attr

pin_forward_arc_list / pin_forward_arc_list_end
pin_backward_arc_list / pin_backward_arc_list_end
timing_arc_from_pin_id / timing_arc_to_pin_id
pin_num_fanin
pin_fanout_list / pin_fanout_list_end

timing_arc_id_map
arc_types
arc_id2test_id
test_id2_arc_id
test_id2_endpoint_id
primary_output2_endpoint_id

flat_node2pin_start_map / flat_node2pin_map / pin2node_map
flat_net2pin_start_map / flat_net2pin_map / pin2net_map
net_mask

dmp_* threshold/library arrays
```

并记录：

```text
num_arcs = gtdb.num_arcs
num_timings = gtdb.num_timings
num_tests = gtdb.num_tests
num_POs = gtdb.num_POs
total_num_fanouts = gtdb.total_num_fanouts
res_unit / cap_unit / clock_period
```

生命周期保护：

```text
gtdb_holder = gtdb_
timing_raw_db_holder = timing_raw_db_
```

审查点：

- constructor 发生在 `readSdc()` 后；否则 `pinSlew/pinLoad/pinRAT/pinAT` 未定义会直接 `data_ptr` 崩。
- constructor 只缓存 raw pointers；后续如果替换 Torch tensor 对象，旧 pointer 会悬空。当前流程中 graph/state tensors 不应在 constructor 后重新分配。

## 17. gt::GPUTimer::initialize

文件：`cpp_to_py/gputimer/core/GPUTimer.cu:341`。

这是第一段真正 CUDA runtime 初始化。

CUDA allocation：

```text
cudaMalloc(pinCap, num_pins * (NUM_ATTR + 2))
if !skip_legacy_rc_tensors:
  cudaMalloc(pinWireCap, num_pins * NUM_ATTR)
  cudaMalloc(pinRootRes, num_pins * NUM_ATTR)

cudaMalloc(testRelatedAT, num_tests * NUM_ATTR)
cudaMalloc(testRAT, num_tests * NUM_ATTR)
cudaMalloc(testConstraint, num_tests * NUM_ATTR)

cudaMalloc(net_is_clock, num_nets)
cudaMalloc(pin_is_clk, num_pins)
cudaMalloc(pin_is_ideal_clk, num_pins)
cudaMalloc(level_list, num_pins)
cudaMalloc(primary_outputs, num_POs)
```

host-to-device copies：

```text
cudaMemcpy(pinCap, gtdb.pin_capacitance)
cudaMemset(net_is_clock/pin_is_clk/pin_is_ideal_clk, 0)
cudaMemcpy(net_is_clock, gtdb.net_is_clock)       # if present
cudaMemcpy(pin_is_clk, gtdb.pin_is_clk)           # if present
cudaMemcpy(pin_is_ideal_clk, gtdb.pin_is_ideal_clk)
cudaMemcpy(primary_outputs, gtdb.primary_outputs)
```

Liberty timing LUT upload：

```text
allocator = new GPULutAllocator()
allocator->AllocateBatch(gtdb.liberty_timing_arcs)
allocator->CopyToGPU()
cudaMalloc(d_allocator)
cudaMemcpy(d_allocator, allocator)
allocator->CopyToGPU(d_allocator)
```

底层函数：

- `cpp_to_py/gputimer/core/gputiming_host.cu::GPULutAllocator::AllocateBatch`
  - flatten 每个 `TimingArc` 的 delay/slew/constraint LUT。
  - 记录 timing sense、rising/falling edge trigger、constraint flag、latch clock flag。
- `cpp_to_py/gputimer/core/gputiming_host.cu::GPULutAllocator::CopyToGPU`
  - `cudaMalloc` LUT arrays。
  - `cudaMemcpy` x/y/table/offset/flags。
  - 生成 `d_allocated_bits` bitset。
- `cpp_to_py/gputimer/core/gputiming.h::GPULutAllocator::query/lut`
  - 后续 propagation kernel 的 device-side lookup/interpolation。

Power internal LUT upload：

```text
power_allocator = new GPUPowerLutAllocator()
power_allocator->AllocateBatch(gtdb.liberty_internal_powers)
power_allocator->CopyToGPU()
cudaMalloc(d_power_allocator)
cudaMemcpy(d_power_allocator, power_allocator)
power_allocator->CopyToGPU(d_power_allocator)
```

状态备份：

```text
if !GPUTIMER_DISABLE_STATE_BACKUP_TENSORS:
  cudaMalloc(__pinSlew__/__pinLoad__/__pinRAT__/__pinAT__)
  device_copy_batch<<<...>>>(pinSlew, __pinSlew__)
  device_copy_batch<<<...>>>(pinLoad, __pinLoad__)
  device_copy_batch<<<...>>>(pinRAT, __pinRAT__)
  device_copy_batch<<<...>>>(pinAT, __pinAT__)
```

底层 kernel：

```text
cpp_to_py/gputimer/core/utils.cuh::device_copy_batch<T>
```

审查点：

- `GPUTimer::initialize()` 的 `CUDA_CHECK` 只 print，不 throw；`levelize.cu` 的 CUDA check 会 throw。错误处理语义不一致。
- 如果 `SetDrivingCell` 在 `readSdc()` 中追加了 `gtdb.liberty_timing_arcs`，`GPULutAllocator` 会看到追加后的 list；但 `timing_raw_db.dmp_timing_library_ids` 已在 `ExtractTimingGraph()` 里 materialize，可能没有追加 timing id 的 DMP library id。需要核对 DMP driving-cell source 对 `dmp_timing_library_ids[timing_id]` 的访问是否可能越界或语义缺失。
- `pin_is_clk/pin_is_ideal_clk` 先由 `readSdc()` 填 host vector，再在 `initialize()` 复制到独立 CUDA malloc 区；这与 `timing_raw_db.pin_clock_*` tensors 是两套 storage。

## 18. gt::GPUTimer::levelize

文件：`cpp_to_py/gputimer/core/levelize.cu:223`。

使用 GPU 做 Kahn-style topo levelization。

host 准备：

```text
num_frontiers = gtdb.pin_frontiers.size()
cudaMalloc(frontiers, num_pins)
cudaMalloc(next_frontiers, num_pins)
cudaMalloc(next_num_frontiers, 1)
cudaMalloc(last_idx, 1)
cudaMemcpy(frontiers, gtdb.pin_frontiers)

TimingLevelizeModel {
  frontiers
  next_frontiers
  level_list
  pin_fanout_list_end
  pin_fanout_list
  pin_num_fanin
  next_num_frontiers
  last_idx
}
cudaMalloc(d_level_model)
cudaMemcpy(d_level_model, &level_model)
```

主循环：

```text
level_list_end_cpu = [0]
total_num_frontiers = 0

while num_frontiers > 0:
  total_num_frontiers += num_frontiers
  level_list_end_cpu.push_back(total_num_frontiers)

  advanceLevel<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(d_level_model, num_frontiers)
  CUDA_CHECK("advanceLevel")

  cudaMemcpy(&num_frontiers, next_num_frontiers, D2H)
  device_copy<<<1, 1>>>(next_frontiers, frontiers, num_frontiers)
  cudaMemset(next_num_frontiers, 0)
```

底层 kernel：`advanceLevel`，位于 `levelize.cu:57`。

每个 frontier pin：

```text
pin_id = frontiers[idx]
ptr = atomicAdd(last_idx, 1)
level_list[ptr] = pin_id

for i in pin_fanout_list_end[pin_id] .. pin_fanout_list_end[pin_id+1]:
  fo_pin_id = pin_fanout_list[i]
  prev_num = atomicAdd(&pin_num_fanin[fo_pin_id], -1)
  if prev_num == 1:
    end = atomicAdd(next_num_frontiers, 1)
    next_frontiers[end] = fo_pin_id
```

循环后：

```text
cudaFree(d_level_model)
cudaMalloc(level_list_end)
cudaMemcpy(level_list_end, level_list_end_cpu)
cudaMemcpy(level_list_host, level_list)
build pin_level_cpu on host
```

审查点：

- `pin_num_fanin` 指向 `timing_raw_db.pin_num_fanin` 的 GPU tensor，会被 `advanceLevel` 原地递减到 0。后续如果还需要原始 fanin，必须从 `gtdb.pin_num_fanin` 或重新 materialize。
- `frontiers`、`next_frontiers`、`next_num_frontiers`、`last_idx` 当前在 `levelize()` 结束没有 `cudaFree`；只有 `d_level_model` 被释放。`levelize_power()` 末尾有完整 free，对比可确认这是一个应审查的临时 GPU buffer 泄漏点。
- kernel launch grid 使用 `BLOCK_NUMBER(num_pins)`，kernel 内部 guard `idx < num_frontiers`；这不是错误，但 frontiers 很少时会有额外空线程。
- 如果 graph 有 cycle 或 fanin 计数错误，`while` 会提前停，`total_num_frontiers < num_pins`；当前 `checkTimingGraph(...)` 调用被注释掉。

## 19. levelize_power 不是本入口，但会复用 graph

文件：`cpp_to_py/gputimer/core/levelize.cu:284`。

`levelize_power(...)` 不在本文入口片段中，它由 `cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp` 后续调用。

它使用不同模型：

```text
PowerLevelizeModel
countPowerFanin<<<...>>>
seedPowerFrontiers<<<...>>>
advancePowerLevel<<<...>>>
appendUnemittedPowerPins<<<...>>>
```

差异：

- 会根据 sequential output pin、arc skip、load pin、net driver 等 power activity 规则过滤边。
- 对未 emit 的 pins 会 append 到末尾，避免 power level list 缺 pin。
- 末尾会释放所有临时 CUDA buffers。

审查意义：

- timing `levelize()` 和 power `levelize_power()` 消费同一批 graph arrays，但 topo 语义不同。
- 如果改 `pin_forward_arc_list`、`arc_types`、`timing_arc_to_pin_id`，必须同时检查 timing propagation 和 power activity levelization。

## 20. 下游第一批 consumer

虽然不在入口片段里，审查 graph correctness 时要知道谁消费这些 arrays。

普通 timing propagation：

```text
cpp_to_py/gputimer/core/propagate.cpp::GPUTimer::update_timing()
  fills TimingPropagationModel
  update_timing_cuda(model)

cpp_to_py/gputimer/core/propagate.cu
  propagatePin<<<...>>>       # forward AT/slew/delay/test
  propagatePinBack<<<...>>>   # backward RAT
```

`TimingPropagationModel` 直接消费：

```text
level_list / level_list_end_cpu
pin_forward_arc_list(_end)
pin_backward_arc_list(_end)
timing_arc_from_pin_id / timing_arc_to_pin_id
arc_types
arc_id2test_id
timing_arc_id_map
pinSlew / pinLoad / pinAT / pinRAT
arcDelay
testRelatedAT / testRAT / testConstraint
d_allocator
```

DMP timing / route-segment path：

```text
GPUTimer::update_timing_dmp()
  apply_dmp_driving_cell_source_slew(...)
  update_timing_dmp_cuda(this)
```

Power CUDA input：

```text
PowerCudaInputBuild.cpp
  reads gtdb.liberty_* vectors
  reads graph/timing arrays
  calls levelize_power(...)
```

## 21. 人工审查清单

建议按这个顺序看代码：

1. `src/core/timing_opt.py:100-126`
   - Python 传入 tensor shape、device、lower-left/center 坐标语义。
2. `cpp_to_py/gputimer/PyBindCppMain.cpp:47-84`
   - `num_threads`、direct RC mode、SDC/graph/readSdc 顺序。
3. `cpp_to_py/gputimer/db/GTDatabase.cpp:1087`
   - `TimingTorchRawDB` flat maps 和 connected-node layout。
4. `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:35-66`
   - SDC target pin-name map 是否覆盖当前 SDC 命令。
5. `cpp_to_py/gputimer/db/GTDatabase.cpp:255-569` and `cpp_to_py/gputimer/db/GTDatabase.cpp:948-1085`
   - graph IDs 是否稳定，net arcs/cell arcs 是否覆盖完整。
6. `cpp_to_py/gputimer/db/sdc/*.cpp`
   - `_read_sdc` 是否可能改变 `liberty_timing_arcs`、pin states、clock states。
7. `cpp_to_py/gputimer/core/GPUTimer.cpp:14-120`
   - constructor pointer binding 是否在所有 tensors 定义后发生。
8. `cpp_to_py/gputimer/core/GPUTimer.cu:341-421`
   - `cudaMalloc/cudaMemcpy`、LUT upload、backup tensor 开关。
9. `cpp_to_py/gputimer/core/levelize.cu:57-72` 和 `223-281`
   - topo kernel 是否覆盖所有 pins，临时 GPU buffer 是否释放。
10. `cpp_to_py/gputimer/core/propagate.cpp`、`propagate.cu`、`DmpTiming.cu`、`power/cuda_input`
    - graph arrays 的真实 consumer。

## 22. 高风险点汇总

- `levelize()` 临时 CUDA buffers 未释放：`frontiers`、`next_frontiers`、`next_num_frontiers`、`last_idx`。
- `advanceLevel` 原地修改 `pin_num_fanin` GPU tensor；不要在 levelize 后把它当原始 fanin。
- `SetDrivingCell` 可能追加 `liberty_timing_arcs`，要核对追加 timing id 和 `dmp_timing_library_ids`、DMP gate lookup 的一致性。
- `GTDatabase::ExtractTimingGraph()` 是 CPU/OpenMP 建图；如果要调 CUDA 错，先确认错误是否发生在 tensor materialization、`initialize()` 还是 `levelize()`。
- graph arc order 是语义：net arcs first，cell arcs second；pin arc list bucket 内 arc 顺序当前不作为 timing/power 语义。
- net driver 来自 `gpdb.getNets()[net_id].pins()[0]`；pin order 错会直接影响 timing graph。
- `pin_name2pin_id` 默认是 lazy target map；新增 SDC pin-name lookup 要补 `preparePinNameMapForSdc()`。
- constructor 缓存 Torch tensor raw pointers；constructor 后不能重分配这些 tensor。

## 23. 调试开关

```text
XPLACE_TIMER_PROFILE=1
  打印 Python/C++ phase 和 ExtractTimingGraph subphase。

XPLACE_TIMER_GRAPH_THREADS=<N>
  覆盖 graph extraction OpenMP 线程数。

GPUTIMER_MEM_PROFILE=1
  打印 CUDA memory info。

GPUTIMER_EMPTY_CACHE_AFTER_GTDB=1
  ExtractTimingGraph end 后清 PyTorch CUDA cache，主要用于内存诊断。

GPUTIMER_BUILD_FULL_PIN_NAME_MAP=1
  强制完整 pin_name2pin_id map，主要用于 SDC name debug。

GPUTIMER_DISABLE_REF_TIMING_TENSORS=1
  跳过 legacy reference/ratio timing tensors。

GPUTIMER_DISABLE_STATE_BACKUP_TENSORS=1
  跳过 __pinSlew__/__pinLoad__/__pinRAT__/__pinAT__ 备份。

DMP_PROFILE_LUTS=1
  initialize() 中统计 DMP gate LUT 使用情况。
```
