# 04_TIMING_ANALYSYS.md

Last reviewed: 2026-06-10

本文展开 `00_POWER_ARCHITECTURE.md` 里的 `timer stage` 和后面的可选 release stage：

```text
timer stage
  gt::GPUTimer::update_timing_dmp()
  timer_only.GPUTimer.report_timing_slack()
    gt::GPUTimer::update_endpoints()
    gt::GPUTimer::report_wns_and_tns()

optional release stage
  gt::GPUTimer::release_dmp_timing_scratch_for_power()
```

目标是给代码审查用：从当前 Python 入口一直追到最底层 C++/CUDA 函数，标出每一层读什么、写什么、哪里同步、哪里用 atomic、哪里需要重点核查。

## 1. 范围和前置状态

本文只覆盖 direct OpenROAD `--route_segments` 路径下已经完成 `build_rc stage` 之后的 timing propagation 和 slack report。

进入本文时默认已经完成：

```text
gt::GPUTimer::update_states()
gt::GPUTimer::init_dmp_rc_route_segments(route_segments)
```

也就是说 DMP RC 相关输入已经存在：

```text
DmpModel::C1 / C2 / r_pi
DmpModel::elmore_delay
timing_raw_db.pinLoad
level_list / level_list_end_cpu
pin_forward_arc_list(_end)
pin_backward_arc_list(_end)
timing_arc_from_pin_id / timing_arc_to_pin_id
arc_types
arc_id2test_id
timing_arc_id_map
d_allocator Liberty LUT
sparse SDC clock id/table arrays
```

`NUM_ATTR = 4`，约定是：

```text
0 early-rise
1 early-fall
2 late-rise
3 late-fall
```

`arcDelay` 是 `[num_arcs, 2 * NUM_ATTR]`。net arc 只用同相 transition 槽位 `{0, 3, 4, 7}`；gate arc 用完整 `el * 4 + input_rf * 2 + output_rf`。

## 2. 当前入口

当前 power/timing 对比脚本把 RC 构建和 timing propagation 拆成两个 stage 计时：

```text
tools/compare_ispd25_route_power_timing.py::run_xplace_worker(...)
  build_rc()
    gputimer.timer.update_states()
    gputimer.timer.init_dmp_rc_route_segments(timer_args.route_segments)

  run_timer()
    gputimer.timer.update_timing_dmp()
    return gputimer.report_timing_slack()

  time_stage(stages, "timer", run_timer, torch)
```

`time_stage(...)` 在函数前后都会 `torch.cuda.synchronize()`，所以 stage 时间包含 `update_timing_dmp()`、`update_endpoints()`、`report_wns_and_tns()` 里尚未完成的 CUDA work。

普通 `run_timer.py --route_segments ...` 入口没有拆 stage：

```text
run_timer.py::main()
  gputimer.update_timing_dmp_route_segments(args.route_segments)
    timer_only/timing_opt.py::GPUTimer.update_timing_dmp_route_segments(...)
      self.timer.update_states()
      self.timer.init_dmp_rc_route_segments(route_segments_file)
      self.timer.update_timing_dmp()

  gputimer.report_timing_slack()
```

代码审查时要先确认自己看的是哪条入口：compare worker 的 timer stage 不包含 RC 构建；`run_timer.py` wrapper 包含 RC 构建。

## 3. 一眼看完整链路

```text
Python worker
  tools/compare_ispd25_route_power_timing.py::run_timer()
    pybind GPUTimer.update_timing_dmp()
      cpp_to_py/gputimer/core/DmpModel.cpp
        gt::GPUTimer::update_timing_dmp()
          apply_dmp_driving_cell_source_slew(*this)
            cpp_to_py/gputimer/core/DmpGateEval.cu
              apply_dmp_driving_cell_source_slew_cuda(...)
                applyDrivingCellSourceSlewKernel<<<...>>>()
                  DmpModel::computeDrivingCellDriverWave(...)
                    DmpModel::makeGateArcMetaForTiming(...)
                    DmpModel::computeGateDriverWaveForSlot(...)
                      DmpGateArcMeta::capDelaySlew(...)
                      DmpGateArcMeta::estimateRd(...)
                      DmpRcParams::selectAlg(...)
                      DmpModel::computeZeroC2DriverWave(...)
                      DmpModel::computePiDriverWave(...)
                      DmpDriverWave::findDriverDelaySlew(...)
                      DmpDriverWave::findDriverCrossing(...)
                  writes source at_prefix_* and pinSlew

          update_timing_dmp_cuda(this)
            cpp_to_py/gputimer/core/DmpTiming.cu
              dmp_get_forward_schedule(...)
                build_forward_arc_levels(...)
                dmp_upload_forward_schedule(...)
              cudaMemset(pin_at_winner)
              dmpResetForwardTargetsKernel<<<...>>>()

              for each forward level:
                dmpGateKernel<<<...>>>()
                  makeGateArcMetaForTiming / computeGateDriverWaveForSlot
                  updateAtWinner
                  loadDelaySlewFromDriverWave
                  dmpAtomicSelectFloatKey
                dmpDirectNetKernel<<<...>>>()
                  DmpModel::propagateLoadSlewDelay(...)
                dmpNetWinnerKernel<<<...>>>()
                  decode selected net delay
                  updateAtWinner
                dmpPinWinnerKernel<<<...>>>()
                  decode selected pin AT and slew
                  write pinAT / pinSlew / at_prefix_*
                dmpTestKernel<<<...>>>()
                  DmpModel::propagatePinTests(...)
                    DmpModel::propagateTest(...)
                      GPULutAllocator::query(..., constraint)
                      write testRelatedAT / testConstraint / testRAT / pinRAT

              for each backward level:
                dmpBackwardKernel<<<...>>>()
                  DmpModel::propagatePinBack(...)
                    DmpModel::propagateRAT(...)
                    DmpModel::updatePinRat(...)

    timer_only.GPUTimer.report_timing_slack()
      gt::GPUTimer::time_unit()
      gt::GPUTimer::update_endpoints()
        update_endpoints_kernel0<<<...>>>()
        update_endpoints_kernel1<<<...>>>()
        update_endpoint_pin_slacks_kernel0<<<...>>>()
        update_endpoint_pin_slacks_kernel1<<<...>>>()
      gt::GPUTimer::report_wns_and_tns()
        endpoint_pin_slacks_for_report(...)
          report_pin_slack()
        torch min / clamp / sum
      convert seconds to ns in Python

optional release
  gt::GPUTimer::release_dmp_timing_scratch_for_power()
    dmp_release_after_timing(h_dmp_db, dmp_db)
      DmpModel::release_after_timing()
```

## 4. `update_timing_dmp()` C++ 入口

文件：

```text
cpp_to_py/gputimer/core/DmpModel.cpp
cpp_to_py/gputimer/PyBindCppMain.cpp
```

pybind 只暴露 thin method：

```text
.def("update_timing_dmp", &gt::GPUTimer::update_timing_dmp)
```

C++ member 本身也很薄：

```text
gt::GPUTimer::update_timing_dmp()
  optional progress printf
  apply_dmp_driving_cell_source_slew(*this)
  update_timing_dmp_cuda(this)
```

主要审查点：

- `apply_dmp_driving_cell_source_slew(...)` 只在 `gtdb.driving_cell_sources` 非空时做事。
- 它把每个 source 的 `pin_id`、每个 attr 的 `timing_id`、`input_rf`、`input_slew` 打平成 host vectors。
- 真正 CUDA 入口是 `apply_dmp_driving_cell_source_slew_cuda(...)`。
- `update_timing_dmp_cuda(this)` 是 timing propagation 主体。

## 5. `set_driving_cell` source slew 预处理

文件：

```text
cpp_to_py/gputimer/core/DmpModel.cpp
cpp_to_py/gputimer/core/DmpGateEval.cu
cpp_to_py/gputimer/core/DmpGateModel.cuh
```

调用链：

```text
apply_dmp_driving_cell_source_slew(GPUTimer& timer)
  collect pin_ids / timing_ids / input_rfs / input_slews
  apply_dmp_driving_cell_source_slew_cuda(dmp_db, ...)
    cudaMalloc / cudaMemcpy temporary input arrays
    optional d_counts
    applyDrivingCellSourceSlewKernel<<<DMP_TIMING_BLOCK_NUMBER(total), DMP_TIMING_BLOCK_SIZE>>>()
    cudaPeekAtLastError()
    optional cudaEvent profile / counts copyback
    cudaFree temporary arrays
```

`applyDrivingCellSourceSlewKernel` 每个 thread 处理一个 `(source, attr)`：

```text
applyDrivingCellSourceSlewKernel(...)
  validate pin_id / timing_id / input_rf / input_slew
  pin_slot = pin_id * NUM_ATTR + attr
  DmpModel::computeDrivingCellDriverWave(...)
    DmpModel::makeGateArcMetaForTiming(...)
      timingLibraryId(...)
      driverLibraryThresholds(...)
      GPULutAllocator::makeGateLutMeta(delay)
      GPULutAllocator::makeGateLutMeta(slew)
      DmpGateArcMeta::hasValidLuts()
    DmpModel::computeGateDriverWaveForSlot(...)
      DmpGateArcMeta::capDelaySlew(C1 + C2)
      DmpGateArcMeta::estimateRd(...)
      DmpRcParams::selectAlg(...)
      DmpModel::computeZeroC2DriverWave(...) or computePiDriverWave(...)
      fallback DMP_ALG_CAP if PI/zero-C2 cannot model
  intrinsic_delay = d_allocator->query(..., load=0, delay)
  write at_prefix_pin = -1
  write at_prefix_arc = packed driving-cell tag
  write at_prefix_attr = DMP_DRIVING_CELL_PREFIX_ATTR
  write pinSlew[pin_slot] = input_slew
```

这一步不直接写 `pinAT`。它给 source pin 留下虚拟 driving cell tag，后续 direct net propagation 会用这个 tag 重建 driver waveform。

最底层 device 计算：

```text
GPULutAllocator::makeGateLutMeta(...)
GPULutAllocator::gateLutWithMeta(...)
  dmpGateLowerBound(...)
    dmpLowerBound7(...) or binary search
  interpolate(...)

DmpGateArcMeta::capDelaySlew(...)
DmpGateArcMeta::estimateRd(...)
DmpRcParams::initZeroC2(...)
DmpRcParams::initPi(...)
DmpRcParams::selectAlg(...)
DmpDriverWave::findDriverDelaySlew(...)
  DmpDriverWave::findDriverCrossing(...)
    driverRootFunc(...)
DmpDriverWave::findLoadCrossing(...)
  loadRootFunc(...)
```

## 6. Forward schedule 构建

文件：

```text
cpp_to_py/gputimer/core/DmpTiming.cu
```

`update_timing_dmp_cuda(...)` 首先获得 forward schedule：

```text
dmp_get_forward_schedule(timer)
  if cached schedule shape matches:
    reuse
  else:
    build_forward_arc_levels(timer)
    dmp_upload_forward_schedule(schedule)
```

`build_forward_arc_levels(...)` 是 host 端函数，按 pin level 扫 `pin_backward_arc_list`，拆出三类列表：

```text
gate_arc_list
  arc_type == 1

net_arc_list
  arc_type == 0

direct_net_arc_list
  arc_type == 0 and from pin has no gate driver fanin
```

同时它计算 profile/debug 统计：

```text
max_level_gate_arcs
max_level_net_arcs
max_level_direct_net_arcs
gate_net_pairs
valid_pair_lanes
invalid_pair_lanes
```

`dmp_upload_forward_schedule(...)` 把三个 host vectors 传到 GPU：

```text
dmp_upload_vector(gate_arc_list, &d_forward_gate_arc_list)
dmp_upload_vector(net_arc_list, &d_forward_net_arc_list)
dmp_upload_vector(direct_net_arc_list, &d_forward_direct_net_arc_list)
```

审查重点：

- 这个 schedule 是按 `num_pins`、`num_arcs`、`level_list_size` 缓存复用的。
- 如果 timing graph 不变但 arc 属性或 timing map 变了，现有复用条件不会重建 schedule。
- `release_dmp_forward_schedule_cuda(...)` 负责释放三个 device vectors。

## 7. Forward propagation 主循环

文件：

```text
cpp_to_py/gputimer/core/DmpTiming.cu
cpp_to_py/gputimer/core/DmpGateEval.cu
cpp_to_py/gputimer/core/DmpGateProp.cu
```

入口：

```text
update_timing_dmp_cuda(GPUTimer* timer)
  dmp_clear_stale_cuda_error("DMP timing entry")
  cudaMemset(pin_at_winner, 0)
  dmpResetForwardTargetsKernel<<<...>>>(dmp_db)
  for i = 1 .. level_list_end_cpu.size() - 2:
    launch forward kernels for level i
  cudaDeviceSynchronize()
```

### 7.1 `dmpResetForwardTargetsKernel`

`dmpResetForwardTargetsKernel` 清理 forward propagation 的目标 scratch：

```text
for pin slots:
  preserve source slew for primary input, source clock pin, driving-cell tag
  otherwise reset encoded pinSlew winner storage to 0

for arc delay slots:
  gate arc: arcDelay = NaN
  net arc: reset encoded delay winner storage to 0
```

它不清 `pinAT`、`pinRAT` 的全部内容。AT/RAT 的有效更新依赖后续 winner/test/backward 写回以及前置 `update_states()` 的 baseline label。

### 7.2 `dmpGateKernel`

`dmpGateKernel` 每个 gate arc 用 8 个 lane：

```text
lane 0..7
  el = lane >> 2
  from_attr = lane >> 1
  to_attr = ((lane & 0b100) >> 1) + (lane & 1)
  input_rf = from_attr & 1
  output_rf = to_attr & 1
```

内部流程：

```text
dmpGateKernel(...)
  validate arc_id / timing_id / from_pin / to_pin
  ideal_clock_arc handling
  input_slew = idealClockSlew(...) or pinSlew[from_slot]

  DmpModel::makeGateArcMetaForTiming(...)
  DmpModel::computeGateDriverWaveForSlot(...)
    DmpGateArcMeta::capDelaySlew(...)
    DmpGateArcMeta::estimateRd(...)
    DmpRcParams::selectAlg(...)
    DmpModel::computeZeroC2DriverWave(...)
    DmpModel::computePiDriverWave(...)
    fallback cap table

  arcDelay[gate_arc_id * 8 + lane] = gate_delay
  dmpAtomicSelectFloatKey(pinSlew[to_slot], driver_wave.vo_slew, pick early-min/late-max)

  from_at = idealClockEdgeTime(...) or pinAt[from_slot]
  DmpModel::updateAtWinner(to_slot, from_at + gate_delay, gate_arc_id, from_attr)

  for each fanout net arc from gate output:
    load_elmore = elmore_delay[load_to_slot]
    DmpModel::loadDelaySlewFromDriverWave(...)
      DmpDriverWave::findLoadCrossing(...)
      thresholdAdjust(...)
    dmpAtomicSelectFloatKey(pinSlew[load_to_slot], load_slew, pick)
    dmpAtomicSelectFloatKey(arcDelay[net_arc_delay_slot], wire_delay, pick)
```

最小 atomic helpers：

```text
dmpFloatWinnerKey(value, pick_max)
dmpAtomicSelectFloatKey(addr, value, pick_max)
DmpModel::updateAtWinner(to_slot, at, arc_id, from_attr)
  dmpPackWinner(...)
  atomicMax(pin_at_winner[to_slot], packed)
```

语义：

- early slots 选最小 arrival/delay/slew。
- late slots 选最大 arrival/delay/slew。
- `pin_at_winner` 同时保存 winner value 和 traceback payload。
- `pinSlew` 和 net `arcDelay` 在 finalize 前临时保存 encoded winner key，不是普通 float。

### 7.2.1 Sparse clock 查询

文件：

```text
cpp_to_py/gputimer/core/DmpModel.h
cpp_to_py/gputimer/core/DmpModel.cu
cpp_to_py/gputimer/core/DmpGateProp.cu
```

当前 DMP device state 不再保存 per-pin waveform/slew 和 per-test uncertainty dense arrays，而是保存 sparse clock id + clock table：

```text
pin_clock_ids[num_pins]                  # uint16_t, invalid = 65535
test_clock_ids[num_tests]                # uint16_t
clock_periods[num_clocks]
clock_rise_edges[num_clocks]
clock_fall_edges[num_clocks]
clock_waveform_rise_edges[num_clocks]
clock_waveform_fall_edges[num_clocks]
clock_slews[num_clocks * NUM_ATTR]
clock_setup_uncertainties[num_clocks]
clock_hold_uncertainties[num_clocks]
pin_clock_latency_overrides[num_pins]    # dense float, NaN means unset
```

关键 device helper：

```text
DmpModel::clockPeriodForTest(test_id)
  clock_id = test_clock_ids[test_id]
  return clock_periods[clock_id] if valid else fallback clock_period

DmpModel::pinClockEdge(pin_id, fall)
  clock_id = pin_clock_ids[pin_id]
  override = pin_clock_latency_overrides[pin_id]
  if override finite and clock_id valid:
    return clock_waveform_*_edges[clock_id] + override
  if override finite:
    return override
  return clock_*_edges[clock_id]

DmpModel::idealClockEdgeTime(timing_id, from_pin_id)
  choose rise/fall edge from timing trigger/latch rule
  call pinClockEdge(from_pin_id, use_fall_edge)
  fallback to pinAt if no finite sparse clock edge

DmpModel::idealClockSlew(from_pin_id, attr)
  clock_id = pin_clock_ids[from_pin_id]
  return finite clock_slews[clock_id * NUM_ATTR + attr] or 0

DmpModel::setupUncertaintyForTest(test_id)
DmpModel::holdUncertaintyForTest(test_id)
  clock_id = test_clock_ids[test_id]
  return clock_setup/hold_uncertainties[clock_id] or 0
```

审查重点：

- `clock_id` 是 SDC clock identity，不是 period id；同 period 不同 waveform/uncertainty 不会混掉。
- `set_clock_latency [get_pins ...]` 第一版只支持 scalar override；DMP 侧用 dense `pin_clock_latency_overrides[num_pins]` 修正 edge。
- power CUDA 会复用 `h_dmp_db->pin_clock_ids` 和 `h_dmp_db->clock_slews`，所以 timing 后的 scratch release 不能释放这些 clock tables。

### 7.3 `dmpDirectNetKernel`

`dmpDirectNetKernel` 处理没有 gate driver fanin 的 direct net arcs：

```text
dmpDirectNetKernel(...)
  arc_pos = idx >> 3
  attr = idx & 0b11
  DmpModel::propagateLoadSlewDelay(arc_id, attr)
```

`DmpModel::propagateLoadSlewDelay(...)`：

```text
from_slot = from_pin_id * NUM_ATTR + attr
to_slot = to_pin_id * NUM_ATTR + attr
elmore = elmore_delay[to_slot]
source_slew = pinSlew[from_slot]

if from slot has DMP_DRIVING_CELL_PREFIX_ATTR:
  rebuild virtual driving cell waveform
  loadDelaySlewFromDriverWave(...)
  add extra gate delay over intrinsic delay
else if primary input:
  inputPortDelaySlew(...)
else:
  thresholdAdjust(...)

dmpAtomicSelectFloatKey(pinSlew[to_slot], final_slew, pick)
dmpAtomicSelectFloatKey(arcDelay[net_delay_slot], final_delay, pick)
```

审查注意：当前 kernel 用 `arc_pos = idx >> 3`，但 `attr = idx & 0b11`。这会给每个 direct net arc 发 8 个 work items，其中每个 attr 重复两次。因为写入走 atomic winner key，通常是重复工作而不是直接错值，但这是性能和可读性审查点。

### 7.4 `dmpNetWinnerKernel`

`dmpNetWinnerKernel` 把 net arc 临时 encoded delay 还原成真正 delay，并提交 AT winner：

```text
dmpNetWinnerKernel(...)
  arc_pos = idx >> 2
  attr = idx & 0b11
  delay_idx = (attr << 1) + (attr & 1)
  delay_key = arcDelay[delay_slot]
  decoded_delay = dmpDecodeWinnerFloat(delay_key, pick)
  arcDelay[delay_slot] = decoded_delay
  from_at = pinAt[from_pin_id * NUM_ATTR + attr]
  updateAtWinner(to_pin_id * NUM_ATTR + attr, from_at + decoded_delay, arc_id, attr)
```

这里 net arc 的 `arcDelay` 从 encoded key 变回普通 float。

### 7.5 `dmpPinWinnerKernel`

`dmpPinWinnerKernel` 把每个 pin slot 的 winner 写入正式 timing arrays：

```text
dmpPinWinnerKernel(...)
  packed_at = pin_at_winner[to_slot]
  decode cmp_key and payload
  pinAT[to_slot] = decoded at
  at_prefix_pin[to_slot] = timing_arc_from_pin_id[arc_id]
  at_prefix_arc[to_slot] = arc_id
  at_prefix_attr[to_slot] = from_attr
  pin_at_winner[to_slot] = 0

  if not source:
    decode encoded pinSlew key into ordinary float
    or write NaN if no winner
```

这一步之后，`pinAT`、`pinSlew`、`at_prefix_*` 才是后续 test/backward/power 能直接消费的普通状态。

### 7.6 `dmpTestKernel`

`dmpTestKernel` 在 forward level 末尾计算 endpoint/test 约束并种 RAT：

```text
dmpTestKernel(...)
  pin_id = idx >> 3
  DmpModel::propagatePinTests(level_start_offset + pin_id)

DmpModel::propagatePinTests(...)
  to_pin = level_list[to_pin_idx]
  attr = idx & 0b111
  el = attr >> 1
  rf = attr & 1
  for each backward arc into to_pin:
    timing_id = timing_arc_id_map[arc_id * 2 + el]
    propagateTest(test_id, from_pin_id, attr, el, rf, timing_id, to_slot)

DmpModel::propagateTest(...)
  if test_id == -1: return
  if attr < NUM_ATTR:
    choose related clock slot
    handle ideal clock arcs
    write testRelatedAT[test_id, attr]
    query constraint LUT:
      d_allocator->query(timing_id, frf, rf, related_slew, constrained_slew, 2)
    write testConstraint[test_id, attr]
    write pinRat[to_slot]
    write testRAT[test_id, attr]
```

审查红点：`NUM_ATTR = 4`，但 `propagatePinTests(...)` 用 `attr = idx & 0b111`，并且在调用 `propagateTest(...)` 前已经用 `el = attr >> 1` 访问 `timing_arc_id_map[arc_id * 2 + el]`。当 `attr` 是 4..7 时，`el` 是 2..3，而 `timing_arc_id_map` 的布局是 `[arc_id * 2 + early_or_late]`。这里需要重点核查是否存在越界读，或者是否应该在计算 `timing_id` 前限制 `attr < NUM_ATTR`。

## 8. Backward RAT propagation

文件：

```text
cpp_to_py/gputimer/core/DmpTiming.cu
```

forward 结束后：

```text
for i = level_list_end_cpu.size() - 3 .. 0:
  dmpBackwardKernel<<<blocks, DMP_TIMING_BLOCK_SIZE, shared floats>>>(...)
cudaDeviceSynchronize()
```

调用链：

```text
dmpBackwardKernel(...)
  pin_idx = idx >> 3
  shared from_rats[]
  DmpModel::propagatePinBack(level_start_offset + pin_idx, from_rats)

DmpModel::propagatePinBack(...)
  from_pin_id = level_list[level_idx]
  lane = threadIdx.x & 7
  group_rats = shared group of 8 lanes
  for each fanout arc from from_pin_id:
    clear group_rats[lane]
    __syncwarp(group_mask)
    propagateRAT(arc_id, group_rats)
    __syncwarp(group_mask)
    if lane == 0:
      updatePinRat(arc_id, group_rats)
    __syncwarp(group_mask)
```

`DmpModel::propagateRAT(...)`：

```text
if net arc:
  for i < NUM_ATTR:
    delay_idx = (i << 1) + (i & 1)
    rat = pinRat[to_pin, i] - arcDelay[net_delay_idx]
    write pinRat[from_pin, i]

if gate arc:
  lane maps to el/input/output transition
  skip constraint timing arcs
  from_rats[lane] = pinRat[to_pin, tel_rf] - arcDelay[gate_arc, lane]
```

`DmpModel::updatePinRat(...)` folds the 8 gate lanes back to four `from_pin` attrs:

```text
for ti in 0..7:
  i = ti & 0b111
  fel_rf = i >> 1
  el = i >> 2
  pick min for early, max for late
  atomicExch(pinRat[from_pin, fel_rf], rat)
```

审查重点：

- `propagatePinBack` assumes one warp group owns one source pin and serially scans that source pin's fanout arcs.
- `updatePinRat` uses compare then `atomicExch`, not an atomic min/max packed key. Correctness depends on scheduling preventing multiple groups from racing on the same `from_pin` slot, or on equivalent values.
- Gate constraint arcs are skipped in backward propagation; their RAT seed came from `dmpTestKernel`.

## 9. CUDA error and sync boundaries in timing propagation

`update_timing_dmp_cuda(...)` has these boundaries:

```text
dmp_clear_stale_cuda_error("DMP timing entry")
after dmpResetForwardTargetsKernel: cudaGetLastError()
after each forward kernel launch: cudaPeekAtLastError()
if DMP_PROFILE_KERNELS: cudaDeviceSynchronize per kernel for timing
after all forward levels: cudaDeviceSynchronize()
after each backward kernel launch: cudaPeekAtLastError()
if DMP_PROFILE_KERNELS: cudaDeviceSynchronize per backward level
after all backward levels: cudaDeviceSynchronize()
dmp_clear_stale_cuda_error("DMP timing exit")
```

所以普通非-profile path 中，forward kernel runtime errors generally surface at the final forward `cudaDeviceSynchronize()`，backward runtime errors surface at the final backward `cudaDeviceSynchronize()`。

外层 compare worker 的 `time_stage(..., torch)` 又会在 Python stage 前后同步一次。

## 10. `report_timing_slack()` Python wrapper

文件：

```text
src/core/timing_opt.py
```

调用链：

```text
timer_only.GPUTimer.report_timing_slack()
  time_unit = self.timer.time_unit()
  self.timer.update_endpoints()
  wns_early, tns_early, wns_late, tns_late = self.timer.report_wns_and_tns()
  convert all four tensors to ns:
    value.item() * (time_unit * 1e9)
  self.push_metric(-wns_late, -tns_late)
  return four Python floats
```

`time_unit()` 是 `gt::GPUTimer::time_unit()`，返回 `gtdb.time_unit`。

## 11. `update_endpoints()`

文件：

```text
cpp_to_py/gputimer/core/timing/EndpointSlack.cu
cpp_to_py/gputimer/core/GPUTimer.h
```

入口：

```text
gt::GPUTimer::update_endpoints()
  endpoint_clear_stale_cuda_error("update_endpoints")
  cudaSetDevice(pinAT.device)
  allocate endpoints0 [num_tests, NUM_ATTR], filled NaN
  allocate endpoints1 [num_POs, NUM_ATTR], filled NaN
  allocate endpoint_pin_slacks [num_endpoint_pins, NUM_ATTR], filled FLT_MAX
  fill EndpointSlackModel with raw device pointers
  cudaMalloc / cudaMemcpy d_model
  launch four kernels
  cudaFree d_model
  optional debug count
  endpoint_slacks = torch::cat({endpoints0, endpoints1}, 0).contiguous()
```

### 11.1 `update_endpoints_kernel0`

Per `(test_id, attr)`:

```text
arc_id = test_id2_arc_id[test_id]
to_pin_id = timing_arc_to_pin_id[arc_id]
if pinAT or testRAT is NaN: return
early attr: endpoint slack = pinAT - testRAT
late attr: endpoint slack = testRAT - pinAT
write endpoints0[test_id, attr]
```

This is the test/check endpoint table.

### 11.2 `update_endpoints_kernel1`

Per `(primary_output, attr)`:

```text
pin_idx = primary_outputs[po_idx]
if pinAT or pinRAT is NaN: return
early attr: endpoint slack = pinAT - pinRAT
late attr: endpoint slack = pinRAT - pinAT
write endpoints1[po_idx, attr]
```

This is the primary output endpoint table.

### 11.3 `update_endpoint_pin_slacks_kernel0/1`

These kernels fold test endpoints and PO endpoints into unique endpoint-pin rows:

```text
endpoint_id = test_id2_endpoint_id[test_idx]
or endpoint_id = primary_output2_endpoint_id[po_idx]
slack = early ? AT - RAT : RAT - AT
atomicMinFloatValue(&endpoint_pin_slacks[endpoint_id, attr], slack)
```

`atomicMinFloatValue(...)` uses `atomicCAS` on the float bit pattern and ignores non-finite values.

## 12. `report_wns_and_tns()`

文件：

```text
cpp_to_py/gputimer/core/GPUTimer.cpp
```

入口：

```text
gt::GPUTimer::report_wns_and_tns()
  ep_slacks = endpoint_pin_slacks_for_report(*this)
  slack_e = min(ep_slacks[:, 0:2], dim=1)
  slack_e.clamp_max_(0)
  slack_l = min(ep_slacks[:, 2:4], dim=1)
  slack_l.clamp_max_(0)
  return {
    min(ep_slacks[:, 0:2]),
    sum(slack_e),
    min(ep_slacks[:, 2:4]),
    sum(slack_l)
  }
```

Important: `report_wns_and_tns()` does not directly consume `endpoint_pin_slacks` produced by `update_endpoints()`.

It calls:

```text
endpoint_pin_slacks_for_report(GPUTimer& timer)
  pin_level_slacks = timer.report_pin_slack()
  endpoints_id = unique(timer.timing_raw_db.endpoints_id)
  return nan_to_num(pin_level_slacks.index_select(0, endpoints_id), FLT_MAX)

report_pin_slack()
  pin_slacks = zeros_like(pinAT)
  s1 = pinAT - pinRAT
  s2 = pinRAT - pinAT
  pin_slacks[:, 0:2] = s1[:, 0:2]
  pin_slacks[:, 2:4] = s2[:, 2:4]
  return contiguous pin_slacks
```

所以 `update_endpoints()` 仍会生成 `endpoint_slacks` 和 `endpoint_pin_slacks` side effects，供 `report_endpoint_slack()` / `report_endpoint_pin_slack()` 等接口使用；但当前 `report_wns_and_tns()` 的 WNS/TNS 是从 `pinAT/pinRAT` 重新算出来的。

审查重点：

- 如果将来修改 `update_endpoints()` 的 endpoint aggregation，不一定会改变 `report_wns_and_tns()`。
- 如果 `pinAT/pinRAT` 中有 NaN，`report_pin_slack()` 先产生 NaN，`endpoint_pin_slacks_for_report()` 再对 endpoint rows 做 `nan_to_num(..., FLT_MAX)`。
- `torch::_unique(endpoints_id)` 的顺序和 endpoint side effect rows 不绑定，只影响 report 中选择的 unique pin rows。

## 13. Optional release stage

文件：

```text
cpp_to_py/gputimer/core/DmpModel.cu
cpp_to_py/gputimer/core/DmpModel.cpp
cpp_to_py/gputimer/PyBindCppMain.cpp
```

pybind：

```text
.def("release_dmp_timing_scratch_for_power", &gt::GPUTimer::release_dmp_timing_scratch_for_power)
```

调用链：

```text
gt::GPUTimer::release_dmp_timing_scratch_for_power()
  dmp_release_after_timing(h_dmp_db, dmp_db)
    if !DMP_DEFER_TIMING_ALLOC or null pointers:
      return
    h_dmp_db->release_after_timing()
      cudaFree(pin_at_winner)
      cudaFree(pin_flags)
      cudaFree(r_pi)
      cudaFree(elmore_delay)
      set those host pointers to nullptr
    cudaMemcpy(dmp_db, h_dmp_db, sizeof(DmpModel), cudaMemcpyHostToDevice)
```

审查重点：

- 这个 stage 只有在 `DMP_DEFER_TIMING_ALLOC` 启用时才实际释放；否则是 no-op。
- 它释放 `r_pi` 和 `elmore_delay`。power stage 如果还需要这两个数组会出问题；当前 power path 主要消费 `pinLoad`、`pinSlew`、`pinAT/pinRAT`、`C1/C2`、clock 信息，不应依赖 `r_pi/elmore_delay`。
- sparse clock device tables 保留不释放：`pin_clock_ids`、`test_clock_ids`、`clock_periods`、`clock_rise_edges/fall_edges`、`clock_waveform_*_edges`、`clock_slews`、`clock_setup/hold_uncertainties`、`pin_clock_latency_overrides`。
- release 后不要再调用 `update_timing_dmp()` 或 path trace 依赖的 DMP traceback/RC timing scratch，除非重新准备/分配。

## 14. 主要状态读写表

| 阶段 | 主要读取 | 主要写入 |
|---|---|---|
| `applyDrivingCellSourceSlewKernel` | `driving_cell_sources` 展平数组、Liberty LUT、`C1/C2/r_pi` | source `pinSlew`、`at_prefix_*` |
| `dmpResetForwardTargetsKernel` | pin flags、arc types | encoded `pinSlew` scratch、encoded/net `arcDelay` scratch、gate `arcDelay=NaN` |
| `dmpGateKernel` | `pinAT`、`pinSlew`、Liberty LUT、`C1/C2/r_pi`、`elmore_delay`、sparse clock tables | gate `arcDelay`、encoded `pinSlew`、`pin_at_winner`、net encoded `arcDelay` |
| `dmpDirectNetKernel` | source `pinSlew`、`elmore_delay`、driving-cell tags | encoded `pinSlew`、net encoded `arcDelay` |
| `dmpNetWinnerKernel` | encoded net `arcDelay`、source `pinAT` | decoded net `arcDelay`、`pin_at_winner` |
| `dmpPinWinnerKernel` | `pin_at_winner`、encoded `pinSlew` | `pinAT`、decoded `pinSlew`、`at_prefix_*` |
| `dmpTestKernel` | `pinAT`、`pinSlew`、`test_clock_ids`、clock period/uncertainty tables、constraint LUT | `testRelatedAT`、`testConstraint`、`testRAT`、`pinRAT` |
| `dmpBackwardKernel` | `pinRAT`、`arcDelay`、graph arcs | upstream `pinRAT` |
| `update_endpoints` kernels | `pinAT`、`pinRAT`、`testRAT`、endpoint maps | `endpoint_slacks`、`endpoint_pin_slacks` |
| `report_wns_and_tns` | `pinAT`、`pinRAT`、`endpoints_id` | temporary torch slacks, returned WNS/TNS tensors |
| `release_dmp_timing_scratch_for_power` | `DMP_DEFER_TIMING_ALLOC`, `h_dmp_db` | frees selected timing scratch; keeps sparse clock tables |

## 15. 最小函数索引

审查可以按这个顺序打开代码：

```text
tools/compare_ispd25_route_power_timing.py
  time_stage(...)
  run_xplace_worker(...)

src/core/timing_opt.py
  GPUTimer.update_timing_dmp_route_segments(...)
  GPUTimer.report_timing_slack(...)

cpp_to_py/gputimer/PyBindCppMain.cpp
  pybind methods: update_timing_dmp, update_endpoints, report_wns_and_tns,
                  release_dmp_timing_scratch_for_power

cpp_to_py/gputimer/core/DmpModel.cpp
  GPUTimer::update_timing_dmp()
  apply_dmp_driving_cell_source_slew(...)

cpp_to_py/gputimer/core/DmpTiming.cu
  build_forward_arc_levels(...)
  dmp_upload_vector(...)
  dmp_upload_forward_schedule(...)
  release_dmp_forward_schedule_cuda(...)
  dmp_get_forward_schedule(...)
  update_timing_dmp_cuda(...)
  DmpModel::propagateRAT(...)
  DmpModel::propagatePinBack(...)
  DmpModel::updatePinRat(...)
  dmpBackwardKernel<<<...>>>()

cpp_to_py/gputimer/core/DmpGateProp.cu
  DmpModel::propagateTest(...)
  DmpModel::propagatePinTests(...)
  dmpTestKernel<<<...>>>()
  dmpResetForwardTargetsKernel<<<...>>>()
  dmpDirectNetKernel<<<...>>>()
  dmpPinWinnerKernel<<<...>>>()
  dmpNetWinnerKernel<<<...>>>()

cpp_to_py/gputimer/core/DmpGateEval.cu
  dmpAtomicSelectFloatKey(...)
  DmpModel::makeGateArcMetaForTiming(...)
  DmpModel::computeGateDriverWaveForSlot(...)
  DmpModel::computeZeroC2DriverWave(...)
  DmpModel::computePiDriverWave(...)
  DmpModel::loadDelaySlewFromDriverWave(...)
  DmpModel::propagateLoadSlewDelay(...)
  dmpGateKernel<<<...>>>()
  applyDrivingCellSourceSlewKernel<<<...>>>()
  apply_dmp_driving_cell_source_slew_cuda(...)

cpp_to_py/gputimer/core/DmpGateModel.cuh
  DmpGateArcMeta::hasValidLuts(...)
  DmpGateArcMeta::estimateRd(...)
  DmpRcParams::selectAlg(...)
  DmpRcParams::initZeroC2(...)
  DmpRcParams::initPi(...)
  DmpDriverWave::LoadWaveValueEval

cpp_to_py/gputimer/core/timing/EndpointSlack.cu
  EndpointSlackModel
  atomicMinFloatValue(...)
  update_endpoints_kernel0<<<...>>>()
  update_endpoints_kernel1<<<...>>>()
  update_endpoint_pin_slacks_kernel0<<<...>>>()
  update_endpoint_pin_slacks_kernel1<<<...>>>()
  GPUTimer::update_endpoints()

cpp_to_py/gputimer/core/GPUTimer.cpp
  GPUTimer::time_unit()
  GPUTimer::report_pin_slack()
  endpoint_pin_slacks_for_report(...)
  GPUTimer::report_wns_and_tns()
  GPUTimer::report_endpoint_pin_slack()

cpp_to_py/gputimer/core/DmpModel.cu
  DmpModel::release_after_timing()
  dmp_release_after_timing(...)
  GPUTimer::release_dmp_timing_scratch_for_power()
```

## 16. 人工审查清单

建议按下面顺序查：

1. 入口边界

确认 compare worker 的 `timer` stage 不包含 `init_dmp_rc_route_segments(...)`，而普通 `run_timer.py` wrapper 包含 RC 构建。

2. DMP scratch 生命周期

确认 `h_dmp_db`、`dmp_db`、`pin_at_winner`、`C1/C2/r_pi`、`elmore_delay` 在 timer stage 前有效；确认 optional release 后不会再走 DMP timing。

3. Forward schedule 缓存

确认 schedule 复用条件只看 `num_pins/num_arcs/level_list_size` 是否足够；如果 arc type、fanout、timing map 可能变化，应审查是否需要额外 invalidation。

4. Atomic winner 语义

重点看 `dmpAtomicSelectFloatKey`、`dmpPackWinner`、`dmpDecodeWinnerFloat`、`updateAtWinner` 是否对 early-min 和 late-max 都一致。注意 `pinSlew` / net `arcDelay` 在 finalize 前不是普通 float。

5. `dmpTestKernel` lane/index

重点核查 `propagatePinTests` 的 `attr = idx & 0b111` 和 `timing_arc_id_map[arc_id * 2 + el]`。在 `NUM_ATTR=4` 时，attr 4..7 会让 `el` 变成 2..3，看起来可能越界。

6. Direct net duplicate work

`dmpDirectNetKernel` 每 arc 8 lanes 但只取 `attr = idx & 0b11`，每个 attr 重复两次。通常不改值，但影响性能和 profiler 解释。

7. RAT backward race

`updatePinRat` 使用 compare 后 `atomicExch`。审查时确认同一 `from_pin` slot 是否只由一个 warp group 写，或者同 level 内不存在跨 group 竞争。

8. Endpoint report 语义

不要误以为 `report_wns_and_tns()` 直接用 `update_endpoints()` 的 `endpoint_pin_slacks`。当前 WNS/TNS 从 `pinAT/pinRAT` 重新算，再选择 `endpoints_id` unique rows。

9. CUDA error 定位

普通 path 中很多 kernel 只做 launch error check，runtime error 到 level 后的 `cudaDeviceSynchronize()` 才暴露。开 `DMP_PROFILE_KERNELS=1` 可以把同步压到每个 kernel，便于定位。

10. Power 前 release

`release_dmp_timing_scratch_for_power()` 只有 `DMP_DEFER_TIMING_ALLOC` enabled 时才释放。确认 power CUDA input 不读被释放的 `r_pi/elmore_delay/pin_flags`；同时确认 `pin_clock_ids/test_clock_ids/clock_*` sparse tables 仍保留，因为 power 会复用 `pin_clock_ids` 和 `clock_slews`。
