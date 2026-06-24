# Sparse Clock Plan

目标：去掉当前把 clock 信息展开到每个 pin/test 的大数组，改成：

- 每个真实 SDC clock 一个 `uint16_t clock_id`。
- pin/test 只保存 `clock_id`。
- `set_clock_latency [get_pins ...]` 不生成新的 clock state；第一版只支持 scalar pin override，用 dense per-pin float 表处理。
- CUDA 通过 `clock_id` 查 clock 小表，通过 `pin_clock_latency_overrides[pin_id]` 修正 pin edge。
- host 侧同样通过 `clock_id` 查表；不要保留 `Clock*` per-pin context，避免和 CUDA 侧出现两套 clock 语义。

不要继续使用当前的 `clock_period_id` 语义区分 clock。period id 只能查 period，不能代表 SDC clock domain。

## 当前问题

当前 CUDA 逻辑：

```text
period:
  test_clock_ids[test_id] -> clock_periods[id]

waveform:
  pin_clock_rise_edges[pin_id]
  pin_clock_fall_edges[pin_id]

slew:
  pin_clock_slews[pin_id * NUM_ATTR + attr]

uncertainty:
  test_setup_uncertainties[test_id]
  test_hold_uncertainties[test_id]
```

问题：

1. `pin_clock_periods/pin_clock_rise_edges/pin_clock_fall_edges/pin_clock_slews` 把 clock 属性按 pin 展开，占显存。
2. `test_setup_uncertainties/test_hold_uncertainties` 按 test 展开，也可以通过 capture clock 查表。
3. 当前 `test_clock_ids` 实际是 period id，同 period 不同 clock 会混掉，不适合作 false path / clock domain 语义。

## 新数据结构

放在 `GTDatabase` host 侧：

```cpp
static constexpr uint16_t kInvalidClockId = 65535;

std::vector<std::string> clock_names;  // [clock_id]
std::unordered_map<std::string, uint16_t> clock_name2id;

std::vector<uint16_t> pin_clock_ids;   // [num_pins], real SDC clock id
std::vector<uint16_t> test_clock_ids;  // [num_tests], capture clock id

std::vector<float> clock_periods;              // [num_clocks]
std::vector<float> clock_rise_edges;           // [num_clocks], waveform edge + clock object latency
std::vector<float> clock_fall_edges;
std::vector<float> clock_waveform_rise_edges;  // [num_clocks], no clock object latency
std::vector<float> clock_waveform_fall_edges;
std::vector<float> clock_slews;                // [num_clocks * NUM_ATTR]
std::vector<float> clock_setup_uncertainties;  // [num_clocks]
std::vector<float> clock_hold_uncertainties;   // [num_clocks]

std::vector<float> pin_clock_latency_overrides;  // [num_pins], NaN means no pin override
std::vector<char> pin_clock_is_default_fallback; // [num_pins], host-only warning provenance
```

第一版 latency override 约束：

- 只支持 scalar `set_clock_latency`，也就是不区分 early/late/rise/fall。
- 当前 ISPD2025 SDC 里的 `set_clock_latency` 都是 scalar，所以这个约束不影响当前 case。
- 如果 `set_clock_latency` 带 `-early/-late/-rise/-fall/-min/-max/-source`，第一版打 `logger.warning` 后跳过该 command，不要 silent 当作 scalar 正常支持。
- 由于只支持 scalar，dense `float[num_pins]` 比 `uint16_t id + value table` 更直接；若后续 SDC 里 pin override 特别多，它也只占 `4 * num_pins` bytes，不会像旧 per-pin period/rise/fall/slew 那样膨胀。

删除或停止上传这些旧数组：

```cpp
std::vector<float> test_clock_periods;
std::vector<float> test_setup_uncertainties;
std::vector<float> test_hold_uncertainties;
std::vector<float> pin_clock_periods;
std::vector<float> pin_clock_rise_edges;
std::vector<float> pin_clock_fall_edges;
std::vector<float> pin_clock_slews;
```

## Host 函数拆分

这些函数都作为 `GTDatabase` 成员函数，先在 `GTDatabase_sdc.cpp` 里实现。

### BuildClockIdTablesForSdc

```cpp
uint16_t GTDatabase::BuildClockIdTablesForSdc();
```

输入：

- `clocks`
- `clock_transitions`
- `clock_setup_uncertainty`
- `clock_hold_uncertainty`

输出：

- `clock_names`
- `clock_name2id`
- `clock_periods`
- `clock_rise_edges`
- `clock_fall_edges`
- `clock_waveform_rise_edges`
- `clock_waveform_fall_edges`
- `clock_slews`
- `clock_setup_uncertainties`
- `clock_hold_uncertainties`
- 返回 `default_clock_id`

功能：

- 每个真实 SDC clock 分配一个 `uint16_t clock_id`。
- 相同 period 的不同 clock 也必须有不同 id。
- `clock_rise_edges/fall_edges` 用 `clock.rise_edge()/fall_edge()`，保留 clock object latency。
- `clock_waveform_rise_edges/fall_edges` 用 `clock.waveform_rise_edge()/waveform_fall_edge()`，给 pin latency override 使用。
- `clock_slews` 从 `set_clock_transition` 读到的 `clock_transitions[clock.name()]` 填；没有值为 `NaN`。
- `clock_setup_uncertainties/clock_hold_uncertainties` 从对应 map 填；没有值为 `0.0f`。

### AssignClockIdsToPins

```cpp
void GTDatabase::AssignClockIdsToPins(
    uint16_t default_clock_id,
    std::vector<uint16_t>& net_clock_ids,
    std::vector<char>& pin_clock_is_default_fallback,
    int sdc_threads);
```

输入：

- `default_clock_id`
- `clock_name2id`
- `clocks`
- `gpdb.getPins()`
- `gpdb.getNets()`
- `num_pins`
- `sdc_threads`

输出：

- `pin_clock_ids[num_pins]`
- `net_clock_ids[num_nets]`
- `net_is_clock[num_nets]`
- `pin_clock_is_default_fallback[num_pins]`

功能：

- 初始化所有 pin 为 `default_clock_id`。
- 初始化 `pin_clock_is_default_fallback` 为 `1`，表示该 pin 只是 fallback 到 default clock，不是由 clock source net 显式匹配得到。
- 初始化 `net_clock_ids` 为 invalid clock id，初始化 `net_is_clock` 为 `0`。
- 对每个 clock source pin 建立 clock-net 映射：
  - 找 source pin 所在 net。
  - 如果该 net 还没有 clock，记录到 `clock_net_ids`。
  - 如果该 net 已经有不同 `clock_id`，这是当前 single-scenario timer 不支持的 multi-clock-net 约束；必须报 warning/error，不能静默覆盖。
  - `net_clock_ids[net_id] = clock_id`。
  - `net_is_clock[net_id] = 1`。
  - source pin 的 `pin_clock_ids[source_pin] = clock_id`。
  - source pin 的 `pin_clock_is_default_fallback[source_pin] = 0`。
- 不做全局 `gpdb.getPins()` 扫描。
- 只遍历 `clock_net_ids` 对应 net 的 pin 列表：
  - `for net_id in clock_net_ids`
  - `for pin_id in gpdb.getNets()[net_id].pins()`
  - 把该 clock net 上的所有 pin 映射到 `net_clock_ids[net_id]`。
  - 同步写 `pin_clock_is_default_fallback[pin_id] = 0`。
- `clock_net_ids` 先去重，因此可以按 clock net 并行；不同 net 的 pin 不应重叠。
- host 侧不再维护 `Clock* pin_clock_context`；propagated clock 判断和 latency override 统一通过 `pin_clock_ids[pin_id]` 查 `clock_names` / `clock_waveform_*` / `clock_*` 表。
- `pin_clock_is_default_fallback` 只用于 warning 统计，区分 pin latency override 是落在真实 clock-net/source context 上，还是只 fallback 到 default clock。它不参与 timing 数值、不上传 CUDA。

### InitPinClockLatencyOverrides

```cpp
void GTDatabase::InitPinClockLatencyOverrides();
```

输入：

- `num_pins`

输出：

- `pin_clock_latency_overrides[num_pins]`

功能：

- 在 `readSdc()` visit SDC commands 之前调用。
- 初始化所有 pin 为 `NaN`，表示没有 pin latency override。
- `_read_sdc(SetClockLatency)` 后续直接写 `pin_clock_latency_overrides[pin_id] = delay`。

### `_read_sdc(SetClockLatency)` pin latency override

这一段不是新 API，只是把现有 `set_clock_latency` 处理拆清楚。

`set_clock_latency` 有两类目标，含义不同：

- `set_clock_latency delay [get_clocks ...]`
  - 这是 clock object latency。
  - 继续写到 `Clock::set_latency(delay)`。
  - 后续构建 clock 小表时，clock waveform/edge 会带上这个 clock 级 latency。
- `set_clock_latency delay [get_pins/get_ports ...]`
  - 这是 pin/port 上的 clock latency override。
  - 不创建 clock，也不改 clock object。
  - 只写 `pin_clock_latency_overrides[pin_id] = delay`。
  - 后续 edge 计算用 `pin_clock_ids[pin_id]` 查 clock waveform，再加这个 pin override。

第一版只支持没有 mask 的 scalar delay。若 command 带 `rise/fall/min/max/early/late/source` 任一选项，打印 warning 后跳过该 command。

原因：dense `pin_clock_latency_overrides[pin_id]` 只有一个 float，不能表达 rise/fall、min/max、early/late、source latency 这些分开的语义。跳过比把 masked latency 错误当成全属性 scalar 更安全。

局部逻辑可以写成：

```cpp
if (has_rise_fall_min_max_early_late_or_source(obj)) {
    warn_unsupported_masked_set_clock_latency(obj);
    return;
}

if (target_is_clock) {
    clock.set_latency(delay);
} else if (target_is_pin_or_port) {
    pin_clock_latency_overrides[pin_id] = delay;
}
```

### MapTestsToClockIds

```cpp
void GTDatabase::MapTestsToClockIds(
    const std::vector<uint16_t>& net_clock_ids,
    uint16_t default_clock_id,
    int sdc_threads);
```

输入：

- `test_id2_arc_id`
- `timing_arc_from_pin_id`
- `gpdb.getPins()`
- `net_clock_ids`
- `default_clock_id`
- `sdc_threads`

输出：

- `test_clock_ids[num_tests]`

功能：

- 初始化所有 test 为 `default_clock_id`。
- 对每个 timing test：
  - `arc_id = test_id2_arc_id[test_id]`
  - `clock_pin_id = timing_arc_from_pin_id[arc_id]`
  - `net_id = gp_pins[clock_pin_id].getParNetId()`
  - 若 `net_clock_ids[net_id]` 有效，则 `test_clock_ids[test_id] = net_clock_ids[net_id]`。
- setup/hold uncertainty 不再按 test 存；CUDA 用 `test_clock_ids[test_id]` 查 `clock_setup_uncertainties/clock_hold_uncertainties`。

### Clock Lookup Helpers

```cpp
float GTDatabase::ClockPeriodForPin(int pin_id) const;
float GTDatabase::ClockRiseEdgeForPin(int pin_id) const;
float GTDatabase::ClockFallEdgeForPin(int pin_id) const;
float GTDatabase::ClockSlewForPin(int pin_id, int attr) const;
float GTDatabase::ClockSetupUncertaintyForTest(int test_id) const;
float GTDatabase::ClockHoldUncertaintyForTest(int test_id) const;
```

输入：

- `pin_clock_ids`
- `test_clock_ids`
- `pin_clock_latency_overrides`
- `clock_*` 表

输出：

- 对应 clock value。

pin edge 规则：

```text
clock_id = pin_clock_ids[pin_id]
override = pin_clock_latency_overrides[pin_id]

if override finite and clock_id valid:
  edge = clock_waveform_edge[clock_id] + override
else if override finite and clock_id invalid:
  edge = override
else if clock_id valid:
  edge = clock_edge[clock_id]
else:
  NaN
```

这些 helper 用于 host power、debug dump、ideal clock AT 初始化等位置。

## CUDA / DMP 接口

放在 `DmpModel`：

```cpp
uint16_t* pin_clock_ids;
uint16_t* test_clock_ids;

float* clock_periods;
float* clock_rise_edges;
float* clock_fall_edges;
float* clock_waveform_rise_edges;
float* clock_waveform_fall_edges;
float* clock_slews;
float* clock_setup_uncertainties;
float* clock_hold_uncertainties;
float* pin_clock_latency_overrides;

int clock_count;
```

设备函数：

```cpp
__device__ uint16_t DmpModel::pinClockId(int pin_id) const;
__device__ uint16_t DmpModel::testClockId(int test_id) const;
__device__ float DmpModel::clockPeriodForTest(int test_id) const;
__device__ float DmpModel::pinClockEdge(int pin_id, bool fall) const;
__device__ float DmpModel::idealClockEdgeTime(int timing_id, int from_pin_id) const;
__device__ float DmpModel::idealClockSlew(int from_pin_id, int attr) const;
__device__ float DmpModel::setupUncertaintyForTest(int test_id) const;
__device__ float DmpModel::holdUncertaintyForTest(int test_id) const;
```

CUDA 查询逻辑：

```text
clockPeriodForTest(test_id):
  clock_id = test_clock_ids[test_id]
  return clock_periods[clock_id]

pinClockEdge(pin_id, fall):
  clock_id = pin_clock_ids[pin_id]
  override = pin_clock_latency_overrides[pin_id]
  if override finite and clock_id valid:
    return clock_waveform_{rise/fall}_edges[clock_id] + override
  if override finite:
    return override
  if clock_id valid:
    return clock_{rise/fall}_edges[clock_id]
  return NaN

idealClockSlew(pin_id, attr):
  clock_id = pin_clock_ids[pin_id]
  if clock_id valid:
    return clock_slews[clock_id * NUM_ATTR + attr]
  return 0.0f

setupUncertaintyForTest(test_id):
  clock_id = test_clock_ids[test_id]
  return clock_setup_uncertainties[clock_id]

holdUncertaintyForTest(test_id):
  clock_id = test_clock_ids[test_id]
  return clock_hold_uncertainties[clock_id]
```

## Torch / CUDA 上传策略

`uint16_t` 不建议优先走 `torch::from_blob(..., torch::kInt16)` 再在 CUDA 里解释 unsigned，容易产生语义混乱。

推荐第一版：

- `GTDatabase` clock id host vectors 用 `std::vector<uint16_t>`；`pin_clock_latency_overrides` 用 `std::vector<float>`。
- `GPUTimer` 不通过 tensor 暴露这些 id。
- `DmpModel.cu` 直接 `cudaMalloc/cudaMemcpy` 上传：
  - `pin_clock_ids`
  - `test_clock_ids`
  - `pin_clock_latency_overrides`
  - all `clock_*` float tables
- power CUDA 当前也需要 device 侧 clock slew；第一版必须同步改成 sparse clock-id 查表：
  - power 不再上传/读取 dense `pin_clock_slews[pin * NUM_ATTR + attr]`。
  - power 复用同一批 `pin_clock_ids + clock_slews` device tables，或在 power input/model 生命周期内上传同内容小表。
  - `power_clock_slew_pins` 只保留为 sparse mask，表示哪些 pin 在 power activity/internal power 中允许用 clock slew 覆盖普通 `pinSlew`。

如果后面必须走 Torch tensor：

- 可用 `torch::kInt16` 存 raw 16-bit。
- CUDA 端需要明确 cast/reinterpret，且所有 invalid id 要按 `uint16_t(65535)` 处理。
- 这条路径不建议作为第一版。

## 需要替换的调用点

### `GTDatabase_sdc.cpp::readSdc`

旧逻辑：

```text
intern_clock_period_id()
pin_clock_periods/rise/fall/slews per pin
test_clock_periods/setup/hold per test
```

新逻辑：

```text
BuildClockIdTablesForSdc()
AssignClockIdsToPins()
MapTestsToClockIds()
用 ClockRise/FallEdgeForPin 初始化 ideal clock hostPinAT
用 pin_clock_ids[pin_id] -> clock_names[clock_id] 判断 propagated clock
```

`readSdc()` visit command 前先：

```text
pin_clock_latency_overrides.assign(num_pins, NaN)
pin_clock_is_default_fallback.assign(num_pins, 1)
```

`_read_sdc(SetClockLatency)` 直接写这个 dense array。masked latency 第一版 warning 后跳过，不要在文档或代码里声称支持。

处理 pin latency override warning 时可以继续保留三类统计：

```text
clock-net/source context: pin_clock_is_default_fallback[pin_id] == 0
default-clock context:    pin_clock_is_default_fallback[pin_id] == 1 && pin_clock_ids[pin_id] valid
no-clock context:         pin_clock_ids[pin_id] invalid
```

这个分类只用于日志诊断；实际 edge 计算只由 `pin_clock_ids`、`clock_waveform_*` 和 `pin_clock_latency_overrides` 决定。

### `GPUTimer`

删除旧 pointer：

```cpp
float* test_clock_periods;
float* test_setup_uncertainties;
float* test_hold_uncertainties;
float* pin_clock_periods;
float* pin_clock_rise_edges;
float* pin_clock_fall_edges;
float* pin_clock_slews;
```

如果只有 DMP/Power 使用 clock 表，可以不在 `GPUTimer` 保存这些 pointer，直接让 `DmpModel` 从 `timer->gtdb` 拷贝。

### `DmpModel.cu/.h`

替换：

```text
test_clock_ids + clock_periods 仍保留名字，但语义变成 real clock id。
pin_clock_rise_edges/fall_edges/slews 删除。
test_setup_uncertainties/test_hold_uncertainties 删除。
```

新 helper 替换：

```text
clockPeriodForTest()
idealClockEdgeTime()
idealClockSlew()
propagateTest() uncertainty lookup
isIdealClockTimingArc() 的 waveform finite 判断
```

`isIdealClockTimingArc()` 里原来用 `pin_clock_rise_edges/fall_edges` 作为 SDC waveform evidence。新逻辑改成：

```text
hasPinFlag(DMP_PIN_IDEAL_CLK)
or (hasPinFlag(DMP_PIN_CLK) and pin_clock_ids[pin] valid)
```

### `propagate_infer.cu`

当前 fallback 用 `model->pin_clock_fall_edges[pin_id]`。

新逻辑：

```text
fall_edge = pinClockEdge(pin_id, true)
```

这需要 `InferTimingModel` 也拿到同样的 sparse clock 表，或者保留一个小 helper/header 共享查询逻辑。

### Power host

`PowerActivityHostUtils.cpp::powerClockActivityForPin`

旧：

```text
pin_clock_periods[pin]
pin_clock_rise_edges[pin]
pin_clock_fall_edges[pin]
```

新：

```text
period = ClockPeriodForPin(pin)
rise = ClockRiseEdgeForPin(pin)
fall = ClockFallEdgeForPin(pin)
```

### Power CUDA input/activity

旧：

```text
pin_clock_slews[pin * NUM_ATTR + attr]
```

新：

```text
clock_id = pin_clock_ids[pin]
slew = clock_slews[clock_id * NUM_ATTR + attr]
```

当前 power CUDA 有两处会用 clock slew：

- `PowerActivityOps::maxActivityDensityFromSlew()`：seq clock input pin 的 activity slew cap 可以用 clock slew 覆盖普通 `pinSlew`。
- `PowerInternalContribOps::accumulate()`：internal power LUT 查询的 input slew 可以用 clock slew 覆盖普通 `pinSlew`。

因此 power 侧也要删除 dense slew 值表，只保留 sparse pin mask：

```text
PowerGraphDeviceView:
  delete const float* pin_clock_slews
  add    const uint16_t* pin_clock_ids
  add    const float* clock_slews
  add    int clock_count

PowerInternalContribModel:
  delete const float* pin_clock_slews
  add    const uint16_t* pin_clock_ids
  add    const float* clock_slews
  add    int clock_count
```

device helper 统一写成：

```text
powerClockSlewForPin(pin, attr):
  if pin not in power_clock_slew_pins:
    return NaN
  clock_id = pin_clock_ids[pin]
  if clock_id valid and clock_id < clock_count:
    slew = clock_slews[clock_id * NUM_ATTR + attr]
    if slew finite:
      return slew
  return power_clock_slew_fallback[attr]
```

`buildPowerClockSlews()` 的职责只剩两件事：

- 构造 `power_clock_slew_pins` sparse mask。
- 构造 `power_clock_slew_fallback[NUM_ATTR]`。

它不再通过 `gtdb.pin_clock_slews[pin * NUM_ATTR + attr]` 判断或取值；改用 `pin_clock_ids[pin] -> clock_slews[clock_id * NUM_ATTR + attr]` 判断 clock slew 是否存在。fallback 从第一组 finite `clock_slews` 取；仍然缺失时填 `0.0f`，保持当前 fallback 行为。

pin latency override 不影响 slew。latency override 只改 clock edge；power 的 slew 查询只看 `clock_slews`。

### Debug dump

`EndpointSlack.cu::debug_dump_endpoint_tests`

旧输出：

```text
pin_clock_slew
pin_clock_rise
pin_clock_fall
pin_clock_period
test_clock_period
test_setup_uncertainty
test_hold_uncertainty
```

新输出仍然可以保持列名，但值来自 helper：

```text
ClockSlewForPin(from_pin_id, attr)
ClockRiseEdgeForPin(from_pin_id)
ClockFallEdgeForPin(from_pin_id)
ClockPeriodForPin(from_pin_id)
clock_periods[test_clock_ids[test_id]]
clock_setup_uncertainties[test_clock_ids[test_id]]
clock_hold_uncertainties[test_clock_ids[test_id]]
```

## 实现顺序

1. Host 数据结构和 `readSdc()` 拆分先改完，保留旧数组并用新 helper 生成旧数组做过渡。
2. 改 DMP CUDA，直接使用 `uint16_t clock_id` 和 clock 小表。
3. 删除 DMP 对旧 per-pin clock arrays 的依赖。
4. 改 power host/CUDA clock 查询：`power_clock_slew_pins` 保留为 sparse mask，实际 slew 值通过 `pin_clock_ids + clock_slews` 查。
5. 改 debug dump。
6. 删除旧数组和旧 tensor 字段。

第一步可以先保留旧数组方便对比；最后一步再真正删显存上传路径。

## 审查点

1. `clock_id` 必须是真实 SDC clock identity，不能按 period 合并。
2. `set_clock_latency [get_pins]` 只进 `pin_clock_latency_overrides[pin_id]`，不创建 clock id。
3. `clock_rise_edges/fall_edges` 和 `clock_waveform_rise_edges/fall_edges` 都要保留；override edge 必须用 waveform edge + override。
4. `pin_clock_ids` 和 `test_clock_ids` 用 `uint16_t`，invalid 值统一是 `65535`。
5. 若 clock 数超过 `65534`，直接报错，不要 silent truncate。
6. setup/hold uncertainty 通过 capture clock id 查，不再按 test 展开。
7. false path / clock domain 后续必须通过 `clock_id` 或 clock name 查，不允许再用 period id。
8. 若 pin 没有 clock context 但有 latency override，保持当前行为：rise/fall edge 都等于 override。
9. power CUDA 不允许保留 dense `pin_clock_slews` 作为旁路；DMP 和 power 必须共享同一套 `pin_clock_ids + clock_slews` 语义。
10. 第一版不支持 masked `set_clock_latency`；发现 `-early/-late/-rise/-fall/-min/-max/-source` 必须 warning，不能静默按 scalar 解释。
