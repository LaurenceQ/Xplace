# Power Stage 审查解析

Last reviewed: 2026-06-08

本文展开 `00_POWER_ARCHITECTURE.md` 里的 power 部分，从 Python 验收入口一路追到最底层 C++/CUDA kernel。文件名保留当前要求的拼写：`05_POWER_ANAYLSYS.md`。

入口片段：

```text
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
```

审查时先分清两条主线：

- 功耗数值主线：`report_power_total_cuda()` -> `compute_power_activity_cuda(...)` -> CUDA activity propagation -> switching/internal/leakage component kernels。
- 汇总/分类主线：Python reduce/group/csv + C++ `report_power_group_codes()` CPU 分类。它不参与功耗计算，只决定 group 汇总归属。

## 1. 一眼看完整链路

```text
tools/compare_ispd25_route_power_timing.py::run_xplace_worker(...)
  tensors = time_stage("power", gputimer.timer.report_power_total_cuda, torch)
    pybind GPUTimer.report_power_total_cuda()
      cpp_to_py/gputimer/core/power/report/PowerReport.cpp
        gt::GPUTimer::report_power_total_cuda()
          compute_power_activity_cuda(
            &inst_switching_gpu,
            nullptr,
            &inst_internal_gpu,
            nullptr,
            nullptr,
            &inst_leakage_gpu,
            nullptr,
            nullptr,
            output_power_tensors_cuda=true)
          inst_total_gpu = internal + switching + leakage
          cuda synchronize if tensor is CUDA
          return (internal, switching, leakage, total)

  power = time_stage("power_total_summary", summarize_xplace_power_total(tensors), torch)
    sum first three tensors and recompute total in Python

  group_codes = time_stage("power_group_codes", gputimer.timer.report_power_group_codes, torch)
    pybind GPUTimer.report_power_group_codes()
      cpp_to_py/gputimer/core/power/report/PowerGroups.cpp
        CPU classify node -> int64 code

  power_groups = time_stage("power_group_summary", summarize_xplace_power_groups(...), torch)
    torch.stack first three tensors
    index_add by group code on tensor device
    copy small group sums to CPU

  optional write_power_csv
    copy tensors to CPU
    write per-node internal/switching/leakage/total rows

  write summary JSON
```

`time_stage(..., torch)` 在 stage 前后都会 `torch.cuda.synchronize()`。所以 `power` stage 计到的是 CUDA activity/component kernels 加 C++ 末尾 total tensor 加法；`power_total_summary`、`power_group_summary`、`write_power_csv` 是 Python 侧 reduce/copy/write 成本。

## 2. Python Stage 合约

文件：`tools/compare_ispd25_route_power_timing.py`

### `summarize_xplace_power_total(tensors)`

输入 tensor 顺序必须是：

```text
(internal, switching, leakage, total)
```

但 summary 只对前三个 tensor 求和：

```text
internal = sum(tensors[0])
switching = sum(tensors[1])
leakage = sum(tensors[2])
total = internal + switching + leakage
```

审查点：

- C++ 返回的第四个 `total` tensor 不参与 `power_total_summary` 的 total 标量计算；它主要给 top instance/csv 等 per-node 路径用。
- 如果未来改返回顺序，Python 总功耗和 group/csv 都会错。
- `tensor_sum()` 会 `detach().double().sum().item()`，因此每个 component 都有一次 device-to-host scalar 同步。

### `summarize_xplace_power_groups(gpdb, tensors, group_codes)`

逻辑：

```text
num_nodes = tensors[0].numel()
code_tensor = group_codes.detach().int64()[:num_nodes]
values = stack([internal, switching, leakage], dim=1)
sums = zeros([5, 3])
sums.index_add_(0, code_tensor, values)
copy sums to CPU
group total = internal + switching + leakage
```

group 编码固定：

```text
0 sequential
1 combinational
2 clock
3 macro
4 pad
```

审查点：

- `group_codes` 短于 `num_nodes` 时有 fallback；但已存在的 code 如果是负数或大于 4，`index_add_` 会报错，不会像 CSV fallback 一样自动修正。
- group reduce 在 tensor device 上做；如果 `report_power_total_cuda()` 返回 CUDA tensor，这一步不会逐 node 拷 CPU。
- `POWER_GROUPS` 顺序必须和 C++ `report_power_group_codes()` 编码一致。

### `write_xplace_power_csv(gpdb, tensors, path, group_codes)`

逻辑：

- 读取 `gpdb.node_id2celltype_name()` 和可选 `gpdb.node_id2node_name()`。
- 把所有 tensor `detach().cpu().double()[:num_nodes]`。
- 用 `power_group_names(...)` 把 group code 转 group name；invalid code 有 cell type fallback。
- 每个 node 写 `name, cell_type, power_group, internal, switching, leakage, total`。

审查点：

- CSV 是 CPU 全量 copy，不能和核心 CUDA power kernel 性能混在一起看。
- `num_nodes = min(len(cell_types), tensors[0].numel())`，所以如果 gpdb node 表和 C++ power tensor 长度不一致，CSV 会静默截断。

## 3. pybind 和 C++ Report API

文件：

- `timer_only/timing_opt.py`
- `cpp_to_py/gputimer/PyBindCppMain.cpp`
- `cpp_to_py/gputimer/core/GPUTimer.h`
- `cpp_to_py/gputimer/core/power/report/PowerReport.cpp`
- `cpp_to_py/gputimer/core/power/report/PowerGroups.cpp`

Python wrapper `timer_only.timing_opt.GPUTimer` 创建 C++ timer：

```text
create_timing_rawdb(...)
create_gputimer(params, rawdb, gpdb, timing_raw_db)
self.timer.init()
self.timer.levelize()
```

power API 通过 pybind 暴露：

```text
report_power_activity_cuda
report_power_switching_cuda
report_power_internal_cuda
report_power_internal_arcs_cuda
report_power_leakage_cuda
report_power_leakage_rows_cuda
report_power_total_cuda
report_power_group_codes
```

`PowerReport.cpp` 是 thin wrapper。除 `report_power_group_codes()` 外，CUDA report 基本都转到：

```cpp
GPUTimer::compute_power_activity_cuda(...)
```

非空输出指针决定请求哪些结果：

```text
switching: inst_switching_cpu 或 pin_switching_cpu 非空
internal: inst_internal_cpu 或 internal_row_power_cpu/meta 非空
leakage:  inst_leakage_cpu 或 leakage_row_power_cpu/meta 非空
activity: 所有 component 输出指针都为空
```

`report_power_total_cuda()` 特殊点：

- 传入 `output_power_tensors_cuda=true`，所以 internal/switching/leakage 保持 CUDA tensor。
- C++ 里新建 `inst_total_gpu`，执行 `copy_ + add_ + add_`。
- 如果 total tensor 是 CUDA，会 `torch::cuda::synchronize()`。

审查点：

- 变量名仍叫 `*_cpu`，但 `output_power_tensors_cuda=true` 时实际是 CUDA tensor；不要按名字误判 device。
- `report_power_total_cuda()` 是当前默认验收路径，其他 row/arcs report 是 debug/probe 路径。

## 4. Group Codes CPU 分类

入口：`gt::GPUTimer::report_power_group_codes()`

输出：

```text
torch::Tensor int64 CPU [num_nodes]
```

流程：

1. 建 `pin_to_net`、`pin_to_node`、`is_driver_pin`、`is_load_pin`。
2. 从 `gtdb.net_is_clock` 和 `gtdb.pin_is_clk` 标记 clock nets。
3. 沿 clock-transparent combinational cell 向前传播 clock net。透明条件是 CORE combinational cell，并且输出 function 是某个 clock input 的直连或反相。
4. 标记 `is_clock_pin`。
5. node 分类优先级：

```text
IOPin / FloatIOPin / PAD cell -> pad
non-CORE cell type            -> macro
all output pins in clock net  -> clock
Liberty sequential cell       -> sequential
otherwise                     -> combinational
```

审查点：

- clock group 是从 clock network 语义推导，不等价于 sequential cell。
- clock-transparent 判断会编译 Liberty function；function 不支持或不是 direct/inverted pass-through 时不会继续传播。
- 这条路径和 CUDA activity 的 `buildPowerClockPins(...)` 有相似逻辑，但不是同一个函数；两边规则变更要同步审查。

## 5. `compute_power_activity_cuda` Host 主干

文件：`cpp_to_py/gputimer/core/power/cuda_input/PowerCudaInputBuild.cpp`

函数签名：

```cpp
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

稳定顺序：

```text
1. 基本参数和请求类型
2. host pin/node/net maps
3. host pin direction classification
4. net driver pins
5. clock gate maps
6. clock pin activity
7. Liberty function expression table
8. sequential table
9. clock slew sparse override
10. root seed table
11. internal/leakage row table
12. row meta CPU output
13. node-port-pin map
14. upload host vectors to CUDA tensors
15. power-specific levelize
16. finalize root inputs
17. choose activity levels
18. allocate run output buffers
19. choose DMP load pointers
20. assemble device views and model
21. run_power_activity_cuda_launcher(model)
22. optional chunked component rows
23. finish outputs
```

### 5.1 基本参数

关键值：

```text
n = gtdb.pin_names.size()
num_nodes = GPUTimer::num_nodes
sdc_time_scale = canonicalPowerTimeScale(gtdb.sdc_time_unit or gtdb.time_unit)
min_period_sec = min(clock.period * sdc_time_scale)
default_density = 0.1 / min_period_sec
clock_density = 2.0 / min_period_sec
```

没有 clock period 时 fallback 到 `gtdb.time_unit` 或 `1e-9`。

审查点：

- `default_density` 和 `clock_density` 是 activity seed 的根默认值，单位必须和 SDC time scale 一致。
- 函数开头调用 `clear_power_cuda_error()` 清掉历史 stale CUDA error；真正错误仍应由后续 `check_power_cuda_error` 抓。

### 5.2 Host pin maps 和 clock 信息

函数来源：

- `buildPowerPinNodeNetMaps(...)`
- `classifyPowerPins(...)`
- `buildPowerNetDriverPins(...)`
- `buildPowerClockGateMaps(...)`
- `buildPowerClockPinActivity(...)`

产物：

```text
h_pin_to_node[n]
h_pin_to_net[n]
h_is_load_pin[n]
h_is_driver_pin[n]
h_is_cell_pin[n]
h_net_driver_pin[num_nets]
h_clock_gate_out_for_input[n]
h_clock_gate_clock_for_out[n]
h_clock_gate_enable_for_out[n]
h_is_clock_gate_clock_pin[n]
h_clock_pins
h_clock_pin_densities
h_clock_pin_duties
h_clock_pin_enqueue
```

`buildPowerClockPinActivity(...)` 会调用 `buildPowerClockPins(...)`。clock pin 来源包括：

- SDC/OpenSTA 标记的 clock net。
- integrated clock gate clock pin。
- clock-transparent CORE combinational cell 前向传播得到的额外 clock pins。

审查点：

- `classifyPowerPins` 对 IO primary input/output 有特殊 direction 规则。
- `buildPowerClockGateMaps` 依赖 Liberty clock gate port 属性：clock、enable、out 三个端口必须都能识别。
- clock pin activity 里的 `enqueue` 对 sequential clock load 默认不继续传播 clock tree，除非它是 combinational clock tree load。

### 5.3 Expression 和 sequential 表

文件：`PowerCudaInputExpr.cpp`

核心 struct 在 `GPUTimer.h`：

```text
GpuPowerExprOpHost:
  arg: physical pin id, or template negative port encoding
  var_key: Liberty port id, used by BDD/direct evaluator variable identity
  op: 0 pin, 1 const0, 2 const1, 3 not, 4 and, 5 or, 6 xor, 7 missing pin
  zero_density: scan-enable density suppression flag

GpuPowerSeqHost:
  data_expr_id
  clk_expr_id
  node_id
  q_pin
  qn_pin
  is_latch
```

`buildPowerCudaExprInputs(...)`：

- 遍历每个 node 的 output pin。
- 对 Liberty output `function_expr_` 建表达式。
- 优先尝试 template expression：端口用 `-2 - port_id` 编码，运行时靠 `node_port_pin_map` resolve。
- 如果当前 node 缺少某些端口，走 instance expression，并通过 rawdb const net 或 `XPLACE_POWER_CONST_PORT_FILE` 解析常量。
- 对 missing const output 构建 `missing_func_out_start/list`，用于输入变化后补算这些 output。

`buildPowerCudaSeqInputs(...)`：

- 遍历 Liberty `sequentials_`。
- 建 next-state data expression 和 clock expression。
- 通过 output function 匹配 Q/QN physical pin。
- 标记 `is_seq_output_pin`。
- 可选标记 sequential clock input pin。
- 建 `pin_seq_list_start/list`，从 load pin 找到其所属 sequential records。

审查点：

- template expression 和 instance expression 混用时，`node_port_pin_map` 必须覆盖所有 template port。
- `pin_id` 是全链路物理 pin id，不能和 Liberty port offset 混用。
- `op=7 missing pin`、`zero_density`、scan enable density 忽略都影响 activity 和 internal/leakage duty。

### 5.4 Roots、arc skip 和 power level

文件：`PowerCudaInputRoots.cpp`

`buildPowerCudaRootInputs(...)` 负责 seed roots：

```text
primary inputs
clock pins
timing zero-indegree pins
floating load pins
timing loop roots
sequential feedback q/qn or state seeds
constant-generator outputs
```

同时生成：

```text
is_clock_pin[n]
is_primary_input[n]
primary_inputs
feedback_seed_pins
feedback_seed_seqs
seq_output_arc_keep bitset
disabled_loop_arc
seed_reason debug strings
```

`buildPowerCudaArcSkipInputs(...)`：

- 默认跳过 `gtdb.arc_id2test_id[arc] != -1` 的 constraint/test arc。
- 可选跳过 timing loop disabled arc。
- `XPLACE_POWER_APPLY_FALSE_PATHS` 打开时才会把 mapped false path arc 当 power cut。
- 如果 false path 禁用 net arc，会关闭 direct flat net fanout，避免绕过 arc skip。

`levelize_power(...)` 在 `cpp_to_py/gputimer/core/levelize.cu`：

- 使用 power-specific arc predicate 计算 Kahn-style level list。
- 输入包括 `is_seq_output_pin`、`seq_output_arc_keep`、`arc_skip`、`is_load_pin`、net driver 和 flat net2pin。
- 输出：

```text
power_level_list
power_level_list_end_cpu
power_level_root_pins_cpu
power_pin_level_cpu
```

`finalizePowerCudaRootInputs(...)` 在 levelize 后补充 power zero-fanin root，并 dump/debug root stats。

审查点：

- power propagation edge predicate 必须和 CUDA `enqueueAdjacent(...)` 一致，否则 level 不覆盖实际传播边。
- direct flat net fanout 和 timing arc fanout 是两条传播路径；false path/net arc skip 审查时要特别确认没有被 direct net fanout 绕过。
- sequential output arcs 默认会被剪掉一部分，`seq_output_arc_keep` 是关键 bitset。

### 5.5 Internal 和 leakage row 构建

文件：`PowerCudaInputRows.cpp`

`GpuPowerInternalHost` 字段语义：

```text
internal_power_id: Liberty internal_power row id
node_id: instance/node id
to_pin: power row attached port physical pin
from_pin: related input pin for output internal_power
duty_expr_id: when/function expression id
duty_pin: diff duty pin
denom_group: output internal_power normalization group
energy_unit: Liberty internal_power energy unit
kind: 0 input internal_power, 1 output internal_power
duty_mode:
  0 const1
  1 expression duty
  2 diff duty
  3 const0.5
  4 const0
positive_unate: slew rise/fall selection
```

`buildPowerCudaInternalRows(...)`：

- 先按 libcell 统计 reserve，减少 vector realloc。
- input pin internal_power：通常 `kind=0`，根据 when expression 决定 duty。
- output pin internal_power：通过 `related_port` 找 `from_pin`，建立 per-output/per-power-ground denom group，用 related input activity 分摊 output internal power。
- `positiveUnateForPower(...)` 决定 related input slew 的 rise/fall 方向。

`GpuPowerLeakageRowHost` 和 `GpuPowerLeakageGroupHost`：

```text
Leakage row:
  node_id
  group_id
  leakage_power_id
  when_expr_id
  leakage Watts

Leakage group:
  node_id
  cell_leakage Watts fallback
```

`buildPowerCudaLeakageRows(...)`：

- 根据 MAX lib 的 leakage_power row 建条件 leakage rows。
- cell fallback leakage 可来自 MIN lib 对应 cell 的 `leakage_power_`。
- 按 related pg pin 建 leakage group。

审查点：

- internal power 使用 MAX table range；leakage fallback 可能从 MIN lib cell 取值，单位必须明确。
- `denom_group` 是 output internal_power 分摊正确性的核心。
- `duty_mode=1/2` 会走表达式 evaluator，`0/3/4` 走 fast kernel。

## 6. Host-to-CUDA 上传和 Device View ABI

文件：

- `PowerCudaInputBuildInternal.h`
- `PowerCudaModel.h`

上传工具：

```text
powerCudaIntTensor(vector<int>)      -> int32 CUDA tensor
powerCudaU8Tensor(vector<uint8_t>)   -> uint8 CUDA tensor
powerCudaFloatTensor(vector<float>)  -> float CUDA tensor
powerCudaBytesTensor(vector<T>)      -> byte CUDA tensor, device side reinterpret_cast<T*>
```

`PowerCudaUploader` 只包装这些函数并提供 debug/sync 标记。

核心 device view：

```text
PowerGraphDeviceView:
  graph CSR, net fanout, pin flags, timing/load/slew/DMP pointers, level pointers

PowerExprDeviceView:
  expression ops/start/count, node-port-pin map, missing function outputs

PowerActivityState:
  roots, clocks, sequentials, feedback seeds

PowerActivityConfig:
  default_density, clock_density, time_unit, max passes, trace pins,
  precomputed activity pointer, clamp/override knobs

PowerComponentDeviceView:
  internal/leakage rows, power LUT allocator, voltage/cap unit,
  output pointers

PowerActivityCudaModel:
  all device views plus n and output activity pointer
```

审查点：

- device view 只保存 raw pointers；对应 torch tensor 必须在 launcher 返回前保持活着。当前实现把 tensor 局部变量保留在 `compute_power_activity_cuda` 栈上，直到 launcher 和 chunk launcher 都结束。
- bytes tensor 的 device pointer 必须按原 struct alignment/size reinterpret；`GpuPowerExprOpHost` 和 `GpuPowerInternalHost` 有 `static_assert` 保护大小。
- 空 vector 上传成 1-element sentinel，避免空 tensor data_ptr 问题；kernel 必须同时看 count，不可只看 pointer 非空。

## 7. CUDA Activity Launcher

文件：`cpp_to_py/gputimer/core/power/cuda_activity/PowerCudaActivity.cu`

入口：

```cpp
run_power_activity_cuda_launcher(const PowerActivityCudaModel& model)
```

主步骤：

1. 解包 `model` 到 host local aliases。
2. 根据请求判断 scratch 需求：
   - 是否需要 activity propagation。
   - 是否需要 density/duty。
   - 是否需要 origin。
   - 是否需要 sequential state。
   - 是否 inline 计算 internal/leakage。
3. 分配 scratch：
   - `density/duty`
   - `prev_density/prev_duty`
   - `seq_pin_density/seq_pin_duty/seq_pin_valid`
   - `origin/prev_origin`
   - `active` bitset
   - `active_level`
   - `visit_active`
   - `pending_seq/pending_seq_count`
4. 把 `PowerActivityCudaModel` 和 `PowerActivityScratchView` 拷到 device。
5. 把 env knobs 拷到 device symbols：
   - clock override
   - min density/duty
   - slew cap disable
   - sequential clock limit tolerance
   - direct expression enable/check/max vars
6. activity source：
   - 有 precomputed activity：copy/unpack，然后跳过 propagation。
   - 无 precomputed activity：seed roots，按 scheduler 传播。
7. pack activity output 和 optional final dump。
8. component kernels：
   - switching
   - internal denom/contrib
   - leakage rows/summary
9. free scratch/model buffers。

审查点：

- `density/duty` 可能 alias caller output buffer，也可能是 launcher 自己 malloc 的 scratch。
- `origin` 只在 propagation/debug 需要；component power 只需要 density/duty。
- `precomputed_activity` 是 CPU activity fallback 或 probe path 的旁路，必须保持 shape `[n, 3]` 并按 row-major `pin*3 + field` 解包。

## 8. Activity Propagation 最小语义单元

文件：

- `PowerCudaActivityDevice.cuh`
- `PowerCudaActivityDevice.cu`
- `PowerCudaActivitySeeds.cu`
- `PowerCudaActivityQueue.cu`

### `PowerActivityOps::setActivity(...)`

这是最小写 activity 单元：

```text
输入: pin, density, duty, origin, force
处理:
  if origin is clock and override disabled, reject non-force update
  clamp density >= 0
  clamp density <= min(slew cap, clock cap) unless force
  abs(density) < min_activity_density -> 0
  duty clamp to [0, 1], optional min duty snap
  changed if density/duty percent change > 1% or origin changed
  write density/duty/origin
```

origin 当前语义：

```text
1 default/primary/root
2 clock
3 propagated/computed
4 case value
0 unknown/unset
```

审查点：

- slew cap 来自 `pinSlew`，sequential clock input 可用 sparse clock slew override。
- `force=true` 会绕过普通 slew cap，只受 clock density cap。

### `PowerActivityOps::enqueueAdjacent(pin)`

传播边：

1. 如果 `pin` 是 net driver，且 flat net fanout 可用，直接把同 net load pins 置 active。
2. 遍历 `pin_forward_arc_list`：
   - 跳过 `arc_skip`。
   - 跳过被剪掉的 sequential output arc。
   - 把 `to_pin` 置 active。
3. 同时根据 `pin_power_level[to_pin]` 标记 `active_level[level]=1`。

审查点：

- 这是 `levelize_power(...)` 的运行时镜像；两边 predicate 不一致会导致漏传播或活跃 level 不完整。
- direct net fanout 对性能重要，但也可能绕过某些 arc-level exception，所以 `arc_skip_inputs` 里有关闭 flat net fanout 的特殊分支。

### `PowerActivityOps::processLevelPin(pin, defer_pending_seq)`

这是每个 active pin 的核心处理：

```text
case value pin:
  set constant duty, density=0

load pin:
  inherit net driver density/duty

driver pin:
  if seq output valid:
    use seq_pin_density/duty
  else if function expr exists:
    PowerExprView::activity(expr_id)
    set output activity
  apply clock gate output formula

if changed load pin:
  mark pending sequential records
  enqueue clock-gate output
  recompute missing-function outputs if needed

enqueue adjacent pins
```

### `PowerActivityOps::seedSeqActivity(seq_id, direct_ordered)`

sequential fixed-point 单元：

```text
data activity = activity(next_state_expr)
clock activity = activity(clock_expr) or default clock density/duty
if data density exceeds 0.5 * clock density:
  latch: out_density = data_density * clock_duty
  ff:    out_density = 2 * data_duty * (1 - data_duty) * clock_density
else:
  out_density = data_density
q duty = data duty
qn duty = 1 - data duty
write seq_pin_* and activate q/qn
```

审查点：

- sequential loop 停止条件是 `pending_seq_count == 0` 或达到 `max_activity_passes`。
- `XPLACE_POWER_REQUIRE_KNOWN_SEQ_DATA` 会要求 next-state expression 有已知 activity input。

## 9. Expression Evaluator

文件：`PowerCudaActivityDevice.cu`

`PowerExprView::resolvePinArg(arg)`：

- `arg >= 0`：已经是 physical pin id。
- `arg == -1`：缺失/常量，返回 -1。
- `arg <= -2`：template port id，使用 `node_port_pin_start/list` 和 `node_id` resolve physical pin。

`PowerExprView::activity(expr_id, density, duty)`：

1. 默认先尝试 `PowerDirectExprEval`。
2. direct path 安全条件：
   - unique var count 不超过 `g_power_direct_expr_max_vars`。
   - expression op count/stack 在限制内。
3. 如果 direct 不可用或要求 check，走 BDD fallback `PowerBddExprEval`。
4. `XPLACE_POWER_CHECK_DIRECT_EXPR` 可以比较 direct 和 BDD 差异并打印 mismatch。

Direct activity 公式：

```text
NOT: density same, duty = 1 - duty
AND: density = da * ub + db * ua, duty = ua * ub
OR:  density = db * (1 - ua) + da * (1 - ub), duty = ua + ub - ua*ub
XOR: density = da + db, duty = ua*(1 - ub) + (1 - ua)*ub
```

BDD fallback：

- 构建最多 32 个变量、256 个 BDD node 的 device BDD。
- duty 通过 BDD probability 求值。
- density 通过每个变量 cofactors 的 diff duty 加权求和。

审查点：

- direct path 对重复变量或复杂表达式会拒绝，转 BDD。
- `zero_density` 变量 duty 仍可参与逻辑，但 density 贡献置 0。
- template expression 的 `var_key` 是 Liberty port id，不是 physical pin id；这个设计保证同一个 port 在不同 instance 上有稳定变量身份。

## 10. Scheduler：默认 level scan 和 frontier

默认路径在 host 侧扫描 active levels：

```text
seed case / primary input / clock
drain_bfs()
while pending_seq_count > 0 and pass < max_activity_passes:
  seed pending sequentials
  drain_bfs()
```

`drain_bfs()`：

- host 拷回 `active_level`。
- 找下一个 active level。
- `power_snapshot_level_active_kernel` 清 active bit 并写 `visit_active`。
- `power_visit_level_kernel` 并行处理 level pins。
- 可通过 env 走 serial level debug kernel。

frontier 路径：

- `XPLACE_POWER_ACTIVITY_FRONTIER=1` 打开。
- 使用 `PowerActivityQueueView` 的 per-level queue。
- persistent cooperative kernel 或 ordered single-thread kernel drain queue。
- 如果 GPU 不支持 cooperative launch，会 fallback 到默认 level scan。

审查点：

- 默认路径有 host-device 往返，便于调试但可能增加小 level overhead。
- ordered/serial/frontier 多为 debug/实验路径；验收路径通常看默认 scheduler。

## 11. Component Power Kernels

文件：`PowerCudaActivityComponents.cu`

### 11.1 Switching

kernel：

```text
power_switching_kernel
```

公式：

```text
load_internal = max late rise/fall load over attrs 2..3
if DMP C1/C2 available:
  load_internal = max(C1 + C2 over attrs 2..3)
switching = 0.5 * load_internal * cap_unit * voltage^2 * density
atomicAdd inst_switching[node]
optional pin_switching[pin] = switching
```

只对 cell driver pin 计算。

审查点：

- Direct DMP RC 模式下 switching load 优先用 `dmp_C1 + dmp_C2`，不是 legacy `pinLoad`。
- 只取 late/max attrs 2/3，和 OpenSTA max load 语义对齐。

### 11.2 Internal

kernels：

```text
power_internal_denom_fast_kernel
power_internal_denom_kernel
power_internal_contrib_fast_kernel
power_internal_contrib_kernel
```

两阶段：

1. denom：对 output internal_power，用 `from_pin density * row_duty` 累加到 denom group。
2. contrib：计算每 row 权重，并查 Liberty internal_power LUT。

贡献公式：

```text
weight =
  input internal_power: row_duty
  output internal_power: (density[from_pin] * row_duty) / denom[group]

load_internal =
  output row and DMP C1/C2 available: max(C1 + C2)
  otherwise 0

slew pin =
  input row: to_pin
  output row: from_pin

energy = average(query_internal_power(row.internal_power_id, rise/fall, slew, load))
power = weight * energy * energy_unit * density[to_pin]
atomicAdd inst_internal[node]
optional internal_row_power[row_idx] = power
```

审查点：

- fast kernels handle duty modes 0/3/4；expr kernels handle duty modes 1/2。
- `positive_unate` 决定 output row 用 related input rise/fall 还是交换 rise/fall。
- `energy_unit` fallback 是 `cap_unit`，这块单位逻辑要和 Liberty parser 保持一致。

### 11.3 Leakage

kernels：

```text
power_leakage_row_fast_kernel
power_leakage_row_kernel
power_leakage_summary_kernel
```

row 阶段：

```text
no when:
  weighted = leakage
when:
  weighted = leakage * duty(when_expr)
accumulate group_cond_leakage/group_cond_duty_sum/group_cond_count
optional leakage_row_power[row_idx] = weighted
```

summary 阶段：

```text
if group has conditional rows:
  fallback_duty = 1 - group_cond_duty_sum
  leakage = group_cond_leakage + cell_leakage * fallback_duty
else:
  leakage = cell_leakage
atomicAdd inst_leakage[node]
```

审查点：

- conditional row duty sum 大于 1 时 fallback duty 会变负；当前 kernel 没 clamp。审查 alignment 时要确认 OpenSTA 对 overlapping `when` rows 的语义。
- leakage unit 在 host rows 已经转成 Watts。

## 12. Row Chunking 旁路

文件：

- `PowerCudaInputBuild.cpp::runPowerChunkedComponents(...)`
- `PowerCudaChunkLaunchers.cu`

触发条件：

```text
internal_row_bytes > XPLACE_POWER_INTERNAL_ROW_CHUNK_BYTES
leakage_row_bytes > XPLACE_POWER_LEAKAGE_ROW_CHUNK_BYTES
default threshold = 8 GiB
```

行为：

- main launcher 仍负责 activity propagation。
- 被 chunk 的 internal/leakage rows 不传给 main launcher inline component view。
- `preparePowerCudaRunBuffers(...)` 会确保有 density/duty activity buffer。
- chunk launcher 分段上传 row bytes tensor。
- internal chunk：
  - 所有 chunk 先累加到全局 denom。
  - 再逐 chunk contrib。
- leakage chunk：
  - 逐 chunk row accumulation。
  - 最后 summary。

审查点：

- chunk path 和 inline path 共用 component kernels，但 model 的 activity 来源可能是 packed precomputed activity 或 density/duty pointers。
- denom group id 是全局 id，不能按 chunk 局部重编号。

## 13. 输出收尾

`finishPowerActivityOutputs(...)`：

```text
if output pointer non-null:
  outputPowerTensorForRequest(cuda_tensor, output_power_tensors_cuda)

if want_activity_cpu:
  activity_cpu = out_gpu.to(CPU).transpose(0, 1).contiguous()
  return [n, 3] density/duty/origin
else:
  return empty CPU [0, 3]
```

当前默认 `report_power_total_cuda()`：

- internal/switching/leakage 输出指针非空。
- `want_activity_cpu=false`。
- `output_power_tensors_cuda=true`。
- 返回给 Python 的前三个 tensor 是 CUDA `[num_nodes]`。
- 第四个 total tensor 在 `report_power_total_cuda()` 中额外生成。

审查点：

- 其他 API 默认 `output_power_tensors_cuda=false`，所以返回 CPU tensor。
- activity report 返回的是 `[num_pins, 3]` CPU tensor，不是 node power tensor。

## 14. 代码审查检查清单

按这个顺序看最不容易漏：

1. Python stage：
   - stage name 和 CUDA synchronize 是否符合计时口径。
   - `COMPONENTS` 顺序是否仍是 internal/switching/leakage/total。
   - total summary 是否故意不使用第四个 total tensor。

2. pybind/report：
   - pybind 暴露的方法名和 C++ 签名是否一致。
   - `report_power_total_cuda()` 是否仍传 `output_power_tensors_cuda=true`。
   - C++ 返回 tuple 顺序是否与 Python zip 顺序一致。

3. host input build：
   - `pin_id/node_id/net_id` 编号空间是否全程一致。
   - `is_load_pin/is_driver_pin/is_cell_pin` 对 IO、inout、macro 的规则是否符合预期。
   - clock net/group/activity 三处逻辑是否同步。
   - expression template/instance fallback 是否正确处理缺失端口和 const net。
   - sequential Q/QN pin 匹配是否依赖 function expr，是否覆盖目标 library。
   - root seed policy 和 env knob 是否影响验收默认。

4. levelize/propagation：
   - `levelize_power` predicate 和 `enqueueAdjacent` predicate 是否一致。
   - direct flat net fanout 是否会绕过需要跳过的 arc。
   - sequential output arc keep bitset 是否按预期保留/剪枝。

5. CUDA ABI/lifetime：
   - host vector -> torch tensor -> raw pointer 生命周期是否覆盖 launcher 和 chunk launcher。
   - byte tensor reinterpret struct size/alignment 是否仍有 `static_assert`。
   - empty tensor sentinel 是否配合 count 使用。
   - `.cpp` 文件不要新增 CUDA runtime 或 kernel launch；CUDA 错误检查放 `.cu` launch wrapper。

6. component formulas：
   - switching 是否用 DMP C1/C2 覆盖 legacy load。
   - internal denom group 是否全局稳定。
   - internal slew/load/energy unit 是否与 OpenSTA 对齐。
   - leakage conditional rows fallback duty 是否符合目标语义。

7. output/group/csv：
   - group code 范围必须是 0..4。
   - Python group summary 和 CSV fallback 策略不完全相同。
   - CSV 和 top instance 路径会触发 CPU 全量 copy。

## 15. 高风险点速记

- `report_power_total_cuda()` 返回顺序是全链路硬合约：internal、switching、leakage、total。
- `summarize_xplace_power_total()` 不用第四个 total tensor；如果只修 C++ total tensor，不会改变 summary total。
- `report_power_group_codes()` 是 CPU 分类，和 CUDA power 数值计算解耦。
- `compute_power_activity_cuda()` 是 host orchestration，不是单个 CUDA kernel；性能 profile 要继续拆 `pin_maps/uploads/levelize_power/launcher/chunk_components/downloads`。
- `pin_id` 必须是 gpdb/gtdb/timing/power expression 共同的 physical pin id。
- expression 的 template negative arg 需要 `node_port_pin_map` 才能在 device 上 resolve。
- direct net fanout 与 timing arc fanout 并存，exception/skip 逻辑要双查。
- chunked internal/leakage 只切 rows，不切 activity graph。
- env knobs 很多，验收默认和 debug 实验路径要分开审查。
