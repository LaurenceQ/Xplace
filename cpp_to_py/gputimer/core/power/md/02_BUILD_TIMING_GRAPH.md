# Build Timing Graph 审查笔记

Last reviewed: 2026-06-08

本文覆盖下面这段入口，重点是从 Python wrapper 追到最底层 C++ / CUDA 函数，方便逐层审查 timing graph 构建和初始化。

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
      gt::GPUTimer(gtdb, timing_raw_db)
  gt::GPUTimer::initialize()
  gt::GPUTimer::levelize()
```

注意：`GTDatabase::ExtractTimingGraph()` 本身是 CPU host-side 建图，使用 OpenMP、vector、prefix sum 和 Torch tensor copy；真正的 CUDA kernel 从 `gt::GPUTimer::initialize()` 和 `gt::GPUTimer::levelize()` 开始。

## 1. 文件入口表

| 层级 | 文件 | 关键函数 | 作用 |
|---|---|---|---|
| Python wrapper | `src/core/timing_opt.py:42` | `GPUTimer.__init__` | 准备 lower-left node position、pin offset、调用 pybind |
| pybind factory | `cpp_to_py/gputimer/PyBindCppMain.cpp:29` | `create_gputimer` | 组装 `GTDatabase`、读 SDC、建 graph、创建 C++ timer |
| pybind rawdb | `cpp_to_py/gputimer/PyBindCppMain.cpp:242` | `create_timing_rawdb` lambda | 把 Python tensors 包装成 `TimingTorchRawDB` |
| timing raw db | `cpp_to_py/gputimer/db/GTDatabase.cpp:943` | `TimingTorchRawDB::TimingTorchRawDB` | 生成 node/pin/net flat maps 和当前坐标 tensor |
| graph db | `cpp_to_py/gputimer/db/GTDatabase.cpp:218` | `GTDatabase::GTDatabase` | 持有 rawdb/gpdb/timing_raw_db 引用和 min/max Liberty |
| SDC target map | `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:35` | `preparePinNameMapForSdc` | 只为 SDC 需要的 pin name 建目标集合 |
| timing graph | `cpp_to_py/gputimer/db/GTDatabase.cpp:229` | `ExtractTimingGraph` | host-side 构建 pins/arcs/tests/CSR/tensors |
| SDC values | `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:68` | `readSdc` | 生成 `pinSlew/pinLoad/pinRAT/pinAT` 和 clock/test tensors |
| timer object | `cpp_to_py/gputimer/core/GPUTimer.cpp:14` | `GPUTimer::GPUTimer` | 从 Torch tensors 取裸指针，保存 shared_ptr 生命周期 |
| CUDA init | `cpp_to_py/gputimer/core/GPUTimer.cu:341` | `GPUTimer::initialize` | `cudaMalloc/cudaMemcpy`、Liberty LUT 上传、状态备份 |
| CUDA levelize | `cpp_to_py/gputimer/core/levelize.cu:223` | `GPUTimer::levelize` | GPU Kahn-style topo levelization |

## 2. Python Wrapper

入口：`src/core/timing_opt.py::GPUTimer.__init__`。

关键输入来自 `PlaceData`：

```text
data.node_pos
data.node_size
data.pin_rel_lpos
data.pin_size
data.pin_id2node_id
data.pin_id2net_id
data.node2pin_list / data.node2pin_list_end
data.hyperedge_list / data.hyperedge_list_end
data.net_mask
data.movable_index
data.fixed_connected_index
data.site_width
data.microns
```

关键变换：

```text
node_lpos = data.node_pos - data.node_size / 2
pin_rel_lpos = data.pin_rel_lpos + data.pin_size / 2
conn_node_lpos = movable node_lpos + fixed-connected node_lpos
num_movable_nodes = movable_index[1] - movable_index[0]
scale_factor = 1.0 / data.site_width
```

然后调用：

```text
gputimer.create_timing_rawdb(...)
gputimer.create_gputimer(params, rawdb, gpdb, timing_raw_db)
self.timer.init()
self.timer.levelize()
```

审查点：

- `conn_node_lpos` 只包含 movable + fixed-connected nodes；后续 `TimingTorchRawDB::num_nodes` 来自完整 `node_size`，但 `x/y/init_x/init_y` 来自 `conn_node_lpos`。需要确认所有 GPU RC/timing consumer 对 `num_movable_nodes` 和 connected-node layout 的假设一致。
- `params["num_threads"]` 在 Python 这里兜底为至少 1，后面 C++ 会转给 `timing_raw_db->num_threads`。
- direct RC mode 由 `route_segments` 或 `gr_rc` 决定；后续会跳过 legacy RC tensors。

## 3. create_timing_rawdb

pybind 入口：`cpp_to_py/gputimer/PyBindCppMain.cpp:242`。

它只是把 Python tensors 转成：

```text
std::shared_ptr<gt::TimingTorchRawDB>
```

真正构造在 `cpp_to_py/gputimer/db/GTDatabase.cpp:943`。

`TimingTorchRawDB::TimingTorchRawDB(...)` 做这些事：

1. 清空原始大 tensor 成员：

```text
node_lpos_init = torch::Tensor()
node_size = torch::Tensor()
pin_rel_lpos = torch::Tensor()
```

2. 拆出并 clone contiguous 坐标数组：

```text
node_size_x = node_size[...,0].clone().contiguous()
node_size_y = node_size[...,1].clone().contiguous()
init_x = node_lpos_init[...,0].clone().contiguous()
init_y = node_lpos_init[...,1].clone().contiguous()
pin_offset_x = pin_rel_lpos[...,0].clone().contiguous()
pin_offset_y = pin_rel_lpos[...,1].clone().contiguous()
x = init_x.clone().contiguous()
y = init_y.clone().contiguous()
```

3. 记录规模：

```text
num_nodes = node_size_.size(0)
num_pins = pin_id2node_id_.size(0)
num_nets = hyperedge_list_end_.size(0)
num_movable_nodes = num_movable_nodes_
```

4. 生成 CSR-style flat maps：

```text
flat_node2pin_start_map = cat([0], node2pin_list_end).int32()
flat_node2pin_map = node2pin_list.int32()
pin2node_map = pin_id2node_id.int32()

flat_net2pin_start_map = cat([0], hyperedge_list_end).int32()
flat_net2pin_map = hyperedge_list.int32()
pin2net_map = pin_id2net_id.int32()
net_mask = net_mask.bool()
```

5. 初始化参数：

```text
num_threads = 6          # 构造函数默认值
scale_factor
microns
wire_resistance_per_micron
wire_capacitance_per_micron
```

审查点：

- `num_threads` 构造时固定成 6；真正用户传入的线程数在 `create_gputimer()` 里覆盖。
- `flat_*_start_map` 假设 Python 传入的是 prefix-end，不是 start。
- `from_blob(...).to(device)` 的 device 后续取自 `timing_raw_db.node_size_x.device()`；如果 Python data 没有在 CUDA device 上，后续 timer CUDA 指针会不成立。

## 4. create_gputimer

入口：`cpp_to_py/gputimer/PyBindCppMain.cpp:29`。

实际顺序：

```text
if (!rawdb->liberty_read) throw

if kwargs has num_threads:
  timing_raw_db->num_threads = kwargs["num_threads"]

gtdb = make_shared<GTDatabase>(rawdb, gpdb, timing_raw_db)

direct_rc_mode = kwargs has route_segments or gr_rc
gtdb->skip_legacy_rc_tensors = direct_rc_mode

sdc = make_shared<sdc::SDC>()
if kwargs has sdc:
  sdc->read(kwargs["sdc"])

gtdb->preparePinNameMapForSdc(*sdc)
gtdb->ExtractTimingGraph()
gtdb->readSdc(*sdc)

gputimer = make_shared<gt::GPUTimer>(gtdb, timing_raw_db)

if !direct_rc_mode:
  Flute::readLUT(...)

return gputimer
```

profile labels：

```text
construct_gtdb
read_sdc_json
prepare_pin_name_map_targets
extract_timing_graph
read_sdc_into_gtdb
construct_gputimer
read_flute_lut
```

审查点：

- `readSdc()` 必须在 `ExtractTimingGraph()` 后面，因为它依赖 `primary_inputs/outputs`、`test_id2_arc_id`、`timing_arc_from_pin_id`、`pin_is_clk` 等 graph 结果。
- `gt::GPUTimer` 构造必须在 `readSdc()` 后面，因为 constructor 会对 `timing_raw_db.pinSlew/pinLoad/pinRAT/pinAT` 直接取 `data_ptr<float>()`。
- direct route-segment timing 下不读 Flute LUT，也跳过 legacy RC tensors；如果后续调用旧 `update_rc` 路径，需要检查空指针。

## 5. GTDatabase 基础对象

构造函数：`cpp_to_py/gputimer/db/GTDatabase.cpp:218`。

它只保存引用：

```text
rawdb(*rawdb_)
gpdb(*gpdb_)
timing_raw_db(*timing_raw_db_)
pin_names(gpdb.getPinNames())
net_names(gpdb.getNetNames())
cell_libs_[MIN] = rawdb.cell_libs_[MIN]
cell_libs_[MAX] = rawdb.cell_libs_[MAX]
```

`GTDatabase` 不拥有 rawdb/gpdb/timing_raw_db；生命周期靠 `create_gputimer()` 里的 shared_ptr 和 `GPUTimer::gtdb_holder/timing_raw_db_holder` 保住。

## 6. preparePinNameMapForSdc

文件：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:35`。

目的：避免每次都构建完整 `pin_name2pin_id`。默认只收集 SDC 里后续确实会按 full pin name 查找的对象，目前包括 `set_clock_latency`、`set_case_analysis` 和 `set_propagated_clock`。

逻辑：

```text
pin_name_map_targets.clear()
build_full_pin_name_map = env GPUTIMER_BUILD_FULL_PIN_NAME_MAP
if build_full_pin_name_map:
  return

for command in sdc.commands:
  if SetClockLatency has object_list:
    collect GetPins/GetPorts names
  else if SetCaseAnalysis has port_pin_list:
    collect GetPins/GetPorts names
  else if SetPropagatedClock has object_list:
    collect GetPins/GetPorts names
```

名字变体由 `cpp_to_py/gputimer/db/sdc/SdcUtils.cpp::add_pin_name_target_variants` 生成：

```text
original name
last "/" replaced by ":"
```

这里要区分两个数据结构：

```text
pin_name_map_targets
  临时 set，只用于控制 ExtractTimingGraph() 遍历 gpdb pin_names 时哪些 pin 要进入 map。
  它不是最终查询表。

pin_name2pin_id
  最终 unordered_map<string, pin_id>。
  readSdc() 里的 set_clock_latency / set_propagated_clock / set_case_analysis 等逻辑查它。
```

`pin_name_map_targets` 后面在 `GTDatabase::ExtractTimingGraph()` 里使用：

```text
pin_name2pin_id.clear()

for pin_id, pin_name in gpdb.getPinNames():
  if !full_map and pin_name not in pin_name_map_targets:
    continue

  lookup_key = pin_name_colon_to_slash(pin_name)
  pin_name2pin_id.emplace(lookup_key, pin_id)

pin_name_map_targets.clear()
```

所以默认 lazy map 的闭环是：

```text
SDC object name
  -> add_pin_name_target_variants(...)
  -> pin_name_map_targets
  -> ExtractTimingGraph() 用 target 筛 gpdb 内部 pin_name
  -> pin_name2pin_id 只插入一份 slash lookup key
  -> readSdc() handler 用 SDC 原始 slash pin name 直接查 pin_name2pin_id
```

典型例子：

```text
SDC:
  set_clock_latency ... [get_pins {u_top/u_reg/CK}]

add_pin_name_target_variants:
  targets += "u_top/u_reg/CK"
  targets += "u_top/u_reg:CK"      # last "/" -> ":"

GPDB internal pin_names 里通常是:
  "u_top/u_reg:CK"

ExtractTimingGraph:
  看到 "u_top/u_reg:CK" 命中 targets
  pin_name_colon_to_slash("u_top/u_reg:CK") -> "u_top/u_reg/CK"
  插入 pin_name2pin_id["u_top/u_reg/CK"] = pin_id

readSdc(SetClockLatency):
  用 SDC 原名 "u_top/u_reg/CK" 直接查 pin_name2pin_id
  查到同一个 pin_id
```

为什么需要 `last "/" -> ":"`：

- SDC pin path 通常用 `/` 同时表示层级分隔和最后一级 pin 分隔。
- GPDB/Xplace 内部 pin name 通常把最后一级 instance/pin 分隔写成 `:`，例如 `u_top/u_reg:CK`。
- lazy target 阶段遍历的是内部 `pin_names`。如果 target set 只有 SDC 原名 `u_top/u_reg/CK`，内部名 `u_top/u_reg:CK` 不会命中，就不会被插进最终 map。
- 因此 `last "/" -> ":"` 是标准 SDC slash 名到内部 GPDB colon 名的必要桥接。

为什么 target 阶段不做 `last ":" -> "/"`：

- 当前标准 SDC flow 的输入对象名主要是 slash path，例如 `u_top/u_reg/CK`。
- lazy target 的匹配对象是 GPDB 内部 `pin_names`，关键是把 slash path 转成内部 colon pin name。
- 如果输入本来就是 colon 形式，保留 original name 已经能匹配内部 GPDB pin name；再额外生成 slash target 对当前路径没有必要。
- 最终 `pin_name2pin_id` 不再存内部名和 alias 两份 string。它只存 slash key。`:` 到 `/` 的转换只发生在两个地方：内部 GPDB pin name 写入 map 时，以及 debug endpoint 输入查询前。

`pin_name2pin_id` 后续使用点：

```text
SdcClockConstraints.cpp::_read_sdc(SetClockLatency)
  get_pins pin_name -> pin_name2pin_id.find(pin_name)
  命中后写 pin_clock_latency_overrides[pin_id]

SdcClockConstraints.cpp::_read_sdc(SetPropagatedClock)
  get_pins pin_name -> pin_name2pin_id.find(pin_name)
  命中后写 propagated_clock_pins

SdcExceptions.cpp::_read_sdc(SetCaseAnalysis)
  get_pins pin_name -> pin_name2pin_id.find(pin_name)
  命中后写 pin_case_values[pin_id]

EndpointSlack.cu::debug_dump_endpoint_tests
  endpoint pin name -> pin_name_colon_to_slash(pin_name) -> gtdb.pin_name2pin_id.find(...)
```

与 `primary_input2pin_id` / `primary_output2pin_id` 的分工：

```text
get_ports / all_inputs / all_outputs:
  通常查 primary_input2pin_id 或 primary_output2pin_id。

get_pins:
  查 pin_name2pin_id。
```

因此 target collection 目前收集 `SetClockLatency`、`SetCaseAnalysis` 和 `SetPropagatedClock` 的 get_pins/get_ports 对象，是为了这些 handler 里确实需要按 pin name 解析任意内部 pin；如果新增 `_read_sdc(...)` 也要用 `pin_name2pin_id.find(...)`，就要把对应 SDC command 加到 `preparePinNameMapForSdc()`。

审查点：

- 如果新的 SDC 命令需要按 full pin name 查找，必须同步加入 target collection，否则 lazy `pin_name2pin_id` 可能没有对应项。
- 标准 SDC slash path 到 GPDB colon pin name 的核心 target variant 是 `last "/" -> ":"`。
- `GPUTIMER_BUILD_FULL_PIN_NAME_MAP=1` 是诊断开关，不应作为正常 cold-start 性能答案。

## 7. ExtractTimingGraph 总览

文件：`cpp_to_py/gputimer/db/GTDatabase.cpp:229`。

阶段顺序：

```text
1. 读取 Liberty 单位和 DMP threshold defaults
2. flatten Liberty cell/port/timing/internal/leakage data
3. 构建 pin_name2pin_id
4. 遍历 gpdb pins，识别 PI/PO、cell type、port offset、pin cap、clock pin
5. 构建 net arcs
6. 统计并构建 cell arcs 和 timing tests
7. 构建 pin fanout/fanin CSR
8. 生成 frontier pins
9. 标记 register clock pins
10. endpoint compaction
11. 把 topology/liberty/state vectors materialize 成 Torch CUDA tensors
```

异常对齐点：

- `pin_case_values` 是 `set_case_analysis` 解析出来的 per-pin state。当前 Xplace 只在 host GTDatabase 里保存它，power CUDA input 构造 `PowerActivityState` 时 `case_values` 仍传 `nullptr`。
- OpenROAD/OpenSTA 会用 `set_case_analysis` 做 Sim 常量传播，并通过 disabled conditional arcs 影响 timing/power activity 搜索；Xplace 后续需要补齐这条语义，不能只把它当普通 activity seed。

线程控制：

```text
graph_threads = XPLACE_TIMER_GRAPH_THREADS if set, else timing_raw_db.num_threads
graph_threads = max(1, graph_threads)
```

profile labels 来自 `XPLACE_TIMER_PROFILE`：

```text
thresholds
flatten_liberty
liberty_threshold_vectors
pin_name_map
set_pin_map_and_tag
net_arcs
cell_arcs
release_arc_build_temps
pin_fanout_lists
pin_arc_lists
endpoint_compaction
topology_tensors
liberty_tensors
state_tensors
```

## 8. ExtractTimingGraph: Liberty Flatten

代码范围：`GTDatabase.cpp:278-411`。

输出 host vectors：

```text
liberty_cell_type2port_list_end
liberty_port2timing_list_end
liberty_port2internal_power_list_end
liberty_cell_type2leakage_power_list_end
liberty_port_capacitance
liberty_timing_arcs
liberty_internal_powers
liberty_leakage_powers
liberty_port_function_exprs
liberty_port_has_function
dmp_* threshold arrays
```

关键行为：

- 对每个 `rawdb.celltypes` 查 MIN/MAX Liberty cell。
- 对每个 cell type flatten ports。
- 每个 port flatten：
  - pin capacitance，优先 `port_capacitances_[rf][el]`，缺失时 fallback 到另一个 split 或 `port_capacitance_`。
  - internal power groups。
  - non-conditional non-bundle timing arcs。
  - port function expression，用于 power activity。
- DMP library threshold 按 min/max cell 的 threshold tuple 去重。

审查点：

- MIN/MAX Liberty port 顺序被假设一致：`liberty_cell_view[MIN]->ports_[i]` 和 `liberty_cell_view[MAX]->ports_[i]` 成对使用。
- `SetupThresholdAndFlattenLib` 只负责 threshold/defaults 和 Liberty flatten，不再返回 `LibertyCell* -> DMP library id` map；这个局部 map 在 `SetPinMapAndTag` 内部重建。

## 9. SetPinMapAndTag: Pin Map 和 Pin 属性

函数：`GTDatabase::SetPinMapAndTag(...)`，返回 `primary_input_mask[num_pins]` 给后续 net/cell arc helper 判断 PI。

代码范围：`GTDatabase.cpp:413-590` 附近。`graph_threads` 由 `ExtractTimingGraph()` 统一计算并传入，后续 net/cell arcs 继续复用同一个线程数。

该函数开头先按 `rawdb.celltypes` 重建局部 `dmp_library_id_by_cell`：

```text
for cell_type in rawdb.celltypes:
  min_cell/max_cell = cell_libs_[MIN/MAX]->get_cell(cell_type->name)
  threshold_key = 18 个 min/max threshold/derate float
  identical threshold_key -> same dmp library id
  dmp_library_id_by_cell[min_cell/max_cell] = dmp library id

for timing_arc in liberty_timing_arcs:
  liberty_cell = timing_arc->liberty_port_->cell_
  dmp_timing_library_ids[timing_id] = dmp_library_id_by_cell[liberty_cell] or -1
```

这样 `SetupThresholdAndFlattenLib` 不需要把 cell 指针 map 作为返回值传出来，`SetPinMapAndTag` 内部同时给 timing arcs 和 pins 打 DMP library id。
`dmp_timing_library_ids` 在这里按当前 `liberty_timing_arcs.size()` 生成；后续 `readSdc()->SetDrivingCell` 可能追加 timing arc，见本文第 17 节。

先初始化 per-pin arrays 和 thread-local 输出：

```text
pin_id2cell_type_id[num_pins]
pin_id2port_offset_id[num_pins]
dmp_pin_library_ids[num_pins]
pin_is_clk[num_pins]
pin_is_ideal_clk[num_pins]
pin_case_values[num_pins]  # set_case_analysis；当前 power CUDA 未接入，后续需对齐 OpenROAD Sim/disabled-arc 语义
pin_capacitance[num_pins * 6]
local_name_entries[graph_threads]
local_primary_inputs[graph_threads]
local_primary_outputs[graph_threads]
```

一次 OpenMP 遍历 `gpdb.getPins()`，不要再单独并行扫 `pin_names`：

- 如果 `GPUTIMER_BUILD_FULL_PIN_NAME_MAP=1`，每个 thread 对自己负责的 pin 生成全部 name entries。
- 如果 `pin_name_map_targets` 非空，每个 thread 只对命中 targets 的 pin 生成 name entries。
- name entry 存入 `local_name_entries[tid]`，主线程合并后写 `pin_name2pin_id`。
- 内部 GPDB pin name 写入 map 前先做 `pin_name_colon_to_slash()`，因此最终 map 只存 slash key。

- IO pin：`ori_node_pin_id == -1`，根据 rawdb IO pin direction 标记 PI/PO。
  - 每个 OpenMP thread 写自己的 `local_primary_inputs[tid]` / `local_primary_outputs[tid]`。
- cell pin：
  - `pin_id2cell_type_id[pin_id] = dbcell->ctype()->libcell()`
  - `dmp_pin_library_ids[pin_id] = DMP library id`
  - 用 Liberty cell `ports_map_` 找 port offset。
  - 如果 Liberty port `is_clock_`，先标记 `pin_is_clk[pin_id] = 1`。
  - 从 flattened `liberty_port_capacitance` 写入 `pin_capacitance`。

最后生成：

```text
merge local_name_entries -> pin_name2pin_id
pin_name_map_targets.clear()

merge local_primary_outputs -> primary_outputs
merge local_primary_inputs -> primary_inputs

scan primary_outputs:
  endpoints_id.push_back(pin_id)
  primary_output2pin_id[pin_names[pin_id]] = pin_id

scan primary_inputs:
  primary_input2pin_id[pin_names[pin_id]] = pin_id

primary_input_mask[pin_id] = 1 for primary_inputs

primary_outputs
primary_inputs
primary_output2pin_id
primary_input2pin_id
endpoints_id includes primary_outputs
num_POs = primary_outputs.size()
```

审查点：

- 代码中 rawdb IO direction 到 PI/PO 的映射是当前实现的事实来源，不要按直觉改。
- `pin_name2pin_id` 构建和 pin 属性遍历共用同一个 OpenMP scan，不要再多线程扫两遍 pin 数组。
- PI/PO 不再通过 `num_pins` 长度的临时 byte mask 收集；每个 thread 先收集本地 vector，主线程按 thread 顺序合并，再只扫 PI/PO 列表填 endpoint/name map 和返回的 `primary_input_mask`。
- `pin_name_map_targets.clear()` 在 map 构建后执行；后续 SDC 查找只依赖 `pin_name2pin_id`。
- `pin_case_values` 是当前异常对齐点：`_read_sdc(SetCaseAnalysis)` 会写 host array，但 `PowerActivityState.case_values` 目前是 `nullptr`；OpenROAD 通过 STA Sim 常量传播/disabled conditional arcs 间接影响 power activity，后续需要补齐。
- `cell_node_type_map` 当前 parallel fill 假设 `dbcell->gpdb_id` 对每个 rawdb cell 唯一；如果未来有 alias/merged node，这里会变成共享写。

## 10. ExtractTimingGraph: Net Arcs

代码范围：`GTDatabase.cpp:602-838`。

当前 net arc 规则：

```text
for each net:
  pins = gpdb.getNets()[net_id].pins()
  driver_pin = pins[0]
  for sink in pins[1:]:
    arc from driver_pin to sink
```

`pins[0]` 是 driver 的来源不在 `GTDatabase.cpp`，而在 GPDB net pin ordering：

```text
GPDatabase::addPin(...)
  net.addPin(pin.getId(), pintype->direction() == 'o')

GPNet::addPin(pin_id, is_root)
  pins_id.emplace_back(pin_id)
  if is_root:
    swap(pins_id.front(), pins_id.back())
```

因此 direction 为 `'o'` 的 pin 会被交换到 `pins_id[0]`。`ExtractTimingGraph` 后面直接假设 `gpdb.getNets()[net_id].pins()[0]` 是 driver，`pins()[1:]` 全部是 sinks。

这段先建立两个 net 级别临时数组：

```text
net_arc_start[num_nets + 1]
net_driver_pin[num_nets]
```

`net_arc_start` 在 prefix 前不是 start，而是 count；`net_driver_pin` 保存每个 net 的 driver pin，供后面常量传播 helper 查询 fanin driver。

第一段 OpenMP loop：

```text
parallel for net_id:
  pins = gpdb.getNets()[net_id].pins()
  if pins is empty:
    net_driver_pin[net_id] stays -1
    net_arc_start[net_id] stays 0
  else:
    net_driver_pin[net_id] = pins[0]
    net_arc_start[net_id] = pins.size() - 1
```

这个 loop 是 race-free 的，因为每个 thread 只写自己的 `net_id` slot。`pins.size()==1` 时 count 是 0，表示这个 net 有 driver 但没有 sink arc。

```text
net_arc_start[net_id] = pins.size() - 1
net_driver_pin[net_id] = pins[0]
num_net_arcs = prefix_sum_counts(net_arc_start, graph_threads)
```

`prefix_sum_counts(...)` 是 in-place exclusive prefix sum。输入 `net_arc_start[i]` 临时存每个 net 的 arc count，函数返回后 `net_arc_start[i]` 变成这个 net 在全局 arc array 里的起始 offset，最后一个元素变成总数。例如：

```text
before: [2, 0, 3, 0]
after:  [0, 2, 2, 5]
return: 5
```

当前实现是 CPU OpenMP block scan，不用 CUDA：

```text
if num_threads <= 1 or count_size < 4096:
  serial exclusive scan
else:
  parallel pass 1:
    each thread sums one contiguous block into block_offsets[tid]

  serial pass over block_offsets:
    convert block totals to block base offsets
    check total <= INT_MAX

  parallel pass 2:
    each thread scans its original count block again
    starts[i] = block_base + local_prefix

  starts.back() = total
```

这里的输入语义就是 non-negative count array，所以不做 negative count 检查；只保留总数超过 `INT_MAX` 的 overflow guard，因为后续 arc/test ids 仍是 `int`。

它比 GPU prefix sum 更适合放在这里：`GTDatabase.cpp` 是普通 C++ TU，按项目规则不引 CUDA runtime；而且这个 prefix 后马上要被 CPU 端 OpenMP write pass 使用，放 GPU 还会多一次 host/device 同步。

第二段 OpenMP loop 在 `timing_arc_*` host array 已经按 `num_arcs` 分配后执行：

```text
parallel for net_id:
  pins = gpdb.getNets()[net_id].pins()
  if pins.size() <= 1:
    continue

  driver_pin_id = pins[0]
  arc_id = net_arc_start[net_id]
  for i in 1..pins.size()-1:
    timing_arc_from_pin_id[arc_id] = driver_pin_id
    timing_arc_to_pin_id[arc_id] = pins[i]
    arc_id++
```

每个 net 写 `[net_arc_start[net_id], net_arc_start[net_id + 1])` 这段唯一 slice，所以也没有 shared `push_back`。`arc_types` 初始化为 0，因此 net arc 不需要额外写 type；后面 cell arc 才显式写 `arc_types[arc_id] = 1`。

审查点：

- driver 固定为 `pins[0]`，因此 data loader/gpdb net pin order 是 timing graph 语义的一部分。
- 这个约定依赖每个有效 net 最多一个 output pin、且有 fanout 的 net 至少一个 output pin；否则最后遇到的 output 会被放到 front，或无 output 时原始第一个 pin 会被误当 driver。目前这里没有额外 assert。
- `num_net_arcs` 在所有 cell arcs 前面，保证 arc id 顺序是 net arcs then cell arcs。

## 11. ExtractTimingGraph: Cell Arcs 和 Tests

代码范围：`GTDatabase.cpp:617-883`。

这段在 net driver 信息建立之后，先定义一组局部 lambda。它们服务于一个目的：在构建 cell timing arc 时，利用 SDC case value 和 Liberty output function 判断某些 combinational arc 是否在当前常量条件下不可能激活。

### 11.1 局部 lambda 展开

`is_primary_input_pin(pin_id)`：

```text
pin_id in range && primary_input_mask[pin_id] != 0
```

`primary_input_mask` 来自 `SetPinMapAndTag()` 的返回值。当前这段代码里这个 lambda 定义后没有被调用；如果后续要把 PI 当作特殊 known source，需要在这里接入，否则可以考虑删除。

`pin_has_net_fanin(pin_id)`：

```text
pin_id valid
net_id = gpdb.getPins()[pin_id].getParNetId()
net_id valid
net_driver_pin[net_id] >= 0
net_driver_pin[net_id] != pin_id
```

它判断这个 pin 是否是某个 net 的 sink。这里依赖前面第一段 net loop 已经填好 `net_driver_pin`。

`constant_driver_value(pin_id)`：

```text
pin_id -> gpdb pin -> rawdb cell -> LibertyCell -> LibertyPort
if port has function:
  compile LibertyFuncExpr(port->function_expr_)
  port_values = [-1, -1, ...]   # 所有输入未知
  value = expr.eval(port_values)
  return 0/1 only if eval proves constant
else:
  return -1
```

这个 helper 是常量 driver 判断的核心。它不看 cell 名、pin 名、`:HI` / `:LO` 后缀；只看 driver output port 的 Liberty `function`。如果 function 在所有输入未知时仍能 eval 成 0 或 1，就认为这个 output 是常量 driver。比如 tie cell 的 output function 是 `1` 或 `0` 时可以命中；如果 function 依赖普通输入，则返回 unknown `-1`。

`constant_driven_pin_value(pin_id)`：

```text
if pin_id is not a net sink:
  return -1
driver = net_driver_pin[gpdb.getPins()[pin_id].getParNetId()]
return constant_driver_value(driver)
```

它把 sink pin 的逻辑值追到同一条 net 的 driver output 上，只追一跳，不做递归常量传播。

`known_pin_logic_value(pin_id)`：

```text
if pin_case_values[pin_id] is 0/1:
  return pin_case_values[pin_id]
else:
  return constant_driven_pin_value(pin_id)
```

优先级是 SDC `set_case_analysis` 高于 Liberty tie/function 推断。注意 `pin_case_values` 当前对 power CUDA activity 仍是对齐异常点：host 读到了，但 power state 还没接进去。

`is_functional_combinational_timing(timing_arc)`：

```text
timing_type in {
  combinational,
  combinational_rise,
  combinational_fall
}
```

只有这三类 arc 会进入 Liberty output function 过滤。setup/hold 等 constraint arc 不走这个 function-based pruning。

`output_function_allows_timing_arc(timing_arc, gpdb_id)`：

```text
if not combinational timing:
  return true
if from/to port missing, or to_port has no function:
  return true
if to_port cell missing, from port cannot map, or function compile fails:
  return true

compile output function of to_port
scan expr ops:
  for each referenced port except from_port:
    map Liberty port name -> gpdb pin id in this cell
    known = known_pin_logic_value(pin_id)
    if known is 0/1:
      set port_values[port_id] = known
    else:
      append port_id to unknown_ports

if unknown_ports.size() > 16:
  return true

for each assignment of unknown_ports:
  set from_port = 0
  out0 = eval(output function)
  set from_port = 1
  out1 = eval(output function)
  if timing_sense_transition_possible(sense, out0, out1):
    return true

return false
```

这个 lambda 是实际做 functional pruning 的地方。它只在能够证明 arc 不可能产生对应 timing sense 时返回 `false`；遇到信息不足、function 编译失败、未知输入过多时都保守返回 `true`，保留 arc。

`timing_sense_transition_possible(...)` 的判断是：

```text
positive_unate: out(from=0)=0 and out(from=1)=1
negative_unate: out(from=0)=1 and out(from=1)=0
non_unate/unknown: out(from=0) != out(from=1)
if either output is unknown -1: keep arc
```

原来的 sky130 `__mux2_` / `S -> X` 特判本质是在判断 mux select arc 是否能在当前数据输入常量下激活。例如 `X = S ? A1 : A0`：

- `positive_unate S -> X` 只有在可能出现 `A0=0, A1=1` 时才成立。
- `negative_unate S -> X` 只有在可能出现 `A0=1, A1=0` 时才成立。

现在这条逻辑不再依赖 cell 名或 pin 名，而是直接从 Liberty function 推导。mux 只是其中一个例子；任意 combinational output function 都按同一套枚举判断。

`valid_cell_timing_arc(...)`：

```text
valid_cell_timing_arc(dbcell, gpdb_id, libcell_id, el, timing_id,
                      from_pin_id, to_pin_id, is_test)
```

`is_redundant_timing` 过滤：

- `from_port == to_port`
- `related_port_name_` 为空
- non-sequential setup/hold、clear、preset
- split 不匹配的 min/max constraint

`valid_cell_timing_arc` 做：

- 通过 `gpdb.getNodes()[gpdb_id].getPinbyPortName(...)` 找 from/to physical pin。
- 对 combinational timing arc 调 `output_function_allows_timing_arc(...)`；返回 false 才过滤。
- `is_test = timing_arc->is_constraint() && !is_clock_gating_check(timing_arc)`。

### 11.2 cell_node_type_map loop

```text
cell_node_type_map.assign(gpdb.getNodes().size(), -1)

parallel for cell_idx:
  dbcell = rawdb.cells[cell_idx]
  if dbcell and dbcell->ctype():
    gpdb_id = dbcell->gpdb_id
    if gpdb_id in range:
      cell_node_type_map[gpdb_id] = dbcell->ctype()->libcell()
```

这个 map 是 GPDB node id 到 Liberty cell type id 的反查表。它假设每个 rawdb cell 的 `gpdb_id` 唯一；如果未来出现多个 rawdb cell 写同一个 gpdb node，这里会有共享写语义问题。

### 11.3 cell arc/test count pass

cell arc 也是两阶段。第一阶段只计数，不写最终 arc array：

```text
parallel for cell_idx schedule(dynamic, 256):
  dbcell = rawdb.cells[cell_idx]
  skip invalid cell / missing Liberty / invalid gpdb_id

  arc_count = 0
  test_count = 0
  for el in {MIN, MAX}:
    for pin_id in gpdb.getNodes()[gpdb_id].pins():
      port_id = liberty_cell_type2port_list_end[libcell_id] +
                pin_id2port_offset_id[pin_id]
      timing range = liberty_port2timing_list_end[2 * port_id + el :
                                                  2 * port_id + el + 1]
      for timing_id in timing range:
        if valid_cell_timing_arc(...):
          arc_count++
          if is_test:
            test_count++

  cell_arc_start[cell_idx] = arc_count
  cell_test_start[cell_idx] = test_count
```

这里用 `schedule(dynamic, 256)`，因为不同 cell 的 pin 数和 Liberty timing arc 数差别很大。`pin_id2port_offset_id` 已经在 `SetPinMapAndTag()` 阶段由 gpdb pin macro name 映射到 Liberty port offset，所以这里不再做字符串查找 timing arc，只用 flattened vector 的 offset。

计数结束后把 count array 转成 start array：

```text
num_cell_arcs = prefix_sum_counts(cell_arc_start, graph_threads)
num_tests = prefix_sum_counts(cell_test_start, graph_threads)
num_arcs = num_net_arcs + num_cell_arcs
```

`cell_arc_start[cell_idx]` 之后表示这个 cell 的 cell-arc slice 起点；真正 arc id 还要加上 `num_net_arcs`，因为 net arcs 排在全局 arc array 前半段。

### 11.4 预分配 host arrays

```text
timing_arc_from_pin_id[num_arcs]
timing_arc_to_pin_id[num_arcs]
timing_arc_id_map[num_arcs * 2]   # per arc, per split el
arc_types[num_arcs]
arc_id2test_id[num_arcs]
test_id2_arc_id[num_tests]
```

初始化含义：

- `timing_arc_from_pin_id/to_pin_id`：每条 arc 的 source/sink pin。
- `timing_arc_id_map[arc_id * 2 + el]`：cell arc 对应的 Liberty timing id；net arc 和不存在的 split 保持 `-1`。
- `arc_types`：默认 0 是 net arc；cell arc 写成 1。
- `arc_id2test_id` / `test_id2_arc_id`：constraint test 与 arc 的双向映射。

### 11.5 net arc write pass

net arc 写入前面已经展开过。关键是它只填 `timing_arc_from_pin_id/to_pin_id`，不填 `timing_arc_id_map`，也不写 `arc_types`，因为默认值已经表示 net arc。

### 11.6 cell arc/test write pass

第二阶段用和 count pass 完全相同的遍历顺序和 `valid_cell_timing_arc(...)` 过滤条件，直接写每个 cell 的唯一 slice：

```text
parallel for cell_idx schedule(dynamic, 256):
  local_arc = 0
  local_test = 0
  repeat same el/pin/timing_id traversal as count pass
    if !valid_cell_timing_arc(...):
      continue

    arc_id = num_net_arcs + cell_arc_start[cell_idx] + local_arc
    local_arc++

    timing_arc_from_pin_id[arc_id] = from_pin_id
    timing_arc_to_pin_id[arc_id] = to_pin_id
    timing_arc_id_map[arc_id * 2 + el] = timing_id
    arc_types[arc_id] = 1

    if is_test:
      test_id = cell_test_start[cell_idx] + local_test
      local_test++
      arc_id2test_id[arc_id] = test_id
      test_id2_arc_id[test_id] = arc_id
```

这里的正确性依赖 count pass 和 write pass 过滤结果一致。如果中间任何状态会改变 `valid_cell_timing_arc(...)` 的结果，`local_arc` 就可能和 prefix 得到的 slice 不匹配。目前这些输入都是建图阶段只读结构或前面已经固定的 host vector。

### 11.7 test endpoints

所有 timing tests 的 `to_pin` 追加到 `endpoints_id`：

```text
for test_id in 0..num_tests-1:
  arc_id = test_id2_arc_id[test_id]
  endpoints_id.push_back(timing_arc_to_pin_id[arc_id])
```

PO endpoint 已经在 `SetPinMapAndTag()` 里加入；这里追加的是 setup/hold 等 constraint arc 的 endpoint。

审查点：

- `timing_arc_id_map` 只按 `el` 记录两个 split 的 timing id；不存在的 split 是 `-1`。
- test endpoint 来自 constraint arc 的 `to_pin`，PO endpoint 已经在 pin 遍历阶段加入。
- clock-gating check 仍可成为 cell arc，但不会成为 setup/hold test。
- `is_primary_input_pin` 当前定义后未使用。
- function-based pruning 只做一跳常量 driver 和 `pin_case_values`，不是完整 STA Sim 常量传播；这是后续对齐 OpenROAD 的重点之一。

## 12. ExtractTimingGraph: CSR 和 Frontiers

代码范围：`GTDatabase.cpp:885-956`。

先并行计数：

```text
pin_fanout_count[from_pin]++
pin_num_fanin[to_pin]++
```

这里使用 OpenMP atomic update。

然后串行 prefix：

```text
pin_fanout_list_end[pin_id]
pin_forward_arc_list_end[pin_id]
pin_backward_arc_list_end[pin_id]
total_num_fanouts = sum fanouts
```

再串行 scatter：

```text
for arc_id in 0..num_arcs-1:
  from_pin = timing_arc_from_pin_id[arc_id]
  to_pin = timing_arc_to_pin_id[arc_id]

  pin_forward_arc_list[slot] = arc_id
  pin_fanout_list[slot] = to_pin
  pin_backward_arc_list[slot] = arc_id
```

代码注释明确说明：scatter 保持串行，是为了每个 pin 的 arc order 与 increasing `arc_id` 一致。

frontier：

```text
pin_num_fanin[pin] = backward_end[pin+1] - backward_end[pin]
if pin_num_fanin[pin] == 0:
  pin_frontiers.push_back(pin)
```

审查点：

- `pin_forward_arc_list_end` 和 `pin_fanout_list_end` 共用同一套 offset。
- `pin_num_fanin` 会被后续 `levelize()` 的 CUDA kernel 原地递减；如果需要原始 fanin，不能从 GPU tensor 里读。
- 当前建图的 race-free 关键是 count -> prefix -> direct writes -> serial CSR scatter。不要退回 shared vector `push_back`。

## 13. ExtractTimingGraph: Clock Pins 和 Endpoint Compaction

代码范围：`GTDatabase.cpp:958-990`。

clock pin 标记：

```text
for arc_id in 0..num_arcs-1:
  if arc_id2test_id[arc_id] != -1:
    pin_is_clk[timing_arc_from_pin_id[arc_id]] = 1
```

endpoint compaction：

```text
endpoint_unique_pin_ids = unique endpoints_id in first-seen order
primary_output2_endpoint_id[po_index] = compact_endpoint_id(primary_output_pin)
test_id2_endpoint_id[test_id] = compact_endpoint_id(test endpoint pin)
```

审查点：

- `endpoints_id` 会包含 PO 和 timing test endpoint，可能重复。
- `endpoint_unique_pin_ids` 是后续 endpoint slack/report 的 compact index 基准。

## 14. ExtractTimingGraph: Tensor Materialization

代码范围：`GTDatabase.cpp:992-1051`。

device 来自：

```text
device = timing_raw_db.node_size_x.device()
```

topology tensors：

```text
pin_forward_arc_list
pin_forward_arc_list_end
pin_backward_arc_list
pin_backward_arc_list_end
timing_arc_from_pin_id
timing_arc_to_pin_id
pin_num_fanin
pin_fanout_list
pin_fanout_list_end
```

Liberty/timing tensors：

```text
arc_types
timing_arc_id_map
arc_id2test_id
test_id2_arc_id
endpoints_id
endpoint_unique_pin_ids
test_id2_endpoint_id
primary_output2_endpoint_id
dmp_input_thresholds
dmp_output_thresholds
dmp_slew_lower_thresholds
dmp_slew_upper_thresholds
dmp_slew_derates
dmp_timing_library_ids
dmp_pin_library_ids
dmp_library_* thresholds
```

state tensors：

```text
pinImpulse         # only if !skip_legacy_rc_tensors
pinRootDelay       # only if !skip_legacy_rc_tensors
at_prefix_pin
at_prefix_arc
at_prefix_attr
arcDelay
pinImpulse_ref / pinLoad_ref / ratios  # optional
```

direct route-segment mode：

```text
skip_legacy_rc_tensors = true
skip pinImpulse / pinRootDelay
skip reference/ratio timing tensors
```

memory release before materialization：

```text
primary_input_mask
net_arc_start
net_driver_pin
cell_arc_start
cell_test_start
pin_fanout_count
pin_backward_cursor
```

审查点：

- `torch::from_blob(...).contiguous().to(device)` 必须产生 owned tensor；否则 local vector 生命周期会有风险。
- `arcDelay` shape 是 `[num_arcs, 2 * NUM_ATTR]`，因为 cell arc propagation enumerates 8 lanes。
- legacy RC skip 后，所有 consumer 必须处理 nullable `pinImpulse/pinRootDelay/pinWireCap/pinRootRes`。

## 15. readSdc

主文件：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:68`。

SDC handler 文件：

```text
cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp
  _read_sdc(SetUnits)
  _read_sdc(SetInputDelay)
  _read_sdc(SetInputTransition)
  _read_sdc(SetDrivingCell)
  _read_sdc(SetOutputDelay)
  _read_sdc(SetLoad)
  _read_sdc(SetMaxTransition)

cpp_to_py/gputimer/db/sdc/SdcClockConstraints.cpp
  _read_sdc(CreateClock)
  _read_sdc(SetClockUncertainty)
  _read_sdc(SetClockTransition)
  _read_sdc(SetClockLatency)
  _read_sdc(SetPropagatedClock)
  _read_sdc(SetIdealNetwork)

cpp_to_py/gputimer/db/sdc/SdcExceptions.cpp
  _read_sdc(SetCaseAnalysis)
  _read_sdc(SetFalsePath)
```

`readSdc()` 主流程：

```text
clear driving_cell_sources / clock maps / propagated-clock sets
power_disabled_constraint_arc.resize(num_arcs)
host_pin_slew = NaN
host_pin_load = 0
host_pin_rat = NaN
host_pin_at = NaN

for command in sdc.commands:
  visit _read_sdc(command)

num_timings = liberty_timing_arcs.size()
build clock_periods / clock ids / clock transitions
propagate clock context through nets
apply pin clock latency overrides
map timing tests to capture clocks using test_id2_arc_id
fill missing PI slew with 0
seed clock source AT
mark ideal vs propagated clock pins
copy host_pin_slew/load/rat/at and clock arrays to device tensors
```

输出 tensors：

```text
pinSlew [num_pins, NUM_ATTR]
pinLoad [num_pins, NUM_ATTR]
pinRAT  [num_pins, NUM_ATTR]
pinAT   [num_pins, NUM_ATTR]
clock_periods
pin_clock_ids
test_clock_ids
test_clock_periods
test_setup_uncertainties
test_hold_uncertainties
pin_clock_periods
pin_clock_rise_edges
pin_clock_fall_edges
pin_clock_slews
```

审查点：

- `readSdc()` 会依赖 `test_id2_arc_id` 和 `timing_arc_from_pin_id` 来给 timing tests 找 capture clock。
- `SetOutputDelay` 写 PO RAT；`SetClockUncertainty` 可能回头修正这些 RAT。
- `SetDrivingCell` 可能追加 `liberty_timing_arcs` 并填 `driving_cell_sources`；DMP 后续用这些 source metadata。
- `SetFalsePath` 当前是 debug-only power experiment mask，不是默认 timing path pruning。

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
3. `cpp_to_py/gputimer/db/GTDatabase.cpp:943-991`
   - `TimingTorchRawDB` flat maps 和 connected-node layout。
4. `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:35-66`
   - SDC target pin-name map 是否覆盖当前 SDC 命令。
5. `cpp_to_py/gputimer/db/GTDatabase.cpp:229-939`
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
- graph arc order 是语义：net arcs first，cell arcs second；CSR scatter 保持 increasing arc_id order。
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
