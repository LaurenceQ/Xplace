# Build Timing Graph: ExtractTimingGraph

Last reviewed: 2026-06-23

拆自 `02_BUILD_TIMING_GRAPH.md`。本文覆盖 `preparePinNameMapForSdc` 和 `GTDatabase::ExtractTimingGraph()` 的 host-side canonical graph build。

## 5. GTDatabase 基础对象

构造函数：`cpp_to_py/gputimer/db/GTDatabase.cpp:623`。

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

文件：`cpp_to_py/gputimer/db/GTDatabase.cpp:948`。

阶段顺序：

```text
1. 读取 Liberty 单位和 DMP threshold defaults
2. flatten Liberty cell/port/timing/internal/leakage data
3. 构建 pin_name2pin_id
4. 遍历 gpdb pins，识别 PI/PO、cell type、port offset、pin cap、clock pin
5. 构建 net arcs
6. 统计并构建完整 structural cell arcs 和 timing tests，同时标记 register clock pins
7. 构建 pin forward/backward arc lists 和 fanout lists
8. 生成 frontier pins
9. endpoint compaction
10. 把 topology/liberty/state vectors materialize 成 Torch CUDA tensors
11. 后置 readSdc() 解析 SDC state
12. 后置 RunSdcConstantSimulation() 作为常量传播和 disabled-arc mask 的入口
```

异常对齐点：

- `pin_case_values` 是 `readSdc()->_read_sdc(SetCaseAnalysis)` 解析出来的 per-pin state。它不再参与 graph build 阶段的 cell arc pruning。
- OpenROAD/OpenSTA 会用 `set_case_analysis` 做 Sim 常量传播，并通过 disabled conditional arcs 影响 timing search。当前 Xplace 只保留 `RunSdcConstantSimulation()` 入口，后续需要在这里生成 `pin_logic_values` 和 timing-only disabled arc mask。
- power CUDA activity 已有 case/logic value seed 形状，但当前 host input 仍未把 `pin_logic_values` 接进去；这和 timing disabled arc mask 应该分开审查。

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
pin_case_values[num_pins]  # set_case_analysis；后置 readSdc() 填充，power CUDA 仍未接入
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
- `SetPinMapAndTag()` 结束后不读 SDC；`set_case_analysis` 只在后置 `readSdc()` 中写入 `pin_case_values`。
- `cell_node_type_map` 当前在 cell arc collect 中填充，仍假设 `dbcell->gpdb_id` 对每个 rawdb cell 唯一；如果未来有 alias/merged node，这里会变成共享写。
- timing graph 里已有的 thread-local 写法是固定区间分配：

```text
tid = omp_get_thread_num()
start = (count * tid) / graph_threads
end = (count * (tid + 1)) / graph_threads
```

命名统一沿用 `SetPinMapAndTag()` 风格：parallel block 外的二维收集容器叫 `local_*[graph_threads]`，block 内绑定到当前 thread 的引用叫 `thread_* = local_*[tid]`。不要把这种固定 per-thread range 叫 `chunk_*`；`chunk_*` 只用于真正按容量/行数切块的 power row chunking 路径。

## 10. ExtractTimingGraph: Net Arcs

代码范围：`GTDatabase.cpp:255-448` 和 `GTDatabase.cpp:948-999`。

net/cell arc 的前半段由 `GTDatabase` 成员函数统一处理：

```text
GTDatabase::BuildNetCellArcAndTest(...)
  output net_arc_start
  output local_cell_arc_entries
  output thread_cell_arc_start
  output thread_cell_test_start
  return {num_arcs, num_tests}
```

这个函数不写最终 `timing_arc_*` arrays；它只完成 count/prefix、cell entry collect、以及 pin forward/backward count。后续写最终 arrays 的边界是：

```text
GTDatabase::AllocatePinArcListStorage(...)
GTDatabase::WriteNetArcList(...)
GTDatabase::WriteCellArcListAndTest(...)
GTDatabase::AppendTestEndpoints()
GTDatabase::BuildPinFrontiers(...)
GTDatabase::CountRegisterClockPins(...)
GTDatabase::CompactEndpointPins()
```

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

net arc 先做 per-net count/prefix，避免保存 `local_net_arc_entries`：

```text
net_arc_start[num_nets + 1]

parallel for net_id:
  pins = gpdb.getNets()[net_id].pins()
  if pins.size() > 1:
    sink_count = pins.size() - 1
    net_arc_start[net_id] = sink_count
    pin_forward_arc_list_end[pins[0]] += sink_count
    for sink_pin_id in pins[1:]:
      pin_backward_arc_list_end[sink_pin_id]++

num_net_arcs = prefix_sum_counts(net_arc_start, graph_threads)
```

`net_arc_start[net_id]` 在 prefix 前是这个 net 的 sink count，prefix 后是这个 net 在全局 arc array 里的起始 `arc_id`。`pin_forward_arc_list_end/pin_backward_arc_list_end` 此时还没有 prefix，临时作为 per-pin forward/backward arc count array 使用。

`AllocatePinArcListStorage()` 分配 `timing_arc_*` host array 和 pin arc/fanout list 后，`WriteNetArcList()` 直接从 GPDB net pins 写最终数组，同时把 net arc 写进 forward/backward arc list：

```text
parallel for net_id:
  pins = gpdb.getNets()[net_id].pins()
  if pins.size() <= 1:
    continue

  driver_pin_id = pins[0]
  arc_id = net_arc_start[net_id]

  for sink_pin_id in pins[1:]:
    timing_arc_from_pin_id[arc_id] = driver_pin_id
    timing_arc_to_pin_id[arc_id] = sink_pin_id
    pos = atomic_capture(pin_forward_arc_cursor[driver_pin_id]++)
    pin_forward_arc_list[pos] = arc_id
    pin_fanout_list[pos] = sink_pin_id
    pos = atomic_capture(pin_backward_arc_cursor[sink_pin_id]++)
    pin_backward_arc_list[pos] = arc_id
    arc_id++
```

`arc_types` 初始化为 0，因此 net arc 不需要额外写 type；后面 cell arc 才显式写 `arc_types[arc_id] = 1`。

审查点：

- driver 固定为 `pins[0]`，因此 data loader/gpdb net pin order 是 timing graph 语义的一部分。
- 这个约定依赖每个有效 net 最多一个 output pin、且有 fanout 的 net 至少一个 output pin；否则最后遇到的 output 会被放到 front，或无 output 时原始第一个 pin 会被误当 driver。目前这里没有额外 assert。
- `num_net_arcs` 在所有 cell arcs 前面，保证 arc id 顺序是 net arcs then cell arcs。

## 11. ExtractTimingGraph: Cell Arcs 和 Tests

代码范围：`GTDatabase.cpp:255-508` 和 `GTDatabase.cpp:1001-1012`。

`BuildNetCellArcAndTest()` 在 net count/prefix 之后构建完整 structural cell timing arcs。当前 `ExtractTimingGraph()` 不读 SDC，也不根据 case analysis 或 Liberty function 常量条件删 arc；后续常量传播和 timing-disabled arc mask 应放到 `RunSdcConstantSimulation()`。

### 11.1 局部 lambda 展开

`valid_cell_timing_arc(...)`：

```text
valid_cell_timing_arc(gpdb_id, el, timing_id,
                      from_pin_id, to_pin_id, is_test)
```

`is_redundant_timing` 过滤：

- `from_port == to_port`
- `related_port_name_` 为空
- non-sequential setup/hold、clear、preset
- split 不匹配的 min/max constraint

`valid_cell_timing_arc` 做：

- 通过 `gpdb.getNodes()[gpdb_id].getPinbyPortName(...)` 找 from/to physical pin。
- `is_test = timing_arc->is_constraint() && !is_clock_gating_check(timing_arc)`。
- 不检查 `pin_case_values`，不编译 output function，也不做 mux/select 常量条件过滤。
- `dbcell` 和 `libcell_id` 不传进这个 lambda；它们只在外层 cell 合法性检查和 flattened Liberty timing range 计算里使用。

旧版建图期 constant 判定逻辑已经从这里拿走。后续如果实现 simulation，应放进 `RunSdcConstantSimulation()`，并形成两个输出：

```text
pin_logic_values[pin_id]              # -1 unknown, 0/1 propagated constant
timing_arc_disabled_by_constant[arc]  # timing-only disabled mask
```

旧逻辑可迁移的判断单元：

```text
constant_driver_value(pin_id):
  由 output Liberty function 判断 tie/constant output，不看 pin name :HI/:LO

known_pin_logic_value(pin_id):
  先看 readSdc() 填出的 pin_case_values，再看常量传播结果

output_function_allows_timing_arc(...):
  在已知 side inputs 下枚举未知 input，判断 from_port 0->1 是否可能产生 arc timing_sense
```

这些判断不应该再决定 structural graph 是否建 arc；它们应该只生成后置 disabled mask，供 timing traversal/kernel 使用。power 如需利用常量，应复用 `pin_logic_values` 作为 activity seed，而不是直接复用 timing-disabled arc mask。

### 11.2 cell arc/test thread-local collect

cell arc 不再走 per-cell count pass + per-cell prefix + second write pass。当前实现用固定 thread range 一次遍历 cell timing arcs，并把有效 arc 存到 thread-local entries：

```text
CellArcBuildEntry:
  from_pin_id
  to_pin_id
  timing_id
  el
  is_test

local_cell_arc_entries[graph_threads]
local_cell_test_counts[graph_threads]

parallel num_threads(graph_threads):
  tid = omp_get_thread_num()
  start = (num_cells * tid) / graph_threads
  end = (num_cells * (tid + 1)) / graph_threads
  thread_cell_arc_entries = local_cell_arc_entries[tid]

  for cell_idx in [start, end):
    dbcell = rawdb.cells[cell_idx]
    skip null dbcell / missing ctype
    gpdb_id = dbcell->gpdb_id
    libcell_id = dbcell->ctype()->libcell()
    if gpdb_id in range:
      cell_node_type_map[gpdb_id] = libcell_id
    skip invalid libcell / missing Liberty / invalid gpdb_id

    for el in {MIN, MAX}:
      for pin_id in gpdb.getNodes()[gpdb_id].pins():
        port_id = liberty_cell_type2port_list_end[libcell_id] +
                  pin_id2port_offset_id[pin_id]
        timing range = liberty_port2timing_list_end[2 * port_id + el :
                                                    2 * port_id + el + 1]
        for timing_id in timing range:
          if valid_cell_timing_arc(...):
            thread_cell_arc_entries.push_back(...)
            pin_forward_arc_list_end[from_pin]++
            pin_backward_arc_list_end[to_pin]++
            if is_test:
              thread_cell_test_count++

  local_cell_test_counts[tid] = thread_cell_test_count
```

`pin_id2port_offset_id` 已经在 `SetPinMapAndTag()` 阶段由 gpdb pin macro name 映射到 Liberty port offset，所以这里不再做字符串查找 timing arc，只用 flattened vector 的 offset。

`cell_node_type_map` 是 GPDB node id 到 Liberty cell type id 的反查表；它现在和 cell arc collect 共用同一个 cell scan。它仍假设每个 rawdb cell 的 `gpdb_id` 唯一；如果未来有 alias/merged node，这里会变成共享写语义问题。

收集后只对 thread 数量级的 count 做 prefix：

```text
thread_cell_arc_start[tid] = local_cell_arc_entries[tid].size()
thread_cell_test_start[tid] = local_cell_test_counts[tid]

num_cell_arcs = prefix_sum_counts(thread_cell_arc_start, graph_threads)
num_tests = prefix_sum_counts(thread_cell_test_start, graph_threads)
num_arcs = num_net_arcs + num_cell_arcs
```

`thread_cell_arc_start[tid]` 之后表示这个 thread 的 cell-arc slice 起点；真正 arc id 还要加上 `num_net_arcs`，因为 net arcs 排在全局 arc array 前半段。

顺序语义：

- 每个 thread 负责一个连续 `cell_idx` range。
- 主体 arc order 是 `tid` 递增、每个 `tid` 内 `cell_idx` 递增。
- 因为 range 是连续切分，最终 cell arc 顺序仍等价于全局 `cell_idx` 递增。
- 这和 `SetPinMapAndTag()` 的 `local_*[tid]` 收集/合并命名保持一致，不引入 `chunk_*`。

速度收益预期：

- 少一次完整 cell timing arc 遍历。旧版 count pass 和 write pass 都要执行 `valid_cell_timing_arc(...)`、`getPinbyPortName(...)`、Liberty timing range traversal；新版只做一次，然后 copy compact entries。
- per-cell `cell_arc_start/cell_test_start` 两个 `num_cells + 1` prefix 被替换为 `graph_threads + 1` prefix，prefix 成本基本消失。
- 去掉 `schedule(dynamic, 256)` 的调度开销，改成和其他 thread-local 段一致的固定 range。
- 新增成本是 thread-local `CellArcBuildEntry` 临时存储和一次 compact entry copy。entry 当前是 16B 级别，峰值内存会比旧版高；如果 cell arc 数非常大，需要把这项纳入 RSS 审查。
- 固定 range 的风险是 load balance 不如 dynamic schedule。如果少数超大 cell 集中在某个 range，收益会被尾部线程抵消；ISPD 这类标准 cell 大量重复时，通常更可能由少一次 arc traversal 获益。

### 11.3 AllocatePinArcListStorage

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
- `pin_forward_arc_list_end` / `pin_backward_arc_list_end`：输入时是 per-pin count，函数内调用 `prefix_sum_counts(...)` 后变成每个 pin 的 flat-list 起始 offset。
- `pin_fanout_list_end = pin_forward_arc_list_end`，因为 fanout pin list 和 forward arc list 共享同一套 bucket offset。
- `pin_forward_arc_cursor` / `pin_backward_arc_cursor` 是 offset 的临时副本，供 `WriteNetArcList()` 和 `WriteCellArcListAndTest()` 用 atomic capture 分配写入位置。

### 11.4 WriteNetArcList

net arc 写入前面已经展开过。关键是它只填 `timing_arc_from_pin_id/to_pin_id`，不填 `timing_arc_id_map`，也不写 `arc_types`，因为默认值已经表示 net arc。它同时写：

```text
pin_forward_arc_list[pos] = arc_id
pin_fanout_list[pos] = sink_pin_id
pin_backward_arc_list[pos] = arc_id
```

`net_arc_start` 在这个函数结束后释放。

### 11.5 WriteCellArcListAndTest

最终 cell arc 写入不再重复遍历 rawdb/gpdb/Liberty，只遍历 compact thread-local entries：

```text
parallel num_threads(graph_threads):
  tid = omp_get_thread_num()
  thread_cell_arc_entries = local_cell_arc_entries[tid]
  arc_base = num_net_arcs + thread_cell_arc_start[tid]
  test_base = thread_cell_test_start[tid]
  thread_test_offset = 0

  for local_arc in thread_cell_arc_entries:
    entry = thread_cell_arc_entries[local_arc]
    arc_id = arc_base + local_arc

    timing_arc_from_pin_id[arc_id] = entry.from_pin_id
    timing_arc_to_pin_id[arc_id] = entry.to_pin_id
    timing_arc_id_map[arc_id * 2 + entry.el] = entry.timing_id
    arc_types[arc_id] = 1

    if entry.is_test:
      test_id = test_base + thread_test_offset
      thread_test_offset++
      arc_id2test_id[arc_id] = test_id
      test_id2_arc_id[test_id] = arc_id
      pin_is_clk[entry.from_pin_id] = 1
```

这里的正确性不再依赖 count pass 和 write pass 重新执行同一套过滤逻辑，因此少了一个潜在一致性风险。需要审查的是 thread-local entry 收集顺序是否稳定，以及 `thread_cell_arc_start/thread_cell_test_start` prefix 是否按 `tid` 顺序生成唯一 slice。

`pin_is_clk` 的 register clock pin 标记也在这里完成：只有 `entry.is_test` 的 setup/hold 等 constraint arc 会把 `from_pin_id` 标成 clock pin。这样不再需要后面额外扫描 `num_arcs` 查 `arc_id2test_id`。

`local_cell_arc_entries`、`thread_cell_arc_start`、`thread_cell_test_start`、`pin_forward_arc_cursor`、`pin_backward_arc_cursor` 在这个函数结束后释放。

### 11.6 AppendTestEndpoints

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
- cell arc 构建当前保留完整 structural arcs；SDC case analysis 和 Liberty constant function 不在这里剪枝。
- timing disabled arc 后续应由 `RunSdcConstantSimulation()` 生成 mask，再决定是否重建 traversal 或在 CUDA kernel 中跳过。

## 12. ExtractTimingGraph: Pin Arc Lists 和 Frontiers

代码范围：`GTDatabase.cpp:382-448`、`GTDatabase.cpp:449-508`、`GTDatabase.cpp:519-539`。

`pin_forward_arc_list_end` 和 `pin_backward_arc_list_end` 在 net count pass 前先清零。它们先临时作为 per-pin count array 使用：

```text
net count pass:
  pin_forward_arc_list_end[driver_pin] += sink_count
  pin_backward_arc_list_end[sink_pin]++

cell collect pass:
  pin_forward_arc_list_end[from_pin]++
  pin_backward_arc_list_end[to_pin]++
```

这样不需要再额外扫一遍 `num_arcs` 来统计 fanout/fanin。cell count 和 `CellArcBuildEntry` collect 在同一轮 Liberty traversal 里完成。

然后 `AllocatePinArcListStorage()` 把 count array 原地 prefix 成每个 pin 的 flat-list offset：

```text
fanout_total = prefix_sum_counts(pin_forward_arc_list_end)
fanin_total = prefix_sum_counts(pin_backward_arc_list_end)
pin_fanout_list_end = pin_forward_arc_list_end
```

含义：

- `pin_forward_arc_list_end[pin]`：这个 pin 作为 `from_pin` 的 outgoing arc slice 起点。
- `pin_fanout_list_end[pin]`：这个 pin 的 fanout pin slice 起点，和 forward arc 共用 offset。
- `pin_backward_arc_list_end[pin]`：这个 pin 作为 `to_pin` 的 incoming arc slice 起点。

scatter 用 prefix 后的 offset 拷一份 cursor。当前 timing/power 不依赖同一个 pin bucket 内 fanout arc 顺序，所以 `WriteNetArcList()` / `WriteCellArcListAndTest()` 用 atomic capture 分配 slot，直接把当前生成的 `arc_id` 写进 pin arc list：

```text
pin_forward_arc_cursor = pin_forward_arc_list_end
pin_backward_arc_cursor = pin_backward_arc_list_end

net final write pass:
  timing_arc_from_pin_id[arc_id] = driver_pin_id
  timing_arc_to_pin_id[arc_id] = sink_pin_id
  pos = atomic_capture(pin_forward_arc_cursor[driver_pin_id]++)
  pin_forward_arc_list[pos] = arc_id
  pin_fanout_list[pos] = sink_pin_id
  pos = atomic_capture(pin_backward_arc_cursor[sink_pin_id]++)
  pin_backward_arc_list[pos] = arc_id

cell final write pass:
  timing_arc_from_pin_id[arc_id] = entry.from_pin_id
  timing_arc_to_pin_id[arc_id] = entry.to_pin_id
  pos = atomic_capture(pin_forward_arc_cursor[entry.from_pin_id]++)
  pin_forward_arc_list[pos] = arc_id
  pin_fanout_list[pos] = entry.to_pin_id
  pos = atomic_capture(pin_backward_arc_cursor[entry.to_pin_id]++)
  pin_backward_arc_list[pos] = arc_id
```

`pin_backward_arc_list` 就是在这里建出来的：它按 `to_pin` 分桶，每个桶里存所有指向这个 pin 的 `arc_id`。由于用 atomic capture 并行 scatter，同一个 pin bucket 内的 arc 顺序不再保证等价于 increasing `arc_id`；当前 timing/power consumer 只遍历 bucket，不依赖这个顺序。

frontier：

```text
pin_num_fanin[pin] = backward_end[pin+1] - backward_end[pin]
if pin_num_fanin[pin] == 0:
  pin_frontiers.push_back(pin)
```

审查点：

- `pin_forward_arc_list_end` 和 `pin_fanout_list_end` 共用同一套 offset。
- `pin_backward_arc_list_end` 是 backward arc list offset；`pin_backward_arc_list` 只存 arc id，真正的 source pin 通过 `timing_arc_from_pin_id[arc_id]` 查。
- `pin_num_fanin` 会被后续 `levelize()` 的 CUDA kernel 原地递减；如果需要原始 fanin，不能从 GPU tensor 里读。
- 当前建图的 race-free 关键是 atomic count -> prefix -> atomic-capture scatter。不要退回 shared vector `push_back`。
- 这个版本只需要两个 `num_pins + 1` cursor 临时数组，不再开 `2 * graph_threads * num_pins` 的二维 count。

## 13. ExtractTimingGraph: Clock Pins 和 Endpoint Compaction

代码范围：`GTDatabase.cpp:509-569` 和 `GTDatabase.cpp:1014-1021`。

clock pin 标记已经在 cell arc/test copy pass 里完成：

```text
if entry.is_test:
  pin_is_clk[entry.from_pin_id] = 1
```

这里不再额外扫描 `num_arcs`。后面只并行统计最终 clock pin 数量：

```text
parallel reduction sum over pin_id:
  num_clk_pins += pin_is_clk[pin_id] != 0
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
net_arc_start
local_cell_arc_entries
thread_cell_arc_start
thread_cell_test_start
pin_forward_arc_cursor
pin_backward_arc_cursor
```

审查点：

- `torch::from_blob(...).contiguous().to(device)` 必须产生 owned tensor；否则 local vector 生命周期会有风险。
- `arcDelay` shape 是 `[num_arcs, 2 * NUM_ATTR]`，因为 cell arc propagation enumerates 8 lanes。
- legacy RC skip 后，所有 consumer 必须处理 nullable `pinImpulse/pinRootDelay/pinWireCap/pinRootRes`。
