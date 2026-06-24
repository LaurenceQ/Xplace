# Build Timing Graph: SDC

Last reviewed: 2026-06-23

拆自 `02_BUILD_TIMING_GRAPH.md`。本文覆盖 SDC Tcl/JSON bridge、`readSdc()` visitor、sparse clock tables、exception handling 和 constant simulation placeholder。

## 15. readSdc

主文件：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:68`。

### 15.0 SDC 文件到 JSON 的转换链路

当前只要 Python/pybind kwargs 里带 `sdc`，读入路径就是先把 SDC 转成 JSON，再由 C++ 读 JSON 构造 `sdc::SDC::commands`：

```text
PyBindCppMain.cpp::create_gputimer(...)
  if kwargs contains "sdc":
    sdc->read(kwargs["sdc"].cast<std::string>())

cpp_to_py/common/lib/sdc/sdc.cpp::SDC::read(path)
  sdc_path = absolute(path)
  sdc_json = sdc_path with extension replaced by ".json"
  fork()
    child:
      chdir(cwd/cpp_to_py/common/lib/sdc)
      execvp("/usr/bin/tclsh", {"tclsh", "sdc.tcl", sdc_path, sdc_json})
    parent:
      waitpid(child)
      ifstream(sdc_json)
      Json json
      ifs >> json
      for each json command object:
        switch j["command"]
        commands.emplace_back(std::in_place_type_t<...>{}, j)
      remove(sdc_json)
```

Tcl 侧：

```text
cpp_to_py/common/lib/sdc/sdc.tcl
  source sdcparsercore.tcl
  lappend auto_path json
  package require json
  package require json::write
  sdc::register_callback gt::sdc_callback
  sdc::parse_file input.sdc
  write json::write array to output.json

cpp_to_py/common/lib/sdc/sdcparsercore.tcl
  creates slave Tcl interpreters for SDC versions 1.1..2.0
  sources sdc2.0.tcl, sdc1.9.tcl, ...
  sdc::parse_file reads complete Tcl statements
  slave interpreter evaluates each SDC command
  sdc::parse_command validates declared args and calls callback
```

JSON callback 语义：

```text
create_clock / set_*:
  callback gets parsed param array
  emits one JSON object:
    {"command": "...", "-option": "value", "positional": "value", ...}

get_pins:
  returns "__get_pins__ <patterns>"

get_ports:
  returns "<patterns>"

get_clocks:
  currently also returns "<patterns>"

all_inputs / all_outputs / all_clocks:
  returns marker string
```

C++ JSON 到 typed object：

```text
cpp_to_py/common/lib/sdc/sdc.cpp
  SetInputDelay(const Json&)
  SetDrivingCell(const Json&)
  ...

cpp_to_py/common/lib/sdc/object.cpp::parse_port(line)
  "all_inputs"  -> AllInputs
  "all_outputs" -> AllOutputs
  "all_clocks"  -> AllClocks
  "__get_pins__" -> GetPins
  "__get_clocks__" -> GetClocks
  otherwise -> GetPorts
```

审查结论：

- 是的，当前 SDC 读入都先走 Tcl parser -> 临时 JSON 文件 -> nlohmann/json DOM -> C++ typed variant。
- 这不是纯内存转换；它 fork 子进程、启动 `/usr/bin/tclsh`、source 多个 Tcl 文件、写 `<sdc>.json` 到 SDC 文件同目录、父进程再读回并删除。
- 对 ISPD 这种 SDC command 数通常不大的场景，单次 JSON DOM 构造本身未必是主耗时；更明显的固定开销是 fork/exec Tcl、Tcl package/source、以及一次文件写读。如果目标是极限 cold-start，这一段需要 profile 量化。

性能和可靠性风险点：

- 临时文件名是 `sdc_path.replace_extension(".json")`。如果同目录已有同名 JSON，会被覆盖并最终删除。
- 两个进程同时读同一个 SDC 会争用同一个 `<sdc>.json`，存在互相覆盖/删除的 race。
- JSON 写在 SDC 所在目录；如果 benchmark 目录只读，或者 NFS 写小文件很慢，会直接影响 `read_sdc_json` 阶段。
- child `chdir` 用的是 `std::filesystem::current_path() / "cpp_to_py/common/lib/sdc"`，隐含调用进程 cwd 是 Xplace repo root；如果从别的 cwd 调 pybind，Tcl parser 路径会失效。
- `get_clocks` 在 Tcl callback 里没有加 `__get_clocks__` 前缀，C++ `parse_port()` 会把它当 `GetPorts`。所以多个 handler 里都有 “GetPorts-as-clock-token” 兼容逻辑；`object.cpp` 里的 `__get_clocks__` 分支当前不是主路径。
- `logger.infoif(!exists)` / `logger.infoif(!ifs)` 看起来更像记录错误而非硬失败；如果 Tcl child 失败或 JSON 没生成，父进程仍可能继续进入 JSON parse/空 commands 路径，需要按 logger 行为再确认是否会 abort。

可以考虑的优化方向：

- 低风险：把临时 JSON 改到唯一临时路径，例如 `/tmp/xplace_sdc_<pid>_<hash>.json`，避免覆盖和并发 race。
- 中风险：让 Tcl 输出 JSON 到 stdout，父进程用 pipe 读，去掉磁盘临时文件。
- 高收益但改动大：直接在 C++ 里实现当前支持子集的 SDC parser，或者嵌入 Tcl interpreter 后用内存 callback 传结构，避免每次 fork/exec。

核心分发点在 `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:76` 的 `GTDatabase::readSdc(...)`：

```cpp
for (auto& command : sdc.commands) {
    std::visit(Functors{[this](auto&& cmd) { _read_sdc(cmd); }}, command);
}
```

这里 `sdc.commands` 是 SDC parser 产出的 command variant 列表；`std::visit` 根据 variant 的真实类型在编译期分发到对应 overload。overload 声明在 `cpp_to_py/gputimer/db/GTDatabase.h`，定义按 SDC 语义拆到下面三个 source file：

SDC handler 文件：

```text
cpp_to_py/gputimer/db/sdc/SdcTimingConstraints.cpp  # pin IO timing / load / units
  _read_sdc(SetUnits)
  _read_sdc(SetInputDelay)
  _read_sdc(SetInputTransition)
  _read_sdc(SetDrivingCell)
  _read_sdc(SetOutputDelay)
  _read_sdc(SetLoad)
  _read_sdc(SetMaxTransition)

cpp_to_py/gputimer/db/sdc/SdcClockConstraints.cpp   # clocks and clock network state
  _read_sdc(CreateClock)
  _read_sdc(SetClockUncertainty)
  _read_sdc(SetClockTransition)
  _read_sdc(SetClockLatency)
  _read_sdc(SetPropagatedClock)
  _read_sdc(SetIdealNetwork)

cpp_to_py/gputimer/db/sdc/SdcExceptions.cpp         # case analysis / exceptions
  _read_sdc(SetCaseAnalysis)
  _read_sdc(SetFalsePath)
```

审查这个 visitor 时要按 source file 分开看：

- `GTDatabase_sdc.cpp`：只负责清空/初始化 SDC state、遍历 `sdc.commands`、以及所有 command 读完后的 sparse clock id/table 汇总。
- `SdcTimingConstraints.cpp`：写 `host_pin_at/rat/slew/load`、`pin_capacitance`、`driving_cell_sources`、SDC unit override。
- `SdcClockConstraints.cpp`：写 `clocks`、clock uncertainty/transition/latency、`propagated_clock_*`、`pin_clock_latency_overrides`、ideal/propagated clock policy 输入。
- `SdcExceptions.cpp`：写 `pin_case_values`；`SetFalsePath` 当前只生成 debug-only `power_disabled_constraint_arc` mask，不改变 canonical timing graph。

### 15.1 SdcTimingConstraints.cpp

`_read_sdc(SetUnits)`：

```text
input:
  obj.time / obj.capacitance / obj.resistance
write:
  sdc_time_unit / sdc_cap_unit / sdc_res_unit
effect:
  后续 SDC delay/slew/load 读入时按 sdc unit / liberty unit 转换到 GTDatabase 内部单位。
```

`_read_sdc(SetInputDelay)`：

```text
requires:
  obj.delay_value
  obj.port_pin_list = AllInputs or GetPorts
lookup:
  optional obj.clock -> clocks[clock].rise_edge/fall_edge
write:
  hostPinAT(pi, attr)
formula:
  input arrival = selected clock edge + delay
mask:
  min/max/rise/fall 通过 sdc::TimingMask 选择 NUM_ATTR lane
```

审查点：`GetPorts` 只查 `primary_input2pin_id`；找不到 port 只 warning，不会建新 pin。clock 缺失时会 warning，但当前函数仍用默认 `clock_edge = 0.0f` 继续计算。

`_read_sdc(SetInputTransition)`：

```text
requires:
  obj.transition
  obj.port_list = AllInputs or GetPorts
write:
  hostPinSlew(pi, attr)
unit:
  transition * sdc_time_unit / time_unit
```

审查点：这里只设置 PI slew，不写 `pin_capacitance`，也不构造 driving-cell arc metadata。

`_read_sdc(SetDrivingCell)`：

```text
requires:
  obj.transitions[RISE/FALL]
  obj.port_list = AllInputs or GetPorts
write:
  hostPinSlew(pi, attr)
  driving_cell_sources
may append:
  liberty_timing_arcs
lookup:
  cell_libs_[el]->get_cell(obj.lib_cell)
  LibertyCell::get_port(obj.pin)
  output_port->timing_arcs_non_cond_non_bundle_
```

内部逻辑：

```text
transition_for_rf(rf):
  use obj.transitions[rf], fallback to opposite rf, unit-convert

record_driving_cell_source(pin_id):
  for each selected el/rf:
    find LibertyCell by obj.lib_cell
    find output LibertyPort by obj.pin
    choose first non-redundant non-constraint TimingArc whose transition table exists
    timing_id_for_arc(timing_arc):
      return existing liberty_timing_arcs index, or append timing_arc
    write DrivingCellSource lane:
      timing_ids[attr]
      input_rfs[attr]
      input_slews[attr]
```

审查点：`set_driving_cell` 会影响后续 DMP source slew/model metadata；它不是普通 `set_input_transition` 的简单别名。若 `lib_cell/pin` 缺失，只留下直接 PI slew。

`_read_sdc(SetOutputDelay)`：

```text
requires:
  obj.delay_value
  obj.port_pin_list = AllOutputs or GetPorts
  obj.clock exists in clocks
write:
  hostPinRAT(po, attr)
  output_delay_clock_by_pin_attr[po][attr]
formula:
  MIN/hold lane: clock_edge - delay + hold_uncertainty
  MAX/setup lane: clock_edge + period - delay - setup_uncertainty
```

审查点：clock 不存在时直接 return；所以 SDC command 顺序会影响这一项，`create_clock` 必须先读到。`SetClockUncertainty` 后面可能根据 `output_delay_clock_by_pin_attr` 回头修正已经写过的 RAT。

`_read_sdc(SetLoad)`：

```text
requires:
  obj.value
  obj.objects = AllOutputs or GetPorts
write:
  hostPinLoad(po, attr)
  pin_capacitance[6 * po + el * 2 + rf]
  pin_capacitance[6 * po + 4 + el]
unit:
  load * sdc_cap_unit / cap_unit
```

审查点：这里把 SDC output load 同时写进 timing host load 和 pin capacitance vector；后续 graph/tensor materialization 会把这个 capacitance 带到 device。

`_read_sdc(SetMaxTransition)`：

```text
effect:
  no-op
reason:
  set_max_transition 是 design-rule constraint，不直接改变 AT/RAT。
```

审查点：当前只是吃掉命令避免 unsupported；没有生成 violation check。

### 15.2 SdcClockConstraints.cpp

`_read_sdc(CreateClock)`：

```text
requires:
  obj.period
  obj.name
optional:
  obj.port_pin_list = GetPorts with exactly one source port
write:
  clocks[obj.name]
  clock waveform if obj.waveform exists
```

有 source port 时创建 source clock：

```text
primary_input2pin_id[source_port] -> source_id
Clock(name, source_id, period)
```

没有 source port 时创建 virtual clock：

```text
Clock(name, period)
source_id = -1
```

`_read_sdc(SetClockUncertainty)`：

```text
requires:
  obj.uncertainty
write:
  clock_setup_uncertainty[clock_name]
  clock_hold_uncertainty[clock_name]
may update:
  hostPinRAT(pin_id, attr) for output delays already tied to that clock
objects:
  no object_list -> all clocks
  AllClocks / GetClocks / GetPorts-as-clock-token
```

审查点：这个函数不只是写 uncertainty map；如果 `SetOutputDelay` 已经写过某些 PO RAT，它会用 delta 修正这些 RAT。因此 `set_output_delay` 和 `set_clock_uncertainty` 顺序被这里做了补偿。

`_read_sdc(SetClockTransition)`：

```text
requires:
  obj.transition
  obj.clock_list
write:
  clock_transitions[clock_name][attr]
objects:
  AllClocks / GetClocks / GetPorts-as-clock-token
```

审查点：这里不直接写 `host_pin_slew`；`readSdc()` 后半段会根据 clock mapping 给 clock pins/source pins填 clock slew tensors。
现在 clock slew 不再展开成 `pin_clock_slews[pin * NUM_ATTR + attr]`。`SetClockTransition` 只写 `clock_transitions[clock_name][attr]`；`readSdc()` 后半段通过 `BuildClockIdTablesForSdc()` 把它整理到 `clock_slews[clock_id * NUM_ATTR + attr]`，再由 pin/test 的 `clock_id` 查询。

`_read_sdc(SetClockLatency)`：

```text
requires:
  obj.delay
  obj.object_list
write clock latency:
  clocks[clock_name].set_latency(delay)
write pin latency override:
  pin_clock_latency_overrides[pin_id] = delay  # dense float[num_pins], NaN means unset
lookup pin:
  pin_name2pin_id.find(pin_name)
  fallback primary_input2pin_id.find(pin_name)
objects:
  AllClocks / GetClocks / GetPorts / GetPins
```

第一版只支持 scalar latency。若 command 带 `-early/-late/-rise/-fall/-min/-max/-source`，`SetClockLatencyHasUnsupportedMask()` 会 warning 后跳过该 command，不能把 masked latency 静默当成全属性 scalar。

审查点：

- `GetPins` 和非-clock `GetPorts` 依赖 `pin_name2pin_id` lazy map；所以这些 command 必须在 `preparePinNameMapForSdc()` target collection 覆盖到。
- clock object latency 仍写 `Clock::set_latency(delay)`，后续 `clock_rise_edges/fall_edges` 带 clock-level latency。
- pin latency override 不创建新 clock state；后续 edge 计算规则是 `clock_waveform_*_edges[clock_id] + pin_clock_latency_overrides[pin]`。如果 pin 没有有效 clock context，则 rise/fall edge 都退化为 override 本身。

`_read_sdc(SetPropagatedClock)`：

```text
requires:
  obj.object_list
write:
  propagated_all_clocks
  propagated_clock_names
  propagated_clock_pins
objects:
  AllClocks -> propagated_all_clocks = true
  GetClocks -> propagated_clock_names
  GetPins -> propagated_clock_pins by pin_name2pin_id
  GetPorts -> treated as possible clock name and possible source port
```

`GetPorts` 的双重处理：

```text
add_clock_name(token)
if token is a primary input source of an existing clock:
  add that clock name too
```

`add_source_port_clock(port_name)` lambda 具体做法：

```text
primary_input2pin_id.find(port_name)
  miss -> return
  hit  -> source_pin_id

for each clocks[clock_name]:
  if clock.source_id() == source_pin_id:
    propagated_clock_names.insert(clock_name)

if no clock uses this source pin:
  warn_missing_sdc_object("clock source port", port_name)
```

目的：支持 SDC 里用 clock source port 指定 propagated clock，例如：

```text
create_clock -name core_clk -period ... [get_ports clk]
set_propagated_clock [get_ports clk]
```

如果只执行 `add_clock_name("clk")`，而真实 clock name 是 `core_clk`，后面按 `propagated_clock_names` 判断时匹配不到。`add_source_port_clock()` 通过 `primary_input2pin_id` 找到 `clk` 的 pin id，再反查 `clocks[*].source_id()`，把 `core_clk` 也加入 propagated clock name set。

审查点：这里兼容 Tcl bridge 把 `get_clocks` 对象吐成 bare token 的情况，所以 `GetPorts` 不一定真的是 design port。

`_read_sdc(SetIdealNetwork)`：

```text
effect:
  no-op
reason:
  OpenSTA/OpenROAD 解析 set_ideal_network 但不把它作为 propagated-clock 开关。
```

审查点：Xplace 的 clock policy 是 clocks 默认 ideal，只有 `set_propagated_clock` 才把 clock/pin 标成 propagated。

### 15.3 readSdc() sparse clock 汇总

所有 `_read_sdc(...)` command 读完后，`GTDatabase_sdc.cpp::readSdc()` 会把 clock 相关 map 汇总成 host 侧 sparse clock 表。当前不再上传旧的 per-pin/per-test clock tensors：

```text
deleted old dense state:
  pin_clock_periods[num_pins]
  pin_clock_rise_edges[num_pins]
  pin_clock_fall_edges[num_pins]
  pin_clock_slews[num_pins * NUM_ATTR]
  test_clock_periods[num_tests]
  test_setup_uncertainties[num_tests]
  test_hold_uncertainties[num_tests]
```

新 host state：

```text
clock_names[clock_id]
clock_name2id[name] -> uint16_t clock_id

pin_clock_ids[num_pins]      # real SDC clock id, invalid = 65535
test_clock_ids[num_tests]    # capture clock id

clock_periods[num_clocks]
clock_rise_edges[num_clocks]            # waveform edge + clock object latency
clock_fall_edges[num_clocks]
clock_waveform_rise_edges[num_clocks]   # no clock object latency
clock_waveform_fall_edges[num_clocks]
clock_slews[num_clocks * NUM_ATTR]
clock_setup_uncertainties[num_clocks]
clock_hold_uncertainties[num_clocks]

pin_clock_latency_overrides[num_pins]   # dense float, NaN means unset
pin_clock_is_default_fallback[num_pins] # host-only warning provenance
```

`BuildClockIdTablesForSdc()`：

```text
for each real SDC clock name:
  assign unique uint16_t clock_id
  do not merge same-period clocks
  write period/rise/fall/waveform/slew/setup_uncertainty/hold_uncertainty tables
return default_clock_id = first clock in clocks map, or invalid
```

审查点：`clock_id` 是真实 SDC clock identity，不是 period id。false path / clock-domain 语义后续不能再用 period 合并后的 id。

`AssignClockIdsToPins(default_clock_id, net_clock_ids, sdc_threads)`：

```text
pin_clock_ids[:] = default_clock_id
pin_clock_is_default_fallback[:] = default valid ? 1 : 0
net_clock_ids[:] = invalid
net_is_clock[:] = 0

for each clock:
  source_pin = clock.source_id()
  net_id = gp_pins[source_pin].getParNetId()
  net_clock_ids[net_id] = clock_id
  net_is_clock[net_id] = 1
  pin_clock_ids[source_pin] = clock_id
  pin_clock_is_default_fallback[source_pin] = 0

for each unique clock net:
  for pin_id in gpdb.getNets()[net_id].pins():
    pin_clock_ids[pin_id] = net_clock_ids[net_id]
    pin_clock_is_default_fallback[pin_id] = 0
    if clock_slews[clock_id, attr] finite:
      hostPinSlew(pin_id, attr) = clock_slews[clock_id, attr]
```

审查点：不再全量维护 `Clock* pin_clock_context`。propagated-clock 判断、ideal clock edge、power clock slew 都通过 `pin_clock_ids[pin] -> clock_*` 表完成。

`MapTestsToClockIds(net_clock_ids, default_clock_id, sdc_threads)`：

```text
test_clock_ids[:] = default_clock_id
for each test_id:
  arc_id = test_id2_arc_id[test_id]
  clock_pin = timing_arc_from_pin_id[arc_id]
  net_id = gp_pins[clock_pin].getParNetId()
  if net_clock_ids[net_id] valid:
    test_clock_ids[test_id] = net_clock_ids[net_id]
```

setup/hold uncertainty 不再按 test 展开；DMP CUDA 用 `test_clock_ids[test_id]` 查 `clock_setup_uncertainties/clock_hold_uncertainties`。

host helper：

```text
ClockPeriodForPin(pin)
ClockRiseEdgeForPin(pin)
ClockFallEdgeForPin(pin)
ClockSlewForPin(pin, attr)
ClockSetupUncertaintyForTest(test)
ClockHoldUncertaintyForTest(test)
```

pin edge 规则：

```text
clock_id = pin_clock_ids[pin]
override = pin_clock_latency_overrides[pin]

if override finite and clock_id valid:
  edge = clock_waveform_edge[clock_id] + override
else if override finite:
  edge = override
else if clock_id valid:
  edge = clock_edge[clock_id]
else:
  NaN
```

审查点：

- `GPUTimer` constructor 不再从 `TimingTorchRawDB` 取旧 clock tensor `data_ptr`。
- `DmpModel.cu` 直接从 `timer->gtdb` 上传 `uint16_t pin_clock_ids/test_clock_ids` 和 `clock_*` float tables。
- `release_dmp_timing_scratch_for_power()` 不释放 sparse clock tables，因为 power CUDA 还会复用 `h_dmp_db->pin_clock_ids/clock_slews`。

### 15.4 SdcExceptions.cpp

`_read_sdc(SetCaseAnalysis)`：

```text
requires:
  obj.value in {0, 1, zero, one}
  obj.port_pin_list
write:
  pin_case_values[pin_id] = 0/1
objects:
  GetPorts -> primary_input2pin_id
  GetPins -> pin_name2pin_id
extra effect:
  for GetPorts primary input, hostPinAT(pin_id, attr) = NaN
```

审查点：当前它只记录 per-pin case value，不在 graph build 期删除 cell arc。后续常量传播/disabled arc mask 应放进 `RunSdcConstantSimulation()`。

`_read_sdc(SetFalsePath)`：

```text
requires:
  obj.from
optional:
  obj.to; if absent, to_all_clocks = true
write:
  power_disabled_constraint_arc[arc_id] = 1
scope:
  debug-only power experiment mask
  only active downstream when XPLACE_POWER_APPLY_FALSE_PATHS=1
```

内部流程：

```text
collect from_clock_names / to_clock_names:
  AllClocks
  GetClocks
  GetPorts-as-clock-token

build clock_name_to_id
build pin_node_id / pin_net_id from gpdb pins
mark clock source pins and same-net pins with pin_clock_id
map capture_clock_by_pin from test_id2_arc_id:
  test clock pin -> data pin capture clock
find sequential output pins from Liberty sequentials_
mark launch_clock_by_pin for sequential outputs

forward BFS:
  start from launch pins matching -from clocks
  traverse pin_forward_arc_list through power_edge_valid arcs

backward target scan:
  target pins match -to capture clocks
  inspect pin_backward_arc_list
  if predecessor is reached by forward BFS:
    power_disabled_constraint_arc[arc_id] = 1
```

`power_edge_valid(arc_id)` 过滤：

```text
skip invalid arc
skip test arcs
skip cell arcs ending at sequential output pin
```

审查点：这个不是 timing false-path pruning；它不会改 `timing_arc_*`、不会改 `arc_id2test_id`、不会影响 default timing traversal。当前用途是 power debug mask。

OpenROAD/OpenSTA 的 clock-to-clock `set_false_path` 不是这样处理。源码路径是：

```text
OpenROAD/src/sta/sdc/Sdc.i:798
  make_false_path(...)
    -> Sta::makeFalsePath(...)

OpenROAD/src/sta/search/Sta.cc:2008
  Sta::makeFalsePath(...)
    -> sdc->makeFalsePath(...)
    -> search_->arrivalsInvalid()

OpenROAD/src/sta/sdc/Sdc.cc:3891
  Sdc::makeFalsePath(...)
    -> checkFromThrusTo(...)
    -> new FalsePath(from, thrus, to, min_max, ...)
    -> addException(exception)
```

`FalsePath` 是 `ExceptionPath` 的子类，定义在：

```text
OpenROAD/src/sta/include/sta/ExceptionPath.hh:152
  class FalsePath : public ExceptionPath
    isFalse() = true
    type() = ExceptionPathType::false_path
```

OpenROAD timing search 会把 exception state 放在 path tag 里传播，不是全局删除 arc：

```text
OpenROAD/src/sta/include/sta/ExceptionPath.hh:648
  ExceptionState
    exception()
    matchesNextThru(...)
    isComplete()
    nextState()

OpenROAD/src/sta/search/Search.cc:2615
  exceptionThruStates(...) creates/updates ExceptionState
  existing state follows edge traversal
  completed false-path state can stop propagation for that path/tag

OpenROAD/src/sta/search/Search.cc:3669
  Search::exceptionTo(...)
    states = path->tag()->states()
    for state in states:
      exception = state->exception()
      if exception type matches and sdc->isCompleteTo(...):
        choose highest priority exception
```

endpoint/report 阶段会检查 path tag 上是否带有完整 false-path exception：

```text
OpenROAD/src/sta/search/VisitPathEnds.cc:102
  visitPathEnds(...)
    if ... && !falsePathTo(path, pin, end_rf, path_min_max):
      visit endpoint

OpenROAD/src/sta/search/VisitPathEnds.cc:615
  falsePathTo(...)
    -> search_->exceptionTo(ExceptionPathType::false_path, path, pin, ...)
```

因此 OpenROAD 的语义是 path/tag-level exception：

```text
same physical arc can still be used by other non-false paths
false path decision depends on launch/capture clock, through/to completion, rf, min/max, and path tag state
graph topology itself remains available; search/report filters matching paths
```

当前 Xplace/GPUTimer 不支持这套 path-tag exception：

```text
no per-path tag state in timing propagation
pinAT/pinRAT are merged per pin/attr, not per launch clock or exception state
SetFalsePath does not affect timing CUDA traversal
SetFalsePath does not suppress endpoint slack for matching clock-to-clock paths
current clock-to-clock code only builds optional power_disabled_constraint_arc debug mask
```

审查结论：`set_false_path -from [get_clocks ...] -to [get_clocks ...]` 在当前 Xplace 中不是 OpenROAD-equivalent timing exception。它既不是 path-tag 过滤，也不是默认 timing disabled arc；默认 timing WNS/TNS 仍会包含这些 clock-to-clock false paths。

`readSdc()` 主流程：

```text
clear driving_cell_sources / clock maps / propagated-clock sets
power_disabled_constraint_arc.resize(num_arcs)
InitPinClockLatencyOverrides()
host_pin_slew = NaN
host_pin_load = 0
host_pin_rat = NaN
host_pin_at = NaN

for command in sdc.commands:
  visit _read_sdc(command)

num_timings = liberty_timing_arcs.size()
default_clock_id = BuildClockIdTablesForSdc()
AssignClockIdsToPins(default_clock_id, net_clock_ids)
count/log pin clock latency override coverage
apply clock transition slews to clock pins
MapTestsToClockIds(net_clock_ids, default_clock_id)
fill missing PI slew with 0
seed clock source AT
mark ideal vs propagated clock pins
copy only host_pin_slew/load/rat/at to TimingTorchRawDB tensors
```

`readSdc()` 后半段 sparse clock finalize 展开：

代码范围：`cpp_to_py/gputimer/db/GTDatabase_sdc.cpp`。

这段在所有 `_read_sdc(command)` visitor 已经跑完之后执行，输入状态主要来自：

```text
clocks                              # create_clock / create_generated_clock
clock_transitions                   # set_clock_transition
clock_setup_uncertainty             # set_clock_uncertainty -setup / default setup
clock_hold_uncertainty              # set_clock_uncertainty -hold / default hold
pin_clock_latency_overrides         # dense float[num_pins], NaN means no pin override
propagated_all_clocks
propagated_clock_names
propagated_clock_pins
host_pin_slew/load/rat/at           # 前面 SDC timing constraints 写入
pin_is_clk                          # Liberty clock pin + timing test clock pin + clock source pin
test_id2_arc_id / timing_arc_from_pin_id
```

第一步是把 SDC clock 建成 `uint16_t clock_id`，不再按 period 合并：

```text
BuildClockIdTablesForSdc()
  sort clock names for deterministic id assignment
  clock_names[clock_id] = clock_name
  clock_name2id[clock_name] = clock_id
  clock_periods[clock_id] = clock.period()
  clock_rise_edges[clock_id] = clock.rise_edge()
  clock_fall_edges[clock_id] = clock.fall_edge()
  clock_waveform_rise_edges[clock_id] = clock.waveform_rise_edge()
  clock_waveform_fall_edges[clock_id] = clock.waveform_fall_edge()
  clock_slews[clock_id * NUM_ATTR + attr] = set_clock_transition value or NaN
  clock_setup_uncertainties[clock_id] = setup uncertainty or 0
  clock_hold_uncertainties[clock_id] = hold uncertainty or 0
  return default_clock_id or kInvalidClockId
```

审查点：

- `clock_id` 是真实 SDC clock identity，不是 period id；两个 clock period 相同也不会合并。
- `pin_clock_ids` / `test_clock_ids` 是 `uint16_t`，invalid id 是 `65535`。
- `default_clock = clocks.empty() ? nullptr : &clocks.begin()->second` 仍来自 `unordered_map` 的 begin；unmatched pin/test 的 fallback clock 仍不是严格 SDC 语义。
- `set_clock_transition` 只进 `clock_slews[clock_id, attr]`，不再维护 dense `pin_clock_slews[num_pins * NUM_ATTR]`。

第二步把 clock id 传播到 source pin 和 clock net pins：

```text
AssignClockIdsToPins(default_clock_id, net_clock_ids)
  pin_clock_ids[:] = default_clock_id
  pin_clock_is_default_fallback[:] = default_clock_id valid
  net_clock_ids[:] = invalid
  net_is_clock[:] = 0

for clock in clocks:
  clock_id = clock_name2id[clock.name()]
  source_pin = clock.source_id()
  net_id = gp_pins[source_pin].getParNetId()
  net_clock_ids[net_id] = clock_id
  net_is_clock[net_id] = 1
  pin_clock_ids[source_pin] = clock_id
  pin_clock_is_default_fallback[source_pin] = 0

for each clock net:
  for pin_id in gpdb.getNets()[net_id].pins():
    pin_clock_ids[pin_id] = net_clock_ids[net_id]
    pin_clock_is_default_fallback[pin_id] = 0
    if clock_slews[clock_id, attr] finite:
      hostPinSlew(pin_id, attr) = clock_slews[clock_id, attr]
```

如果多个 SDC clock source 落在同一个 net，当前代码 warning，并保留已经写入的第一个 `net_clock_ids[net_id]`。

第三步处理 `set_clock_latency` pin override：

```text
SetClockLatencyHasUnsupportedMask(obj):
  if -early/-late/-rise/-fall/-min/-max/-source appears:
    warn and skip this command

ApplyScalarPinClockLatencyOverride(pin_id, delay):
  pin_clock_latency_overrides[pin_id] = delay

ClockRiseEdgeForPin(pin_id):
  clock_id = pin_clock_ids[pin_id]
  override = pin_clock_latency_overrides[pin_id]
  if override finite and clock_id valid:
    return clock_waveform_rise_edges[clock_id] + override
  if override finite:
    return override
  return clock_rise_edges[clock_id]

ClockFallEdgeForPin(pin_id):
  same rule with clock_waveform_fall_edges / clock_fall_edges
```

这里刻意用 `waveform_*_edge() + override`，而不是 `clock.rise_edge()` / `fall_edge()`。原因是 `Clock::rise_edge()` 已经包含 clock-object latency；pin override 按 OpenSTA 语义应覆盖 pin latency，避免把 clock latency 和 pin latency 叠加两次。

第四步把 timing test 映射到 capture clock：

```text
test_clock_ids.assign(num_tests, default_clock_id)

for test_id:
  arc_id = test_id2_arc_id[test_id]
  clock_pin_id = timing_arc_from_pin_id[arc_id]
  net_id = gp_pins[clock_pin_id].getParNetId()
  if net_clock_ids[net_id] valid:
    test_clock_ids[test_id] = net_clock_ids[net_id]
```

这一步给 DMP test propagation 用：

```text
DmpGateProp.cu::clockPeriodForTest(test_id)
  test_clock_ids[test_id] -> clock_periods[clock_id]

DmpGateProp.cu::propagateTest(...)
  hold:  pinRat = related_at + constraint + clock_hold_uncertainties[test_clock_ids[test_id]]
  setup: pinRat = related_at - constraint - clock_setup_uncertainties[test_clock_ids[test_id]]
```

因此 `output_delay_clock_by_pin_attr` 修正 PO RAT 是 `_read_sdc(SetClockUncertainty)` 里做的；register timing test 的 setup/hold uncertainty 是这里通过 capture clock id 间接查 `clock_*_uncertainties`，CUDA 后续传播 test 时再使用。

第五步处理 clock source AT 和 ideal/propagated clock：

```text
PI missing slew:
  NaN -> 0

for clock source pin:
  pin_is_clk[source_pin] = 1
  if hostPinAT source attr is NaN:
    rise attrs use ClockRiseEdgeForPin(source_pin)
    fall attrs use ClockFallEdgeForPin(source_pin)

for every pin_is_clk pin:
  direct_propagated = pin in propagated_clock_pins
  clock_id = pin_clock_ids[pin_id]
  clock_propagated = propagated_all_clocks || clock_names[clock_id] in propagated_clock_names
  pin_is_ideal_clk[pin_id] = propagated ? 0 : 1

for ideal clock pin:
  hostPinAT rise attrs = ClockRiseEdgeForPin(pin_id)
  hostPinAT fall attrs = ClockFallEdgeForPin(pin_id)
```

也就是说，当前 policy 是：

```text
default: clock pin ideal, AT 直接由 SDC waveform/latency 给定
set_propagated_clock: 把匹配 clock/pin 从 ideal 变 propagated，不再强行 seed AT
```

第六步只把四个 timing state arrays 封成 Torch tensors 并搬到 `timing_raw_db.node_size_x.device()`：

```text
pinSlew/pinLoad/pinRAT/pinAT
```

随后 `host_pin_slew/load/rat/at` 被 `swap` 释放；`clock_names`、`pin_clock_ids`、`test_clock_ids`、`clock_*` 和 `pin_clock_latency_overrides` 这些 `GTDatabase` host vectors 保留。DMP CUDA 不再通过 `TimingTorchRawDB` 读 clock tensors，而是在 `DmpModel.cu` 里从 `timer->gtdb` 直接上传 sparse clock tables：

```text
DmpModel.cu
  cudaMalloc/cudaMemcpy pin_clock_ids[num_pins]
  cudaMalloc/cudaMemcpy test_clock_ids[num_tests]
  cudaMalloc/cudaMemcpy clock_periods[num_clocks]
  cudaMalloc/cudaMemcpy clock_rise_edges / clock_fall_edges
  cudaMalloc/cudaMemcpy clock_waveform_rise_edges / clock_waveform_fall_edges
  cudaMalloc/cudaMemcpy clock_slews[num_clocks * NUM_ATTR]
  cudaMalloc/cudaMemcpy clock_setup_uncertainties / clock_hold_uncertainties
  cudaMalloc/cudaMemcpy dense pin_clock_latency_overrides[num_pins]
```

power host/CUDA 也读这些 sparse clock state：

```text
PowerActivityHostUtils.cpp::powerClockActivityForPin()
  GTDatabase::ClockPeriodForPin / ClockRiseEdgeForPin / ClockFallEdgeForPin

PowerActivityCpu.cpp
  GTDatabase::ClockSlewForPin

PowerCudaInputRoots.cpp::powerIsClockSlewPin()
  GTDatabase::ClockSlewForPin

PowerCudaInputBuild.cpp / PowerCudaActivity.cu
  reuse h_dmp_db->pin_clock_ids and h_dmp_db->clock_slews
```

审查点：

- `clock_id` 可以区分同 period 不同 waveform/uncertainty 的 clock；false-path/debug domain 逻辑也不应再用 period id。
- `set_clock_latency` 第一版只支持 scalar；masked latency 只 warning+skip，不会拆成 early/late/rise/fall state。
- dense `pin_clock_latency_overrides[num_pins]` 牺牲一点 host/device 内存，避免第一版再设计 sparse override id。
- `set_clock_transition` 对 clock net pins 先写 `hostPinSlew`；后面对所有 `pin_is_clk` 再用 `ClockSlewForPin()` 补 clock slew。
- 代码注释写的是 “set nan slew of PIs to half period”，但当前实现把 PI NaN slew 写成 `0.0f`，不是 half period。
- `net_is_clock` 只按 clock source net 标记，后续 `initialize()` 会复制到独立 CUDA buffer；它和 sparse `pin_clock_ids` 不是同一个语义。
- `release_dmp_timing_scratch_for_power()` 不能释放 `h_dmp_db->pin_clock_ids/test_clock_ids/clock_*`，因为 power CUDA 会复用 `pin_clock_ids` 和 `clock_slews`。
- `readSdc()` 依赖 structural graph 先建好：`test_id2_arc_id`、`timing_arc_from_pin_id`、`pin_is_clk`、`primary_inputs` 都来自前面的 `ExtractTimingGraph()`。

`RunSdcConstantSimulation()` 是 `readSdc()` 后的独立阶段：

```text
RunSdcConstantSimulation()
  # current: placeholder only
  # future:
  seed pin_logic_values from pin_case_values
  seed Liberty constant output functions
  propagate constants through net drivers/sinks and cell output functions
  build timing_arc_disabled_by_constant as timing-only mask
```

审查边界：

- `readSdc()` 直接在 pybind flow 里调用，不再包额外 wrapper。
- `RunSdcConstantSimulation()` 可以复用旧建图期 constant helper 的思想，但输出应是后置 simulation state/mask，不应该物理删除 canonical `timing_arc_*` graph arrays。
- power 可以复用 `pin_logic_values` 作为 fixed 0/1 activity seed；不要默认复用 timing disabled arc mask。

输出 tensors：

```text
pinSlew [num_pins, NUM_ATTR]
pinLoad [num_pins, NUM_ATTR]
pinRAT  [num_pins, NUM_ATTR]
pinAT   [num_pins, NUM_ATTR]
```

审查点：

- `readSdc()` 会依赖 `test_id2_arc_id` 和 `timing_arc_from_pin_id` 来给 timing tests 找 capture clock。
- `SetOutputDelay` 写 PO RAT；`SetClockUncertainty` 可能回头修正这些 RAT。
- `SetDrivingCell` 可能追加 `liberty_timing_arcs` 并填 `driving_cell_sources`；DMP 后续用这些 source metadata。
- `SetFalsePath` 当前是 debug-only power experiment mask，不是默认 timing path pruning。
- sparse clock tables 是 `GTDatabase` host state + `DmpModel` CUDA raw buffers，不再是 `TimingTorchRawDB` 输出 tensors。
