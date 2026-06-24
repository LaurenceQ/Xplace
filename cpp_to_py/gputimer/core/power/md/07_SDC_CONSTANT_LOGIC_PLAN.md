# SDC 常量逻辑传播与 OpenSTA 对齐计划

## 目标

当前只做保守对齐，不一次性复刻完整 OpenSTA `Sim`。

第一阶段支持：

- `set_case_analysis 0/1/zero/one`。
- netlist tie-high/tie-low。
- Liberty output `function` 能证明出的 0/1 常量。
- per-arc timing sense override 和 disabled arc mask。
- power 复用传播后的常量 pin value 做 fixed activity seed。

第一阶段不支持：

- `set_case_analysis rising/falling`。
- `set_logic_zero` / `set_logic_one` / `set_logic_dc`。
- 完整 Liberty `when` / mode condition / tristate enable。
- 穿过 sequential state 的传播。
- 用 timing disabled arc 默认切 power path。

正确性原则：

```text
能证明 disabled -> disable
证明不了 / 表达式不支持 / 语义不完整 -> 保持 active
```

也就是说第一版可以比 OpenSTA 少剪 arc，但不能比 OpenSTA 更乐观。

## 合并后删除的旧方案冲突

旧文档里有几类重复或容易误导的内容，合并时统一删掉：

- 两套函数接口：`SeedSdcLogicValues(...)` 和 `InitializeSdcLogicValues()` 语义重复，统一成下面的 `SeedSdcLogicValues(...)`。
- 两套测试列表：保留 synthetic 小测试作为单元验证，真实通过门槛只认 `mempool_tile_wrap` 和 `picorv32a`。
- “直接改 timing arc / timing sense”的描述：删掉。OpenSTA 的结果是 per instance edge，不是 per Liberty timing arc；Xplace 也必须 per structural arc override。
- power 和 timing 的 disabled arc 混用：删掉。power 先只用 `pin_logic_values` 做 fixed activity，不默认使用 timing disabled mask。
- `std::vector<bool>` 作为 CUDA 上传结构：删掉。host 临时 bool 可以用 `bool`，需要上传或跨 CUDA 边界的 mask 用 `uint8_t`，避免 `vector<bool>` proxy 语义。

## OpenSTA 行为基线

OpenSTA 的核心流程是“SDC 只存 seed，Sim lazy propagation，Search 用 sim 结果过滤 edge”。

```text
Sdc.tcl::set_case_analysis(value, pins)
  -> Sdc.i::set_case_analysis_cmd(pin, value)
    -> Sta::setCaseAnalysis(pin, value, mode)
      -> mode->sdc()->setCaseAnalysis(pin, value)
      -> mode->sim()->constantsInvalid()
      -> delaysInvalid()
      -> power_->activitiesInvalid()

Sdc::setCaseAnalysis(pin, value)
  -> case_value_map_[pin] = value
```

关键点：

- `set_case_analysis` 命令本身不传播常量，也不删除 timing edge。
- 它只把 pin -> `LogicValue` 写进 `Sdc::case_value_map_`。
- 之后 timing/search/report 访问 constant 信息时触发 `Sim::ensureConstantsPropagated()`。

OpenSTA propagation：

```text
Sim::ensureConstantsPropagated()
  ensureConstantFuncPins()
  clearSimValues()
  seedConstants()
    enqueueConstantPinInputs()              # netlist tie high/low
    setConstraintConstPins(logicValues)     # set_logic_*
    setConstraintConstPins(caseLogicValues) # set_case_analysis
    setConstFuncPins()                      # Liberty constant function 0/1
  propagateConstants(false)
    evalInstance(inst, false)
      eval output Liberty function
      do not propagate through sequential state
  findDisabledEdges()
```

OpenSTA disabled edge annotation：

```text
Sim::findDisabledEdges(inst, output_pin, output_vertex)
  for each input cell edge into output_vertex:
    if from vertex is constant:
      simTimingSense(edge) = none
    else:
      simTimingSense(edge) = functionSense(inst, from_pin, output_pin)

    if sense != none:
      disabled_cond = isDisabledCond(edge, inst, from_pin, output_pin)
                   || isDisabledMode(edge, inst)

    setSimTimingSense(edge, sense)
    setIsDisabledCond(edge, disabled_cond)
```

OpenSTA timing traversal：

```text
SearchPred0::searchThru(edge)
  reject timing check
  reject SDC disabled constraint
  reject sim->isDisabledCond(edge)
  reject sdc->isDisabledCondDefault(edge)
  reject sim->simTimingSense(edge) == none

searchThruTimingSense(edge, from_rf, to_rf)
  positive_unate: from_rf == to_rf
  negative_unate: from_rf != to_rf
  non_unate/unknown: keep
  none: reject
```

OpenSTA report Tcl：

```text
report_constant obj
  report_pin_constant(pin)
    pin_sim_logic_value(pin)  # triggers/uses Sim propagated value
    pin_case_logic_value(pin) # direct set_case_analysis value
    pin_logic_value(pin)      # direct set_logic_* value

report_disabled_edges
  disabled_edges_sorted
  edge_disable_reason_verbose(edge)
```

源码锚点：

- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/sdc/Sdc.tcl:2674`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/sdc/Sdc.i:1262`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/Sta.cc:1916`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/sdc/Sdc.cc:3767`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/Sim.cc:287`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/Sim.cc:481`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/Sim.cc:638`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/Sim.cc:724`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/Sim.cc:908`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/SearchPred.cc:122`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/SearchPred.cc:270`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/graph/Graph.tcl:194`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/graph/Graph.tcl:237`

## Xplace 当前状态

当前入口已经存在：

```text
PyBindCppMain.cpp
  gtdb->readSdc(*sdc)
  gtdb->RunSdcConstantSimulation()
```

当前实现状态：

```text
GTDatabase_sdc.cpp::RunSdcConstantSimulation()
  当前只是占位

SdcExceptions.cpp::_read_sdc(SetCaseAnalysis)
  接受 0/1/zero/one
  GetPorts -> primary_input2pin_id
  GetPins  -> pin_name2pin_id
  写 pin_case_values[pin_id] = 0/1
```

当前需要注意的架构差异：

- OpenSTA 是 lazy propagation；Xplace 第一版做显式 post-SDC stage。
- Xplace graph 已经在 read SDC 前构建好；constant sim 不改 canonical graph。
- `readSdc()` 当前会上传 `pinSlew/pinLoad/pinRAT/pinAT` 到 tensor；constant sim 后如果有 timing/power tensor 需要新增上传点，不能假设 readSdc 已覆盖。

## Xplace 目标数据流

目标流程：

```text
GTDatabase::ExtractTimingGraph()
  build full structural timing graph

GTDatabase::readSdc(sdc)
  _read_sdc(SetCaseAnalysis)
    pin_case_values[pin_id] = 0/1
  build clock / IO / timing constraint tables
  upload existing timing tensors

GTDatabase::RunSdcConstantSimulation()
  SeedSdcLogicValues()
  PropagateSdcLogicValues()
  AnnotateSdcLogicTimingArcs()
  UploadSdcLogicTensors()

gt::GPUTimer(...)
  timing CUDA reads disabled mask / sense override
  power CUDA reads pin_logic_values for fixed activity seed
```

目标和 OpenSTA 对应关系：

| Xplace | OpenSTA | 含义 |
|---|---|---|
| `pin_case_values` | `Sdc::case_value_map_` | SDC 直接 seed，不代表传播后结果 |
| `pin_logic_values` | `Sim::sim_value_map_` | propagation 后 pin 逻辑值 |
| `sdc_arc_timing_sense` | `Sim::edge_timing_sense_map_` | per structural arc 的 timing sense override |
| `sdc_logic_disabled_arcs` | `simTimingSense(edge)==none` / `isDisabledCond(edge)` | timing traversal skip mask |

## 成员和类型

`GTDatabase` host 成员：

```cpp
std::vector<int8_t> pin_case_values;          // -1 unknown, 0/1 direct SDC seed
std::vector<int8_t> pin_logic_values;         // -1 unknown, 0/1 propagated value
std::vector<uint8_t> sdc_logic_disabled_arcs; // 0 active, 1 timing disabled
std::vector<int8_t> sdc_arc_timing_sense;     // -1 no override, otherwise TimingSense code
```

`uint8_t` 用在 arc mask 上，因为它后续需要传到 CUDA。局部临时判断可以用 `bool`，但不要把 `std::vector<bool>` 作为跨函数/跨 CUDA 边界的数据结构。

`sdc_arc_timing_sense` 建议编码：

```text
-1 no override
 0 none
 1 positive_unate
 2 negative_unate
 3 non_unate
 4 unknown
```

不能修改 `TimingArc::timing_sense_`：

- `TimingArc` 是 Liberty arc 级别。
- OpenSTA 的 `simTimingSense(edge)` 是 graph edge / instance edge 级别。
- 同一个 Liberty timing arc 在不同 instance 上可能因为 side input 常量不同而得到不同 effective sense。

## 函数接口

`GTDatabase` 成员函数：

```cpp
void RunSdcConstantSimulation();
void SeedSdcLogicValues(std::deque<int>& pin_queue);
void PropagateSdcLogicValues(std::deque<int>& pin_queue);
void AnnotateSdcLogicTimingArcs();
void UploadSdcLogicTensors();
```

辅助成员函数：

```cpp
bool SetPinLogicValue(int pin_id, int8_t value, std::deque<int>& pin_queue);
int8_t EvaluateCellOutputLogic(int gpdb_id, int output_port_id) const;
int8_t EvaluateLibertyExprLogic(const LibertyFuncExpr& expr, int gpdb_id) const;
int8_t EvaluateArcTimingSenseWithSdcLogic(int arc_id) const;
bool IsSequentialCellNode(int gpdb_id) const;
```

主流程：

```text
RunSdcConstantSimulation()
  if no pin_case_values and no constant function seeds:
    clear/empty SDC logic tensors
    return

  pin_logic_values.assign(num_pins, -1)
  sdc_logic_disabled_arcs.assign(num_arcs, 0)
  sdc_arc_timing_sense.assign(num_arcs, -1)

  queue = {}
  SeedSdcLogicValues(queue)
  PropagateSdcLogicValues(queue)
  AnnotateSdcLogicTimingArcs()
  UploadSdcLogicTensors()
  log summary
```

### SeedSdcLogicValues

输入：

- `pin_case_values`
- Liberty output `function`
- tie-high/tie-low 信息，如果 rawdb/gpdb 能识别

输出：

- 写 `pin_logic_values`
- changed pin 入 `pin_queue`

规则：

- 直接 SDC seed 优先级最高。
- Liberty output function 如果能在全 unknown input 下证明为 0/1，则 seed output pin。
- 如果同一个 pin 得到冲突值，保留 SDC 直接值并 warning。

### PropagateSdcLogicValues

输入：

- `pin_queue`
- `pin_logic_values`
- `pin_forward_arc_list` / `pin_backward_arc_list`
- `timing_arc_from_pin_id` / `timing_arc_to_pin_id`
- `arc_types`

输出：

- 更新 `pin_logic_values`

规则：

- net arc：driver 已知则传播到 sink。
- cell arc：不要按单条 arc 直接传播 output；按 cell instance 汇总 inputs 后评估 output function。
- sequential cell 不传播 Q/D 状态，保持 OpenSTA `propagateConstants(false)` 的保守语义。
- 表达式不能 reduce 到 0/1 就保持 unknown。

### AnnotateSdcLogicTimingArcs

输入：

- `pin_logic_values`
- structural timing arc arrays
- Liberty timing arc / output function

输出：

- `sdc_logic_disabled_arcs`
- `sdc_arc_timing_sense`

规则：

- net arc：from pin 已知常量，则 disable timing propagation。
- cell arc：
  - from pin 已知常量 -> disable。
  - to pin 已知常量 -> disable。
  - 否则用 output function 和当前 side-input constants 计算 effective sense。
  - effective sense 是 `none` -> disable。
  - effective sense 是 positive/negative/non_unate 且和 Liberty 原始 sense 不同 -> 写 override。
  - unknown / unsupported -> 保持 active，不写 override。
- timing test arc：
  - data pin 已知常量 -> disable。
  - 其它保持 active。

### UploadSdcLogicTensors

第一版需要给 timing/power 都能看到：

```text
pin_logic_values
sdc_logic_disabled_arcs
sdc_arc_timing_sense
```

CUDA 规则：

```text
arc disabled:
  if sdc_logic_disabled_arcs[arc_id] != 0:
    skip this arc

effective timing sense:
  if sdc_arc_timing_sense[arc_id] >= 0:
    use override
  else:
    use Liberty timing sense by timing_id

power fixed activity:
  pin_logic_values[pin_id] == 0 -> density 0, duty 0
  pin_logic_values[pin_id] == 1 -> density 0, duty 1
```

Power 默认不使用 `sdc_logic_disabled_arcs` 切断 activity path。

## Liberty 表达式复用边界

不要把 power activity 的 host class/struct 整体提升成 timing 公共类。可复用的是 Liberty expression evaluation 逻辑，而不是 power 数据结构。

建议抽一个小的公共 helper：

```cpp
class LibertyLogicExprEval {
public:
    static int8_t eval_value(const LibertyFuncExpr& expr,
                             const std::vector<int8_t>& port_values);

    static int8_t eval_timing_sense(const LibertyFuncExpr& expr,
                                    int from_port_id,
                                    const std::vector<int8_t>& port_values);
};
```

第一版不需要完整 BDD，可以做保守枚举：

- 收集 expression 中除 `from_port_id` 外的 unknown input。
- unknown 数量小于阈值时枚举 0/1。
- 所有枚举下 output 都不随 from_port 变化 -> `none`。
- 所有有效枚举同向 -> positive 或 negative。
- 有正有负 -> non_unate。
- 表达式不支持、unknown 太多、无法证明 -> unknown。

## OpenSTA Tcl Debug Sim 计划

目的不是马上跑完整 timing/power，而是先得到 OpenSTA 对 constant propagation 的可审查 oracle：

```text
输入 SDC seed
  -> OpenSTA pin_sim_logic_value
  -> OpenSTA report_constant
  -> OpenSTA report_disabled_edges
  -> OpenSTA report_checks
```

### 1. 先跑 OpenSTA 自带小测试

优先用这两个 Tcl 作为语义样本：

- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/test/search_levelize_sim.tcl`
- `/research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/test/search_sim_const_prop.tcl`

它们覆盖：

- `set_case_analysis 0/1`
- `set_logic_zero/one`
- `pin_sim_logic_value`
- `report_constant`
- `report_disabled_edges`
- `report_checks`
- sequential 不穿透传播

执行模板：

```bash
cd /research/d7/ascstd/qkduan25/OpenROAD/src/sta/search/test
/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad search_levelize_sim.tcl \
  > /tmp/xplace_sdc_sim_search_levelize_sim.log 2>&1
/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad search_sim_const_prop.tcl \
  > /tmp/xplace_sdc_sim_search_sim_const_prop.log 2>&1
```

输出里优先看：

```text
pin_sim_logic_value 的 0/1/X
report_constant 的 sim/case/logic 三列
report_disabled_edges 的 edge 和 reason
report_checks 路径是否消失或变化
```

### 2. 跑 mempool_tile_wrap SDC seed oracle

真实 positive case：

```text
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/visible/mempool_tile_wrap/mempool_tile_wrap.sdc:16
set_case_analysis 0 [get_ports {scan_enable_i}]
```

建议生成独立 debug Tcl 到 `/tmp/xplace_sdc_sim_mempool_tile_wrap.tcl`，不要污染 benchmark 文件：

```tcl
set bench /research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks
set design_dir $bench/visible/mempool_tile_wrap

read_liberty $bench/NanGate45/lib/NangateOpenCellLibrary_typical.lib
foreach lib [glob -nocomplain $bench/NanGate45/lib/fakeram45_*.lib] {
  read_liberty $lib
}
read_lef $bench/NanGate45/lef/NangateOpenCellLibrary.tech.lef
read_lef $bench/NanGate45/lef/NangateOpenCellLibrary.macro.mod.lef
foreach lef [glob -nocomplain $bench/NanGate45/lef/fakeram45_*.lef] {
  read_lef $lef
}
read_verilog $design_dir/mempool_tile_wrap.v.gz
link_design mempool_tile_wrap
read_sdc $design_dir/mempool_tile_wrap.sdc

puts "=== constant seed ==="
report_constant [get_ports scan_enable_i]

puts "=== selected propagated constants ==="
# 后续根据 fanout cone 选择少量 pins/cells，避免 log 巨大。
# 示例：
# report_constant [get_nets -of_objects [get_ports scan_enable_i]]

puts "=== disabled edges ==="
report_disabled_edges

puts "=== timing ==="
report_checks -path_delay max -fields {slew cap input_pins net} -digits 4 -group_count 3
```

注意：

- `report_disabled_edges` 在大设计上可能很大，不要直接刷到终端，统一重定向到 `/tmp`。
- 第一轮先只开 `report_constant [get_ports scan_enable_i]` 和 `report_checks -group_count 3`。
- 第二轮再打开全量 `report_disabled_edges`，确认 log 规模。
- 如果 log 太大，第三版 Tcl 改成只围绕 `scan_enable_i` fanout cone 取样。
- 第一轮目标是确认 `scan_enable_i` 的 `case=0`、下游若干 pin 的 sim value、以及 OpenSTA disabled edge 的格式。

已执行结果：

```text
script:
  /research/d7/ascstd/qkduan25/Xplace/tmp_sdc_sim_mempool_tile_wrap.tcl

log:
  /tmp/xplace_sdc_sim_mempool_tile_wrap.log

OpenSTA report_constant:
  scan_enable_i 0 case=0

get_fanout -from [get_ports scan_enable_i] -pin_levels 1 -flat:
  count = 1
  scan_enable_i 0 case=0

get_fanout -from [get_ports scan_enable_i] -pin_levels 2 -flat:
  count = 1
  scan_enable_i 0 case=0
```

这个 Tcl 只用于 SDC seed/dataflow probe；它没有跑完整 physical timing/power flow。`mempool_tile_wrap` 如果要做 timing/power 对齐，应切到 ISPD2025 官方 OpenROAD flow 或 Xplace 现有 ISPD25 compare flow，而不是继续扩这个简化 Tcl。

### 3. 跑 picorv32a MUX oracle

`picorv32a` 原始 SDC 没有 `set_case_analysis`，它不是 SDC positive case。它用于验证 mux function / timing sense。

现有 `picorv32a.opensta.tcl` 是 OpenSTA 风格入口，只读 Liberty / Verilog / SPEF / SDC，不读 DEF/LEF，也不跑 `report_power`。如果要和 Xplace timing/power flow 对齐，应该按已有 power alignment 脚本的 OpenROAD 顺序跑完整 physical flow：

```text
read_liberty -min/-max
read_lef
read_def
read_verilog
link_design
read_sdc
read_spef
set_propagated_clock
report_checks
report_power
targeted mux probe
```

当前 debug Tcl：

```text
/research/d7/ascstd/qkduan25/Xplace/tmp_sdc_sim_picorv32a_mux.tcl
```

实际读入顺序：

```tcl
set design_dir /research/d7/ascstd/qkduan25/TimingPredict/data/netlists/picorv32a
set techlib /research/d7/ascstd/qkduan25/TimingPredict/data/netlists/techlib

read_liberty -min $techlib/sky130_fd_sc_hd__ff_n40C_1v95.lib
read_liberty -max $techlib/sky130_fd_sc_hd__ss_100C_1v60.lib
read_lef $techlib/merged_unpadded.lef
read_def $design_dir/20-picorv32a.def
read_verilog $design_dir/picorv32a.synthesis_preroute.v
link_design picorv32a
read_sdc $design_dir/picorv32a.cts_1.sdc
read_spef $design_dir/20-picorv32a.spef
set_propagated_clock [get_clocks *]
set_units -power mW
```

targeted probe：

```tcl
puts "=== mux baseline constants ==="
foreach obj {
  _25722_/S
  _25722_/X
  _16254_/A
  _16254_/X
  _16302_/A
  _16302_/Y
} {
  report_constant [get_pins $obj]
}

report_checks -path_delay max -digits 4 -group_count 3
report_power -digits 8

foreach obj {
  _25722_/S
  _25722_/X
  _16254_/A
  _16254_/X
  _16302_/A
  _16302_/Y
} {
  report_checks -through [get_pins $obj] -path_delay max -digits 4 -group_count 1
}
```

如果要把它变成 SDC positive：

```tcl
set_case_analysis 0 [get_pins _25722_/S]
report_constant [get_cells _25722_]
report_disabled_edges
report_checks -through [get_pins _25722_/S] -path_delay max -digits 4
report_checks -through [get_pins _25722_/X] -path_delay max -digits 4
```

通过要求：

- 原始 SDC 下 Xplace 不应因为新增 constant sim 改变 baseline。
- 加 synthetic `set_case_analysis` 后，Xplace 只 disable OpenSTA 也 disable 的 mux arc。
- 绝不允许 Xplace disable OpenSTA 仍 active 的 mux data arc。

已执行结果：

```text
log:
  /tmp/xplace_sdc_sim_picorv32a_mux.log

baseline:
  _25722_/S  = X
  _25722_/X  = X
  _16254_/A  = X
  _16254_/X  = X
  _16302_/A  = X
  _16302_/Y  = X

report_power total:
  internal  = 5.00939291e-03
  switching = 4.00815014e-03
  leakage   = 8.79570843e-05
  total     = 9.10550013e-03

synthetic:
  set_case_analysis 0 [get_pins _25722_/S]
  report_constant [get_cells _25722_]:
    A0 0
    A1 X
    S  0 case=0
    X  0
  report_checks -through _25722_/S: No paths found
  report_checks -through _25722_/X: No paths found
```

这个结果说明 `_25722_` 这个 mux 的 synthetic SDC seed 会让 OpenSTA 把 select pin 和 output pin 都推成常量 0，并让 through `S` / `X` 的 max path 消失。Xplace 对齐时不能只改 pin value，还要让 per-arc timing disable/sense override 影响 `report_checks` 对应的 traversal。

### 4. Tcl 输出解析计划

建议保存三个文件：

```text
/tmp/xplace_sdc_sim_search_levelize_sim.log
/tmp/xplace_sdc_sim_mempool_tile_wrap.log
/tmp/xplace_sdc_sim_picorv32a_mux.log
```

解析字段：

```text
report_constant:
  full pin name
  sim value
  case value
  logic value

report_disabled_edges:
  instance/wire edge
  from pin
  to pin
  reason text

report_checks:
  endpoint
  startpoint
  slack
  path pins
```

后续可以写一个小 parser，但第一轮先人工抽样足够。

## 验证 case

### Synthetic 小单元

这些用于保护基本语义，不作为真实 benchmark 通过门槛。

1. AND controlling value

```text
set_case_analysis 0 a
AND(a, b) -> y
预期 y=0，b->y disabled，a->y disabled
```

2. AND non-controlling value

```text
set_case_analysis 1 a
AND(a, b) -> y
预期 a disabled，b->y active，y unknown
```

3. OR controlling value

```text
set_case_analysis 1 a
OR(a, b) -> y
预期 y=1，b->y disabled
```

4. MUX select fixed

```text
set_case_analysis 0 sel
y = sel ? d1 : d0
预期 d0->y active，d1->y disabled，sel->y disabled
```

5. internal pin case analysis

```text
set_case_analysis 0 [get_pins u1/A1]
预期 preparePinNameMapForSdc() 收集 SetCaseAnalysis get_pins target
```

6. unsupported rising/falling

```text
set_case_analysis rising [get_ports a]
预期 Xplace warning + 保持 active
```

### ISPD2025 positive cases

当前只找到 mempool 系列有 logic sim 命令：

```bash
rg -n --no-ignore --follow \
  "set_case_analysis|set_logic_(zero|one|dc)|set_logic" \
  /research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks \
  -g '*.sdc'
```

结果：

| dataset | design | SDC command |
|---|---|---|
| visible | `mempool_tile_wrap` | `set_case_analysis 0 [get_ports {scan_enable_i}]` |
| blind | `mempool_tile_wrap` | `set_case_analysis 0 [get_ports {scan_enable_i}]` |
| visible | `mempool_group` | `set_case_analysis 0 [get_ports {scan_enable_i}]` |
| blind | `mempool_group` | `set_case_analysis 0 [get_ports {scan_enable_i}]` |
| visible | `mempool_cluster` | `set_case_analysis 0 [get_ports {testmode_i}]` |
| visible | `mempool_cluster` | `set_case_analysis 0 [get_ports {scan_enable_i}]` |
| blind | `mempool_cluster` | `set_case_analysis 0 [get_ports {testmode_i}]` |
| blind | `mempool_cluster` | `set_case_analysis 0 [get_ports {scan_enable_i}]` |

第一阶段通过门槛：

- `visible/mempool_tile_wrap` 必须对齐 OpenSTA `report_constant` 和常量 disabled edge 方向。
- `picorv32a` MUX regression 必须不 over-disable。

后续再跑：

- `visible/mempool_group`
- `visible/mempool_cluster`
- blind mempool 系列

`ariane`、`bsg_chip`、`NV_NVDLA_partition_c` 没有 `set_case_analysis` / `set_logic_*`，只作为 no-regression。

### OpenROAD_ISPD25 baseline

这些 case 没有 logic sim 命令，只验证新增 constant sim 不改变 baseline：

| library | designs |
|---|---|
| Nangate45 | `aes_nangate45`, `gcd_nangate45`, `tinyRocket_nangate45` |
| Sky130 HD | `aes_sky130hd`, `gcd_sky130hd`, `ibex_sky130hd`, `jpeg_sky130hd` |
| Sky130 HS | `aes_sky130hs`, `gcd_sky130hs`, `ibex_sky130hs`, `jpeg_sky130hs` |

## 对齐判据

OpenSTA oracle：

- `report_constant` 中 sim=0/1 的 pin，Xplace `pin_logic_values` 应一致。
- `report_disabled_edges` 中因 constant / simulated sense none disabled 的 edge，Xplace 对应 arc 应 disable；不支持条件下允许保持 active，并记录 conservative fallback。
- OpenSTA active 的 data arc，Xplace 不允许 disable。

Timing：

- 支持子集小 case 的 path 集合和 slack 变化方向应和 OpenSTA 一致。
- Xplace 少 disable 可以接受；over-disable 不能接受。

Power：

- `pin_logic_values=0` -> density 0, duty 0。
- `pin_logic_values=1` -> density 0, duty 1。
- 不要求 power path cut 和 OpenSTA timing disabled edges 一致。

## 实现后命令

Build：

```bash
conda activate gnn
cd /research/d7/ascstd/qkduan25/Xplace/build
make -j8
make install
```

OpenROAD：

```bash
/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad /tmp/xplace_sdc_sim_mempool_tile_wrap.tcl
```

Xplace：

```bash
python run_timer.py --designName mempool_tile_wrap
python compare_dmp_openroad_csv --designName picorv32a --platformPath /research/d7/ascstd/qkduan25/Xplace/sky130hd --designPath /research/d7/ascstd/qkduan25/TimingPredict/data/netlists --gpu 0 --load_from_raw True
```

报告至少记录：

```text
case name
OpenSTA report_constant sample
OpenSTA report_disabled_edges sample
Xplace pin_logic_values count
Xplace disabled arc count
Xplace conservative fallback count
是否出现 over-disable
timing/power 是否回退
```
