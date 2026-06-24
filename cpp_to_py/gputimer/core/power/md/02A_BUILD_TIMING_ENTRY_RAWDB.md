# Build Timing Graph: Entry 和 RawDB

Last reviewed: 2026-06-23

拆自 `02_BUILD_TIMING_GRAPH.md`。本文覆盖 Python wrapper、`create_timing_rawdb`、`create_gputimer` 和 `GTDatabase` 基础对象。

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
      GTDatabase::RunSdcConstantSimulation()
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
| timing raw db | `cpp_to_py/gputimer/db/GTDatabase.cpp:1087` | `TimingTorchRawDB::TimingTorchRawDB` | 生成 node/pin/net flat maps 和当前坐标 tensor |
| graph db | `cpp_to_py/gputimer/db/GTDatabase.cpp:623` | `GTDatabase::GTDatabase` | 持有 rawdb/gpdb/timing_raw_db 引用和 min/max Liberty |
| SDC target map | `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:35` | `preparePinNameMapForSdc` | 只为 SDC 需要的 pin name 建目标集合 |
| timing graph | `cpp_to_py/gputimer/db/GTDatabase.cpp:948` | `ExtractTimingGraph` | host-side 构建 pins/arcs/tests/pin arc lists/tensors |
| SDC values | `cpp_to_py/gputimer/db/GTDatabase_sdc.cpp:68` | `readSdc` | 生成 `pinSlew/pinLoad/pinRAT/pinAT` 和 sparse clock id/table state |
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

真正构造在 `cpp_to_py/gputimer/db/GTDatabase.cpp:1087`。

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

4. 生成 prefix-offset flat maps：

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
gtdb->RunSdcConstantSimulation()

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
sdc_constant_simulation
construct_gputimer
read_flute_lut
```

审查点：

- `readSdc()` 必须在 `ExtractTimingGraph()` 后面，因为它依赖 `primary_inputs/outputs`、`test_id2_arc_id`、`timing_arc_from_pin_id`、`pin_is_clk` 等 graph 结果。
- `ExtractTimingGraph()` 不再接收 SDC，也不预读 `set_case_analysis`；它只构建完整 structural graph。
- `RunSdcConstantSimulation()` 在 `readSdc()` 后执行，是后续放置 SDC case analysis / Liberty constant function propagation / timing-disabled arc mask 的入口。当前版本只保留入口，不改变 graph 或 CUDA tensor。
- `gt::GPUTimer` 构造必须在 `readSdc()` 后面，因为 constructor 会对 `timing_raw_db.pinSlew/pinLoad/pinRAT/pinAT` 直接取 `data_ptr<float>()`。
- direct route-segment timing 下不读 Flute LUT，也跳过 legacy RC tensors；如果后续调用旧 `update_rc` 路径，需要检查空指针。
