# 03_BUILD_ROUTING_RC.md

Last reviewed: 2026-06-08

本文展开 `00_POWER_ARCHITECTURE.md` 里的 `build_rc stage`：

```text
build_rc stage
  gt::GPUTimer::update_states()
  gt::GPUTimer::init_dmp_rc_route_segments(route_segments)
```

目标是从 Python 验收入口一直审查到最底层 C++/CUDA 端口。这个阶段只构建 direct `--route_segments` 的 DMP RC 模型；最终 AT/RAT/slack/slew/load 的 timing propagation 在后续 `timer stage` 的 `gt::GPUTimer::update_timing_dmp()` 中完成。

## 1. 当前入口

当前 power/timing 对比脚本的 `build_rc` 入口是：

```text
tools/compare_ispd25_route_power_timing.py::run_xplace_worker(...)
  params["route_segments"] = timer_args.route_segments
  gputimer = timer_only.timing_opt.GPUTimer(...)

  build_rc()
    gputimer.timer.update_states()
    gputimer.timer.init_dmp_rc_route_segments(timer_args.route_segments)
```

普通 `run_timer.py --route_segments ...` 入口走 wrapper：

```text
run_timer.py::main()
  gputimer.update_timing_dmp_route_segments(args.route_segments)
    timer_only/timing_opt.py::GPUTimer.update_timing_dmp_route_segments(...)
      self.timer.update_states()
      self.timer.init_dmp_rc_route_segments(route_segments_file)
      self.timer.update_timing_dmp()
```

所以审查时要区分：

- `compare_ispd25_route_power_timing.py` 把 RC 构建和 timing propagation 拆成两个 stage 计时。
- `run_timer.py` 的 wrapper 一次性做 `update_states -> init_dmp_rc_route_segments -> update_timing_dmp`。
- 旧记录里可能把 `set_ideal_clock(true)` 写进 Xplace `build_rc` stage；当前代码没有这个 Python 调用。ideal clock 状态来自 SDC/GTDatabase 初始化后上传到 timer/device。

## 2. 一眼看完整链路

```text
Python worker
  tools/compare_ispd25_route_power_timing.py::build_rc()
    pybind GPUTimer.update_states()
      cpp_to_py/gputimer/core/GPUTimer.cu
        gt::GPUTimer::update_states()
          cudaMemset optional RC/timing scratch arrays
          reset_val<<<...>>> timing arrays
          device_copy_batch<<<...>>> baseline timing labels back to live arrays
          cudaDeviceSynchronize()

    pybind GPUTimer.init_dmp_rc_route_segments(file)
      cpp_to_py/gputimer/core/DmpModel.cpp
        gt::GPUTimer::init_dmp_rc_route_segments(file)
          build_openroad_route_segments_rc(file)
            cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp
              mmap route segment file
              optional route-segment cache load
              copy flat_net2pin maps to host
              build net name index
              resolve timer pins to rawdb pins
              scan route blocks
              parse route rows into LocalRcNetGraph
              infer route grid
              attach pins / stubs
              root by driver, repair disconnected nodes, prune to rooted tree
              materialize HostRcGraph
              optional cache save
          initialize_dmp_rc_explicit(...)
            cpp_to_py/gputimer/core/rc/DmpRc.cu
              DmpModel::initialize_rc_explicit(...)
                cudaMalloc/cudaMemcpy explicit graph arrays
                cudaMalloc/cudaMemset RC build scratch
              optional scale_explicit_edge_res_kernel<<<...>>>
              cudaMalloc/cudaMemcpy DmpModel descriptor to device
          release_host_rc_graph_storage(graph)
          calc_res_cap_dmp(dmp_db, num_nets)
            calc_dmp_rc<<<num_nets, 64>>>
            prepare_dmp_rc_propagation_fields(...)
          propagate_rc_tree_dmp(dmp_db, num_nets)
            propagate_rc_dmp<<<ceil(num_nets/64), (64, NUM_ATTR)>>>
          dmp_prepare_timing_after_rc(h_dmp_db, dmp_db)
            release RC transient arrays
            allocate timing scratch
            copy descriptor back to device
          apply_dmp_driving_cell_source_slew(...)
            applyDrivingCellSourceSlewKernel<<<...>>>
```

`NUM_ATTR = 4`，顺序是 early-rise、early-fall、late-rise、late-fall。

## 3. Python 和 pybind 端口

| 层 | 文件 | 入口 | 作用 |
| --- | --- | --- | --- |
| CLI 参数 | `run_timer.py` | `--route_segments` | OpenROAD `write_global_route_segments` 输出文件。 |
| 验收 worker | `tools/compare_ispd25_route_power_timing.py` | `build_rc()` | 只计时 `update_states()` 和 `init_dmp_rc_route_segments(...)`。 |
| Python wrapper | `timer_only/timing_opt.py` | `update_timing_dmp_route_segments(...)` | 普通 run path 下还会继续调用 `update_timing_dmp()`。 |
| pybind | `cpp_to_py/gputimer/PyBindCppMain.cpp` | `.def("update_states", ...)` | 暴露 C++ `gt::GPUTimer::update_states()`。 |
| pybind | `cpp_to_py/gputimer/PyBindCppMain.cpp` | `.def("init_dmp_rc_route_segments", ..., py::arg("file"))` | 暴露 C++ route segment RC 初始化。 |

`create_gputimer(...)` 的 direct RC 相关行为：

```text
direct_rc_mode = kwargs.contains("route_segments") || kwargs.contains("gr_rc")
gtdb->skip_legacy_rc_tensors = direct_rc_mode
if (!direct_rc_mode) read FLUTE LUT
```

这意味着 `route_segments` 先作为 direct RC mode 开关影响 graph/timer 初始化；真正读 route 文件是在后续 `init_dmp_rc_route_segments(file)`。

## 4. `update_states()` 做什么

文件：`cpp_to_py/gputimer/core/GPUTimer.cu`

端口：

```cpp
void gt::GPUTimer::update_states();
```

当前实现是 CUDA state reset，不解析 RC，不读 route 文件。

具体动作：

- 如果以下数组非空，清零每个 pin/corner 槽位：
  - `pinImpulse`
  - `pinRootRes`
  - `pinRootDelay`
  - `pinWireCap`
- 通过 `reset_val<<<...>>>` 重置 timing/test 前缀数组：
  - `arcDelay[2 * num_arcs * NUM_ATTR]`
  - `testRelatedAT[num_tests * NUM_ATTR]`
  - `testRAT[num_tests * NUM_ATTR]`
  - `testConstraint[num_tests * NUM_ATTR]`
  - `at_prefix_pin[num_pins * NUM_ATTR]`
  - `at_prefix_arc[num_pins * NUM_ATTR]`
  - `at_prefix_attr[num_pins * NUM_ATTR]`
- 如果 `__pinSlew__` baseline 存在，用 `device_copy_batch<<<...>>>` 恢复 live label：
  - `__pinSlew__ -> pinSlew`
  - `__pinLoad__ -> pinLoad`
  - `__pinRAT__ -> pinRAT`
  - `__pinAT__ -> pinAT`
- 最后 `cudaDeviceSynchronize()`。

审查重点：

- 这是 `.cu` 文件，允许 CUDA runtime 和 kernel launch。
- `.cpp` 侧不应该新增 CUDA runtime 头或 kernel launch。
- direct route mode 下，部分 optional pointer 可能为空，所以这里已有空指针 guard。

## 5. C++ route RC 主入口

文件：`cpp_to_py/gputimer/core/DmpModel.cpp`

端口：

```cpp
void gt::GPUTimer::init_dmp_rc_route_segments(const std::string& file);
```

逐行语义：

```text
DmpRcStageProfile rc_profile(DMP_RC_PROFILE)
HostRcGraph graph = build_openroad_route_segments_rc(file)
record graph.num_nodes / graph.num_edges for progress log
initialize_dmp_rc_explicit(
  graph.edge_from,
  graph.edge_to,
  graph.net2node_start,
  graph.net2edge_start,
  graph.node2pin,
  graph.edge_res,
  graph.node_cap,
  graph.includes_pin_caps,
  num_nets,
  graph.num_nodes,
  graph.num_edges)
release_host_rc_graph_storage(graph)
calc_res_cap_dmp(dmp_db, num_nets)
propagate_rc_tree_dmp(dmp_db, num_nets)
dmp_prepare_timing_after_rc(h_dmp_db, dmp_db)
apply_dmp_driving_cell_source_slew(*this)
```

注意这里的 `graph` 是 host 侧临时对象；`initialize_dmp_rc_explicit(...)` 上传后会立即释放 host vector storage，降低峰值内存。

## 6. `HostRcGraph` 端口

声明位置：`cpp_to_py/gputimer/core/GPUTimer.h`

```cpp
struct HostRcGraph {
    std::vector<int> edge_from;
    std::vector<int> edge_to;
    std::vector<float> edge_res;
    std::vector<float> node_cap;
    std::vector<int> net2node_start;
    std::vector<int> net2edge_start;
    std::vector<int> node2pin;
    std::vector<std::string> node_names;
    std::vector<uint8_t> includes_pin_caps;
    int skipped_loop_edges = 0;
    int repaired_edges = 0;
    int num_nodes = 0;
    int num_edges = 0;
};
```

字段约定：

| 字段 | shape | 单位/索引 | 含义 |
| --- | --- | --- | --- |
| `edge_from` | `[num_edges]` | global RC node id | 每条 RC edge 的 from node。 |
| `edge_to` | `[num_edges]` | global RC node id | 每条 RC edge 的 to node。 |
| `edge_res` | `[num_edges]` | GPUTimer resistance unit | 显式 edge resistance，route parser 已除以 `gtdb.res_unit`。 |
| `node_cap` | `[num_nodes * NUM_ATTR]` | GPUTimer cap unit | 每个 RC node 的四 corner cap。 |
| `net2node_start` | `[num_nets + 1]` | prefix sum | net `i` 的 nodes 是 `[start[i], start[i+1])`。 |
| `net2edge_start` | `[num_nets + 1]` | prefix sum | net `i` 的 edges 是 `[start[i], start[i+1])`。 |
| `node2pin` | `[num_nodes]` | timer pin id 或 `-1` | pin node 指向 timer pin，纯 route node 为 `-1`。 |
| `node_names` | optional | debug only | 只有 `GPUTIMER_ROUTE_SEG_KEEP_NODE_NAMES=1` 时保留。 |
| `includes_pin_caps` | `[num_nets]` | bool byte | 当前 route segment path 默认不把所有 pin caps 纳入文件语义，高 fanout fallback 可能把缺失 load pin cap 汇总到 driver node。 |
| `skipped_loop_edges` | scalar | count | prune rooted tree 时丢弃的 loop/non-tree edges。 |
| `repaired_edges` | scalar | count | 为连接到 root 而补的 zero-ohm repair edges。 |

`edge_res` 和 `node_cap` 是 route-gradient/FD 验证里最稳定的扰动入口，因为它们已经是解析后的显式 RC 图。

## 7. Route Segment Host Builder

主文件：`cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp`

共享 helper：

| 文件 | 负责内容 |
| --- | --- |
| `openroad/OpenroadRcInternal.h` | route RC 内部 struct、helper 声明、cache 端口声明。 |
| `openroad/OpenroadRcCache.cpp` | route segment graph cache metadata、binary I/O、design signature。 |
| `openroad/OpenroadRcParse.cpp` | name alias、token/int/layer/row parser。 |
| `openroad/OpenroadRcGeometry.cpp` | route grid、track spacing、gcell tile、grid origin 推断。 |
| `openroad/OpenroadRcGraphUtil.cpp` | LocalRcNetGraph node/edge helper、root reorder、repair/prune。 |
| `openroad/OpenroadRcPin.cpp` | timer pin 到 rawdb pin 解析，以及 pin route location。 |
| `openroad/OpenroadRouteSegmentsBuilder.cpp` | saved route segment parser 和 HostRcGraph materialization。 |

主端口：

```cpp
HostRcGraph gt::GPUTimer::build_openroad_route_segments_rc(const std::string& file);
```

输入：

- `file`: OpenROAD saved global-route segment file path。
- `gtdb.net_names`, `gtdb.pin_names`: 用于 route block header 和 pin name 映射。
- `gtdb.rawdb`: layers、tracks、die/gcell geometry、raw pins。
- `timing_raw_db.flat_net2pin_start_map`, `timing_raw_db.flat_net2pin_map`: net -> timer pins。
- `num_nets`, `num_pins`, `num_threads`, `gtdb.res_unit`, `gtdb.cap_unit`。

输出：

- `HostRcGraph graph`，后续直接上传给 DMP explicit RC。

### 7.1 Cache 端口

cache helper 在 `OpenroadRcCache.cpp`：

```cpp
RouteSegmentCacheMeta route_segment_cache_meta(source_file);
std::string route_segment_cache_path(source_file);
std::uint64_t route_segment_design_signature(gtdb, num_nets, num_pins);
bool load_route_segment_cache(..., HostRcGraph& graph);
void save_route_segment_cache(..., const HostRcGraph& graph);
```

cache 启用条件：

```text
cache_enabled =
  !GPUTIMER_ROUTE_SEG_PROFILE
  && GPUTIMER_DEBUG_ROUTE_PIN_NET is empty
  && !GPUTIMER_ROUTE_SEG_KEEP_NODE_NAMES
  && !GPUTIMER_ROUTE_SEG_DISABLE_CACHE
```

cache key/check 内容：

- route source file size。
- route source file mtime。
- cache version/magic，目前 `XPRSEGRCACHE05` / version 5。
- `num_nets`、`num_pins`。
- `missing_high_fanout_skip`。
- `design_signature`，混入：
  - expected `num_nets`
  - expected `num_pins`
  - `gtdb.res_unit`
  - `gtdb.cap_unit`
  - all net names
  - all pin names

cache binary 保存这些 vectors：

- `edge_from`
- `edge_to`
- `edge_res`
- `node_cap`
- `net2node_start`
- `net2edge_start`
- `node2pin`
- `includes_pin_caps`

审查时要注意：cache 是加速 repeated run 的路径；新 case/cold-start 仍然要读 route text、解析和 materialize。

### 7.2 Route file scan

文件读取：

```text
RouteSegmentMappedFile(file)
  open(...)
  fstat(...)
  mmap(..., PROT_READ, MAP_PRIVATE)
```

扫描阶段：

- 把 `timing_raw_db.flat_net2pin_*` 拷到 CPU contiguous tensor。
- 用 `RouteNetNameIndex` 对 `gtdb.net_names` 建 open addressing table。
- 对 rawdb routing layers 建 `layer_name_to_level`，值为 `layer.rIndex + 1`。
- 调 `resolve_openroad_timer_pins(...)` 建 `pin_id_to_dbpin`。
- 多线程扫描 route file：
  - 线程数默认 `num_threads`。
  - 可用 `GPUTIMER_ROUTE_SEG_SCAN_THREADS` 覆盖。
  - 如果 `GPUTIMER_DEBUG_ROUTE_PIN_NET` 非空，则强制单线程。
- scan 只识别 route block header：
  - 非空、非注释。
  - 单 token。
  - 不是单独的 `(` 或 `)`。
  - token 能 resolve 到 `gtdb.net_names`。

scan 输出是 `RouteSegmentBlock{net_idx, begin, end}`。每个 block 后续独立 parse。

### 7.3 Segment row parser

每条有效 route row 的格式：

```text
x1 y1 layer1 x2 y2 layer2
```

解析函数：

```cpp
parse_route_segment_row_range(...)
resolve_route_layer_token(layer_name_to_level, ...)
```

规则：

- 非 Manhattan row 直接计入 `non_manhattan_segments`，最后会报错。
- unknown layer 计入 `unknown_layers`，最后会报错。
- malformed row 计入 `malformed_rows`，最后会报错。
- route grid stats 会记录 row endpoints，后续用于推断 OpenROAD route grid。

wire/via RC 转换：

```text
length_dbu = abs(x1 - x2) + abs(y1 - y2)
length_um = length_dbu / rawdb.DBU_Micron
```

如果 `length_dbu == 0`：

```text
edge_res = nangate45_via_res_ohm(min(layer1, layer2)) / gtdb.res_unit
add_edge(local, from, to, edge_res)
```

如果同层 Manhattan wire：

```text
rc = nangate45_layer_rc(layer)
edge_res = rc.res_ohm_per_um * length_um / gtdb.res_unit
cap = rc.cap_f_per_um * length_um / gtdb.cap_unit
add_edge(local, from, to, edge_res)
add_attr_cap(local.node_cap, from, cap * 0.5)
add_attr_cap(local.node_cap, to, cap * 0.5)
```

parse 多线程：

- 默认 `num_threads`。
- 可用 `GPUTIMER_ROUTE_SEG_PARSE_THREADS` 覆盖。
- debug pin net 或 duplicate route blocks 时强制单线程。
- `#pragma omp parallel for ... schedule(dynamic, 256)`。

### 7.4 Grid inference

相关文件：`OpenroadRcGeometry.cpp`

主要端口：

```cpp
int openroad_gcell_tile_size(const db::Database& rawdb);
OpenroadInferredGrid infer_openroad_route_grid_from_stats(...);
```

语义：

- fallback tile size 来自 routing layer track spacing，当前按 OpenROAD-like gcell size 推断。
- route row endpoints 的 gcd step 用于反推 tile size。
- origin 由 first coordinate 和 die origin 对齐。
- 如果不能得到 positive tile size，会 throw。

这个 grid 后续用于把 rawdb pin shape/location 映射到 route grid center。

### 7.5 Pin resolution 和 pin stub

相关文件：`OpenroadRcPin.cpp`

端口：

```cpp
OpenroadPinMapStats resolve_openroad_timer_pins(
  const GTDatabase& gtdb,
  int num_pins,
  std::vector<db::Pin*>& pin_id_to_dbpin,
  int threads);

bool openroad_pin_route_loc(
  const GTDatabase& gtdb,
  const std::vector<db::Pin*>& pin_id_to_dbpin,
  const std::vector<const db::Layer*>& routing_level_to_layer,
  const OpenroadInferredGrid& openroad_grid,
  int pin_id,
  OpenroadPinRouteLoc& loc);
```

`resolve_openroad_timer_pins(...)` 优先用 rawdb pin 的 `gpdb_id` 直接填 `pin_id_to_dbpin`。如果有 unresolved pin，再建 name alias map，支持 `cell:pin`、`cell/pin`、last separator 替换以及 normalized name。

finalize 每个 net 时：

- 从 `flat_net2pin_start_map` 找 net fanout 和 driver pin。
- 对每个 timer pin 计算 `OpenroadPinRouteLoc`。
- 如果 route graph 里缺 pin node，则 `append_pin_node(...)`。
- 如果能找到对应 route grid node，则加 pin stub edge。
- pin stub RC：
  - edge resistance 由 pin 到 route grid center 的 Manhattan length 和 route layer RC 得到。
  - 如连接到 `conn_layer + 1`，再加一段 via resistance。
  - stub cap 平分到 pin node 和 route node。

### 7.6 Missing nets 和 high-fanout skip

环境变量：

```text
GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP
```

默认值：`300`。

规则：

- route file 里不存在该 net，且 fanout <= 1：跳过 unconnected missing net。
- route file 里不存在该 net，且 `fanout > missing_high_fanout_skip`：
  - 如果 skip 阈值大于 0，则走 high-fanout fallback。
  - 创建 driver node。
  - 把 load pin library cap 累加到 driver node 的 `node_cap`。
  - 不构建完整 route edge。
- 其余 missing net 继续走 fallback pin handling。

这个策略直接影响 timing/power 对齐，审查 stage timing 时也要把 skip policy 写入结论。

### 7.7 Rooting、repair、prune、materialize

相关文件：`OpenroadRcGraphUtil.cpp`

关键 helper：

```cpp
append_route_node(...)
append_pin_node(...)
add_edge(...)
reorder_root(local, driver_node)
build_flat_local_adjacency(local)
prune_to_rooted_tree(local)
```

finalize 每个 net 的后处理：

1. 确保 driver pin 是 local node。
2. `reorder_root(local, driver_node)`，让 driver 成为 node 0。
3. 从 root 做连通性检查。
4. 对未连通节点补 `0.0` ohm repair edge 到 root。
5. `prune_to_rooted_tree(local)`，只保留从 root 可达树上的边。
6. 记录 `net_node_count` / `net_edge_count`。

materialize：

```text
prefix sum net_node_count -> graph.net2node_start
prefix sum net_edge_count -> graph.net2edge_start
resize graph edge/node vectors
parallel copy each LocalRcNetGraph into global HostRcGraph
  global edge node id = node_base + local node id
  graph.edge_res copies contiguous range
  graph.node2pin copies contiguous range
  graph.node_cap copies contiguous NUM_ATTR range
reset local_nets / route_node_maps
```

线程数：

- finalize 默认 `num_threads`，可用 `GPUTIMER_ROUTE_SEG_FINALIZE_THREADS` 覆盖。
- materialize 默认沿用 finalize threads，可用 `GPUTIMER_ROUTE_SEG_MATERIALIZE_THREADS` 覆盖。

如果 `malformed_rows > 0 || unknown_layers > 0 || non_manhattan_segments > 0`，builder 最后 throw，不会返回 graph。

## 8. CUDA explicit RC 初始化

文件：`cpp_to_py/gputimer/core/rc/DmpRc.cu`

外层端口：

```cpp
void gt::GPUTimer::initialize_dmp_rc_explicit(
  const std::vector<int>& host_edge_from,
  const std::vector<int>& host_edge_to,
  const std::vector<int>& host_flat_net2node_start_map,
  const std::vector<int>& host_flat_net2edge_start_map,
  const std::vector<int>& host_node2pin_map,
  std::vector<float>& host_edge_res,
  const std::vector<float>& host_node_cap,
  const std::vector<uint8_t>& host_includes_pin_caps,
  int num_nets,
  int num_nodes,
  int num_edges);
```

动作：

```text
rc_time_factor = (res_unit * cap_unit) / time_unit()
h_dmp_db = new DmpModel(this)
h_dmp_db->initialize_rc_explicit(...)
if num_edges > 0 and rc_time_factor != 1:
  scale_explicit_edge_res_kernel<<<ceil(num_edges/256), 256>>>(h_dmp_db->edge_res, num_edges, rc_time_factor)
cudaMalloc(&dmp_db, sizeof(DmpModel))
cudaMemcpy(dmp_db, h_dmp_db, sizeof(DmpModel), H2D)
```

内部端口：

```cpp
__host__ void DmpModel::initialize_rc_explicit(...);
```

它分配并上传：

| device field | source | shape |
| --- | --- | --- |
| `edge_from` | `host_edge_from` | `[num_edges]` |
| `edge_to` | `host_edge_to` | `[num_edges]` |
| `flat_net2node_start_map` | `host_flat_net2node_start_map` | `[num_nets + 1]` |
| `flat_net2edge_start_map` | `host_flat_net2edge_start_map` | `[num_nets + 1]` |
| `node2pin_map` | `host_node2pin_map` | `[num_nodes]` |
| `edge_res` | `host_edge_res` | `[num_edges]` |
| `node_cap` | `host_node_cap` | `[num_nodes * NUM_ATTR]` |
| `includes_pin_caps` | `host_includes_pin_caps` | `[num_nets]` |

并分配 RC build scratch：

- `root_dist[num_nodes]`
- `cnts[num_nodes]`
- `node_order[num_nodes]`
- `parent_node[num_nodes]`
- `res_parent[num_nodes]`

显式 route RC path 下：

```text
explicit_rc = true
unit_to_micron = 1.0
rf = 0.0
cf = 0.0
edge_wl = nullptr
```

所以 CUDA kernel 不再从 wirelength 推 R/C，而是直接消费 `edge_res` 和 `node_cap`。

## 9. CUDA RC kernels

### 9.1 `calc_res_cap_dmp`

文件：`cpp_to_py/gputimer/core/rc/DmpRc.cu`

wrapper：

```cpp
void calc_res_cap_dmp(DmpModel* dmp_rc_, int num_nets);
```

launch：

```cpp
calc_dmp_rc<<<num_nets, 64>>>(dmp_rc_);
cudaDeviceSynchronize();
prepare_dmp_rc_propagation_fields(dmp_rc_);
```

device entry：

```cpp
__global__ void calc_dmp_rc(DmpModel* dmp_rc_) {
  dmp_rc_->calc_dmp_rc();
}
```

per-net semantics：

- `blockIdx.x` 是 net id。
- root 是该 net 的第一个 node，也就是 finalize 后的 driver node。
- 从 root 开始逐层扩展。
- 对每条 edge：
  - 在 explicit RC 下直接读 `edge_res[i]`。
  - 发现新 node 后写：
    - `root_dist[node]`
    - `parent_node[node]`
    - `res_parent[node]`
    - `node_order[...]`
- 非 explicit RC 才会用 `edge_wl/rf/cf` 生成 cap/res；route segment path 不走这条分支。

`prepare_dmp_rc_propagation_fields(...)` 会：

- 把 `DmpModel` descriptor 从 device 拷回 host。
- 若不保留 build graph，释放 build-only arrays：
  - `edge_from`
  - `edge_to`
  - `flat_net2edge_start_map`
  - `edge_wl`
  - `edge_res`
  - `root_dist`
  - `cnts`
- 分配 propagation/timing RC arrays：
  - `y1[num_nodes * NUM_ATTR]`
  - `y2[num_nodes * NUM_ATTR]`
  - `y3[num_nodes * NUM_ATTR]`
  - optional `node_delay[num_nodes * NUM_ATTR]`
  - optional `down_cap[num_nodes * NUM_ATTR]`
  - `elmore_delay[num_pins * NUM_ATTR]`
  - `C1[dmp_slot_capacity]`
  - `C2[dmp_slot_capacity]`
  - `r_pi[dmp_slot_capacity]`
- 把更新后的 descriptor 拷回 device。

保留 build graph 的条件：

```text
dmp_db->debug_on
or DMP_DEBUG_TIMING
or DMP_KEEP_RC_BUILD_GRAPH
```

### 9.2 `propagate_rc_tree_dmp`

wrapper：

```cpp
void propagate_rc_tree_dmp(DmpModel* dmp_rc_, int num_nets);
```

launch：

```cpp
thread_count = 64
block_size = dim3(64, NUM_ATTR)
num_blocks = ceil(num_nets / 64)
propagate_rc_dmp<<<num_blocks, block_size>>>(dmp_rc_);
```

device entry：

```cpp
__global__ void propagate_rc_dmp(DmpModel* dmp_rc_) {
  dmp_rc_->propagate_dmp_rc();
}
```

per-net/per-corner semantics：

- `idx = blockIdx.x * blockDim.x + threadIdx.x` 是 net id。
- `cond = threadIdx.y` 是 corner id。
- 从 `node_order` 的末端向 root 反向累积：
  - `wire_cap = node_cap[node * NUM_ATTR + cond]`
  - 如果 node 是 pin，且 `includes_pin_caps` false，则加 `pin_cap_attr(pinCap, pin, cond)`。
  - 累积 `y1/y2/y3` moments。
- 在 root pin 处生成 PI model：
  - 正常：`C1`, `C2`, `r_pi`
  - 奇异/非法 fallback：`C1 = max(y1_root, 0)`, `C2 = 0`, `r_pi = 0`
- 从 root 向外计算 Elmore delay：
  - `delay[node] = delay[parent] + down_cap[node] * res_parent[node]`
  - pin node 写入 `elmore_delay[pin * NUM_ATTR + cond]`

`propagate_rc_tree_dmp` 当前没有无条件 `cudaDeviceSynchronize()`；如果启用 `DMP_PROFILE_KERNELS`，event synchronize 会同步该 kernel。正常路径下，紧随其后的 `dmp_prepare_timing_after_rc(...)` 会先 D2H 拷贝 `DmpModel` descriptor，这个 `cudaMemcpy` 会等待前面的 propagation kernel 完成。

## 10. RC 后处理端口

### 10.1 `dmp_prepare_timing_after_rc`

文件：`cpp_to_py/gputimer/core/DmpModel.cu`

端口：

```cpp
void dmp_prepare_timing_after_rc(DmpModel* h_dmp_db, DmpModel* dmp_db);
```

动作：

1. D2H 拷贝 device `DmpModel` descriptor。
2. 用 device descriptor 覆盖 host descriptor，但保留 host ownership flag。
3. `h_dmp_db->release_rc_transient()`：
   - 释放 graph/build/propagation transient arrays：
     - `edge_from`, `edge_to`
     - `flat_net2node_start_map`, `flat_net2edge_start_map`
     - `node2pin_map`
     - `edge_wl`, `edge_res`
     - `node_cap`, `includes_pin_caps`
     - `root_dist`, `cnts`, `node_order`
     - `parent_node`, `res_parent`
     - `node_delay`, `y1`, `y2`, `y3`, `down_cap`
   - 不释放 `C1`, `C2`, `r_pi`, `elmore_delay`。
4. `h_dmp_db->allocate_timing_scratch()`：
   - 分配 `pin_at_winner` 等 timing scratch。
5. H2D 拷回更新后的 descriptor。

### 10.2 `apply_dmp_driving_cell_source_slew`

文件：

- host wrapper: `cpp_to_py/gputimer/core/DmpModel.cpp`
- CUDA wrapper/kernel: `cpp_to_py/gputimer/core/DmpGateEval.cu`

host wrapper：

```text
if gtdb.driving_cell_sources.empty(): return
pack pin_ids, timing_ids, input_rfs, input_slews
apply_dmp_driving_cell_source_slew_cuda(dmp_db, ...)
```

CUDA wrapper：

```cpp
void apply_dmp_driving_cell_source_slew_cuda(
  DmpModel* dmp_db,
  const std::vector<int>& pin_ids,
  const std::vector<int>& timing_ids,
  const std::vector<int>& input_rfs,
  const std::vector<float>& input_slews);
```

它会临时分配并上传 packed arrays，然后 launch：

```cpp
applyDrivingCellSourceSlewKernel<<<DMP_TIMING_BLOCK_NUMBER(total), DMP_TIMING_BLOCK_SIZE>>>(...)
```

作用是把 `set_driving_cell` source slew 写回 DMP timing 状态。`init_dmp_rc_route_segments(...)` 末尾调用一次，`update_timing_dmp()` 开头也会再调用一次。

## 11. build_rc 输出给后续 stage 什么

`build_rc` 结束后，应具备：

- `GPUTimer::dmp_db` 指向 device `DmpModel` descriptor。
- `GPUTimer::h_dmp_db` 指向 host mirror descriptor。
- `DmpModel` 中保留给 timing/power 后续使用的核心 RC/timing arrays：
  - `C1`
  - `C2`
  - `r_pi`
  - `elmore_delay`
  - `pin_at_winner` timing scratch
- RC graph transient arrays 已释放，除非 debug/env 要求保留。
- 最终 timing labels 还没有在 compare worker 的 `build_rc` stage 里传播；后续 `timer stage` 的 `update_timing_dmp()` 才会写 `pinAT/pinRAT/pinSlew/pinLoad/arcDelay` 等 labels。
- power stage 依赖的是 timer stage 后的 labels 和 DMP RC model，而不是 raw `HostRcGraph`。

## 12. Profiling 和 debug 开关

| 环境变量 | 影响 |
| --- | --- |
| `XPLACE_TIMER_PROFILE` | pybind `create_gputimer(...)` 阶段计时，不直接包 route RC parser。 |
| `DMP_RC_PROFILE` | `init_dmp_rc_route_segments` 子阶段计时，并开启 route cache profile。 |
| `GPUTIMER_ROUTE_SEG_PROFILE` | route text scan/parse/finalize/materialize 详细计时；同时禁用 route cache。 |
| `GPUTIMER_ROUTE_SEG_CACHE_PROFILE` | route cache meta/path/signature/load 子阶段计时。 |
| `DMP_PROFILE_KERNELS` | CUDA RC kernel event timing 和 kernel work summary。 |
| `DMP_DEBUG_TIMING` | 增加 DMP debug/profiling，并阻止释放部分 build graph。 |
| `DMP_KEEP_RC_BUILD_GRAPH` | 保留 RC build graph/device arrays，方便 debug dump，但显存更高。 |
| `GPUTIMER_ROUTE_SEG_DISABLE_CACHE` | 禁用 route segment cache，适合 first-run/no-cache 验证。 |
| `GPUTIMER_ROUTE_SEG_CACHE_DIR` | 覆盖 cache 目录，默认 `result/route_segment_cache`。 |
| `GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP` | 缺失高扇出 net fallback 阈值，默认 `300`。 |
| `GPUTIMER_ROUTE_SEG_SCAN_THREADS` | 覆盖 route block scan 线程数。 |
| `GPUTIMER_ROUTE_SEG_PARSE_THREADS` | 覆盖 segment parse 线程数。 |
| `GPUTIMER_ROUTE_SEG_FINALIZE_THREADS` | 覆盖 finalize net 线程数。 |
| `GPUTIMER_ROUTE_SEG_MATERIALIZE_THREADS` | 覆盖 HostRcGraph materialize 线程数。 |
| `GPUTIMER_DEBUG_ROUTE_PIN_NET` | 打印某个 net 的 pin/grid debug，并强制相关 path 单线程。 |
| `GPUTIMER_ROUTE_SEG_KEEP_NODE_NAMES` | 保留 HostRcGraph node names，禁用 cache，增加内存。 |
| `DMP_PROGRESS` / `XPLACE_TIMER_VERBOSE` | 打印 DMP route RC progress。 |

典型 no-cache 审查命令组合：

```bash
GPUTIMER_ROUTE_SEG_DISABLE_CACHE=1 \
GPUTIMER_ROUTE_SEG_PROFILE=1 \
DMP_RC_PROFILE=1 \
DMP_PROFILE_KERNELS=1 \
python run_timer.py --designName <design> --route_segments <path.route_segments>
```

注意：`GPUTIMER_ROUTE_SEG_PROFILE=1` 自身就会禁用 route cache；如果目的是测 cache hit，就不要打开它，只开 `DMP_RC_PROFILE=1` 或 `GPUTIMER_ROUTE_SEG_CACHE_PROFILE=1`。

## 13. 审查清单

读代码时建议按这个顺序：

1. `tools/compare_ispd25_route_power_timing.py`
   - 确认当前 stage 边界只有 `update_states()` 和 `init_dmp_rc_route_segments()`。
2. `timer_only/timing_opt.py`
   - 对比普通 `run_timer.py` wrapper，确认是否把 `update_timing_dmp()` 包进同一调用。
3. `cpp_to_py/gputimer/PyBindCppMain.cpp`
   - 确认 pybind 名字和 C++ method 名字一致。
4. `cpp_to_py/gputimer/core/GPUTimer.cu`
   - 审查 `update_states()` 的 CUDA reset/copy 是否覆盖了新增 timing arrays。
5. `cpp_to_py/gputimer/core/DmpModel.cpp`
   - 审查 `init_dmp_rc_route_segments()` 调用顺序和 host graph 生命周期。
6. `cpp_to_py/gputimer/core/openroad/OpenroadRouteSegmentsBuilder.cpp`
   - 审查 route file parsing、missing net policy、pin attach 和 HostRcGraph materialization。
7. `cpp_to_py/gputimer/core/openroad/OpenroadRcCache.cpp`
   - 审查 cache key 是否包含会改变 graph 语义的输入。
8. `cpp_to_py/gputimer/core/rc/DmpRc.cu`
   - 审查 explicit graph upload、unit scaling、`calc_dmp_rc`、`propagate_rc_dmp`。
9. `cpp_to_py/gputimer/core/DmpModel.cu`
   - 审查 RC transient release 和 timing scratch allocation。
10. `cpp_to_py/gputimer/core/DmpGateEval.cu`
    - 审查 `set_driving_cell` source slew 是否正确应用。

## 14. 常见误读

- `build_rc` 不等于完整 timing update。compare worker 里 timing update 在后续 `timer` stage。
- `HostRcGraph` 是 host 临时显式 RC 图；`build_rc` 完成后默认已经释放 host vectors 和多数 device transient arrays。
- route segment path 的 R/C 主要来自 `nangate45_layer_rc(...)` 和 `nangate45_via_res_ohm(...)`，不是 legacy FLUTE wire RC。
- `edge_res` 在 HostRcGraph 里先按 `gtdb.res_unit` 归一化，上传后可能再按 `(res_unit * cap_unit) / time_unit()` 做 timing unit scale。
- cache hit 的 `build_rc` 时间可能主要花在 design signature 或 cache read，不代表 CUDA RC kernel 慢。
- 打开 debug node names、route profile 或 disable cache 会改变性能路径，不能和 cache-hit stage timing 混报。
