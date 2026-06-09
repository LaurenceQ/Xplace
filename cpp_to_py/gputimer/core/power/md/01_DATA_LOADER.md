# 01_DATA_LOADER.md

Last reviewed: 2026-06-08

本文只展开 `run_xplace_worker(args)` 里 data loader 这段：

```text
run_xplace_worker(args)
  run_timer.getArgs()
  load_design(...)
  data.to_timing_device(...).preprocess_timing()
```

目标是从 Python 入口审查到 data loader 真实进入的最底层 C++ cpybin 端口：`io_parser.start(...) -> start_all(...) -> Database::load/setup -> GPDatabase::setup`，以及 tensor 抽取时的 `GPDatabase::get*Tensor()`。

## 1. 总调用图

```text
tools/compare_ispd25_route_power_timing.py::run_xplace_worker(args)
  Flute.register(8)
  sys.argv = [run_timer-style args]
  timer_args = run_timer.getArgs()
  logger = setup_logger(timer_args, sys.argv)
  set_random_seed(timer_args)
  data, rawdb, gpdb, params = load_design(timer_args, logger)
    timer_only/read_platform.py::load_design(args, logger)
      collect platform LEF/LIB
      choose direct_rc_mode
      build params: benchmark/design_name/lefs/libs/def/sdc[/spef]
      data, rawdb, gpdb = load_dataset(args, logger, params)
        timer_only/database.py::load_dataset(args, logger, params)
          direct_timing_only = bool(args.route_segments or args.gr_rc)
          parser_params = params + parser flags
          rawdb, gpdb = IOParser.read(parser_params, ...)
            timer_only/io_parser.py::IOParser.read(...)
              check_params(...)
              io_parser.start(self.params)
                cpp_to_py/io_parser/PyBindCppMain.cpp::start_all(kwargs)
                  loadParams(kwargs)
                  rawdb = std::make_shared<db::Database>()
                  rawdb->load()
                  rawdb->setup()
                  gpdb = std::make_shared<gp::GPDatabase>(rawdb)
                  gpdb->setup()
                  return rawdb, gpdb
          design_info = IOParser.preprocess_design_info(gpdb, ...)
            gpdb.coreInfo/siteWidth/siteHeight/microns/node_type_indices/names
            gpdb.node_cpos_tensor()
            gpdb.node_size_tensor()
            gpdb.pin_rel_lpos_tensor()
            gpdb.pin_size_tensor()
            gpdb.pin_id2node_id_tensor()
            gpdb.hyperedge_info_tensor()
            gpdb.node2pin_info_tensor()
          data = PlaceData(args, logger, **design_info)
      return data, rawdb, gpdb, params
  device = torch.device(cuda or cpu)
  data = data.to_timing_device(device).preprocess_timing()
    PlaceData.to_timing_device(device)
    PlaceData.preprocess_timing()
      init_transform_state()
      preshift_timing()
      prescale_by_site_width_timing()
      pre_compute_timing_var()
      logging_timing_statistics()
```

`timer_only.timing_opt.GPUTimer(...)` 之后进入 `00_POWER_ARCHITECTURE.md` 的 GPUTimer/power 主干。

## 2. `run_xplace_worker(args)` 端口

文件：`tools/compare_ispd25_route_power_timing.py`

输入：

- `args.worker_split`: `visible` 或 `blind`。
- `args.worker_design`: design name。
- `args.out`: 输出根目录。
- `args.threads`: 传给 parser/GPUTimer 的线程数。
- `args.gpu`: CUDA device id。

关键 data-loader 相关操作：

```text
sys.argv = [
  "compare_ispd25_route_power_timing",
  "--platformPath", PLATFORM,
  "--designPath", BENCH / split,
  "--designName", design,
  "--route_segments", segment_path(split, design),
  "--global_placement", "False",
  "--legalization", "False",
  "--detail_placement", "False",
  "--write_placement", "False",
  "--num_threads", args.threads,
  "--gpu", args.gpu,
  "--result_dir", args.out / "xplace_run_timer_results",
  "--exp_id", case id,
]

timer_args = getArgs()
data, rawdb, gpdb, params = load_design(timer_args, logger)
device = torch.device(f"cuda:{timer_args.gpu}" if torch.cuda.is_available() else "cpu")
data = data.to_timing_device(device).preprocess_timing()
```

输出：

- `data`: timing-only `PlaceData`，已经移到 device 并完成坐标 scale/shift。
- `rawdb`: C++ `std::shared_ptr<db::Database>`。
- `gpdb`: C++ `std::shared_ptr<gp::GPDatabase>`。
- `params`: Python dict，仍保留 `sdc` 和 `route_segments` 给后续 `create_gputimer`。

## 3. `run_timer.getArgs()` 端口

文件：`run_timer.py`

输入：`run_xplace_worker` 临时写入的 `sys.argv`。

输出：`argparse.Namespace timer_args`。

本路径依赖的关键字段：

```text
timer_args.platformPath
timer_args.designPath
timer_args.designName
timer_args.route_segments
timer_args.num_threads
timer_args.gpu
timer_args.load_from_raw
timer_args.verbose_cpp_log
timer_args.cpp_log_level
timer_args.ignore_net_degree
timer_args.num_bin_x
timer_args.num_bin_y
```

注意：`getArgs()` 不读设计文件；它只把 worker argv 解析成后续 loader 使用的配置。

## 4. `load_design(args, logger)` 端口

文件：`timer_only/read_platform.py`

输入：

- `args.platformPath`: platform root，预期有 `lef/` 和 `lib/`。
- `args.designPath`: benchmark split root。
- `args.designName`: design name。
- `args.route_segments` 或 `args.gr_rc`: direct RC mode 开关。

执行：

```text
lef_path = platformPath / "lef"
lib_path = platformPath / "lib"
lefs = sorted(*lef)
libs = sorted(*lib)

direct_rc_mode = bool(args.gr_rc or args.route_segments)
if not direct_rc_mode:
  过滤掉名字含 ram 的 lib

design_dir = designPath / designName
plain_def = design_dir / f"{designName}.def"
plain_sdc = design_dir / f"{designName}.sdc"
if plain_def and plain_sdc exist:
  params = {
    benchmark: "gzz",
    design_name: designName,
    lefs: lefs,
    libs: libs,
    def: plain_def,
    sdc: plain_sdc,
  }
  if not direct_rc_mode:
    params["spef"] = design_dir / f"{designName}.spef"
else:
  使用 fallback TimingPredict/Sky130 风格路径
```

输出：

```text
return data, rawdb, gpdb, params
```

重要边界：

- `params["sdc"]` 在这里构造，但不是 `io_parser.start` 的核心输入；SDC 后面由 `create_gputimer(...)` 读入 `GTDatabase`。
- direct RC mode 下不加入 `spef`，避免 data loader 阶段引入 legacy SPEF RC。
- `load_design` 的下一跳是 `load_dataset(args, logger, params)`。

## 5. `load_dataset(args, logger, params)` 端口

文件：`timer_only/database.py`

输入：

- `params`: `load_design` 生成的路径/config dict。
- `args.load_from_raw`: 是否从原始 benchmark 读。
- `args.route_segments` / `args.gr_rc`: timing-only direct RC mode。
- `args.num_threads`: parser/C++ loader 线程数。

执行：

```text
direct_timing_only = bool(args.route_segments or args.gr_rc)
parser_params = dict(params)
parser_params["enable_pg"] = not direct_timing_only
parser_params["enable_fence"] = not direct_timing_only
parser_params["skip_def_net_wires"] = direct_timing_only
parser_params["skip_def_blockages"] = direct_timing_only

rawdb, gpdb = parser.read(
  parser_params,
  verbose_log=args.verbose_cpp_log,
  log_level=args.cpp_log_level,
  lite_mode=True,
  random_place=False,
  num_threads=args.num_threads)

include_names = (not direct_timing_only) or debug_needs_pin_names
include_celltype_names = (not direct_timing_only) or args.timing_opt

design_info = parser.preprocess_design_info(
  gpdb,
  timing_only=direct_timing_only,
  include_names=include_names,
  include_celltype_names=include_celltype_names)

data = PlaceData(args, logger, **design_info)
```

输出：

```text
return data, rawdb, gpdb
```

Direct timing-only 语义：

- 不需要 PG/fence/blockage 大对象。
- DEF net wires/blockages 可跳过，route segment RC 后续由 `init_dmp_rc_route_segments` 处理。
- `include_names=False` 时减少 Python 名字向量开销；debug path 可强制保留 pin/node names。

## 6. `IOParser.read(...)` 到 C++ `start_all(...)`

Python 文件：`timer_only/io_parser.py`

C++ pybind 文件：`cpp_to_py/io_parser/PyBindCppMain.cpp`

Python 端口：

```text
IOParser.read(params,
              verbose_log=False,
              log_level=2,
              lite_mode=False,
              random_place=True,
              num_threads=8,
              debug=False)
```

执行：

```text
check_params(params, verbose_log, log_level, lite_mode, random_place, num_threads)
if debug:
  io_parser.load_params(self.params)
  rawdb = io_parser.create_database()
  rawdb.load()
  rawdb.setup()
  gpdb = io_parser.create_gpdatabase(rawdb)
  gpdb.setup()
else:
  rawdb, gpdb = io_parser.start(self.params)
```

C++ pybind：

```text
m.def("start", &start_all)
m.def("load_params", &loadParams)
m.def("create_database", [](){ return make_shared<db::Database>(); })
m.def("create_gpdatabase", [](shared_ptr<db::Database> db){ return make_shared<gp::GPDatabase>(db); })
```

C++ `start_all`：

```text
std::tuple<std::shared_ptr<db::Database>, std::shared_ptr<gp::GPDatabase>>
start_all(const pybind11::dict& kwargs) {
  bool load_status = loadParams(kwargs);
  if (!load_status) throw invalid_argument(...);
  auto rawdb_ptr = make_shared<db::Database>();
  rawdb_ptr->load();
  rawdb_ptr->setup();
  auto gpdb_ptr = make_shared<gp::GPDatabase>(rawdb_ptr);
  gpdb_ptr->setup();
  return {rawdb_ptr, gpdb_ptr};
}
```

## 7. `loadParams(kwargs)` 映射

文件：`cpp_to_py/io_parser/PyBindCppMain.cpp`

主要输入到 `db::setting`：

```text
bookshelf/DEF/LEF:
  aux -> db::setting.BookshelfAux
  pl -> db::setting.BookshelfPl
  def -> db::setting.DefFile
  lef -> db::setting.LefFile
  cell_lef/tech_lef -> db::setting.LefCell/LefTech
  lefs -> db::setting.LefFiles

Liberty:
  lib -> db::setting.CellLib
  libs -> db::setting.LibFiles
  early_lib -> db::setting.CellLib_MIN
  late_lib -> db::setting.CellLib_MAX

Loader flags:
  lite_mode -> db::setting.liteMode
  enable_pg -> db::setting.EnablePG
  enable_fence -> db::setting.EnableFence
  skip_def_net_wires -> db::setting.SkipDefNetWires
  skip_def_blockages -> db::setting.SkipDefBlockages
  random_place -> db::setting.random_place
  num_threads -> db::setting.numThreads
  verbose_parser_log/global_log_level -> logger/config
```

注意：`params["sdc"]` 不在这里消费；SDC 是 GPUTimer graph/constraint 阶段的输入。

## 8. `db::Database::load/setup` 最底层入口

文件：`cpp_to_py/common/db/Database.cpp`

`Database::load()`：

```text
if BookshelfAux:
  readBSAux(...)
  def_read = true

if LefFile/LefCell+LefTech/LefFiles:
  readLEF(...)
  lef_read = true

if CellLib/CellLib_MIN+MAX/LibFiles:
  CellLib::read(...)
  CellLib::finish_read()
  liberty_read = true

后续按 setting 继续读 DEF/verilog/SPEF/route 相关 raw DB 输入。
```

`Database::setup()`：

```text
if def_read:
  SetupLayers()
  SetupFloorplan()

SetupCellLibrary()
SetupRegions()
if !setting.liteMode:
  SetupSiteMap()
  SetupRows()
  SetupRowSegments()
```

对于本 route-segment timing-only worker：

- `lite_mode=True`，所以 rawdb setup 会跳过 non-lite 的 site map/rows/row segments。
- `skip_def_net_wires=True` 和 `skip_def_blockages=True` 会影响 DEF reader 读入内容。
- `liberty_read` 必须为真，否则后续 `create_gputimer` 会报 Liberty file not found。

## 9. `gp::GPDatabase::setup()` 和更底层 C++ 审查

文件：`cpp_to_py/io_parser/gp/GPDatabase.cpp`

```text
GPDatabase::setup()
  if db::setting.random_place:
    setup_random_place()
  setupNum()
  setupRegions()
  setupNodes()
  setupNets()
  setupIndexMap()
  setupCheckVar()
  transferOrient()
  return true
```

关键子步骤：

```text
setupNum()
  dieInfo/coreInfo/siteW/siteH
  num_nodes = cells + iopins + placeBlockages
  num_nets = database.nets.size()
  num_pins = sum(net.pins.size())
  num_regions = database.regions.size()
  microns = database.DBU_Micron

setupNodes()
  按连接/固定/IO/blockage 分类并生成 gpdb node 顺序:
    Mov, FloatMov, Fix, IOPin, Blkg, FloatIOPin, FloatFix
  填充 node_types_indices、node_names、node_id2celltype_name 等

setupNets()
  从 rawdb nets/pins 建 gpdb nets/pins
  填充 pin_id2node_id、pin_id2net_id、net_names/pin_names 等

setupIndexMap()
  建 rawdb id 到 gpdb id 的映射

transferOrient()
  把 raw orientation 规范化到 gpdb nodes/pins
```

上面这些不是叶子节点。审查 data loader 时，需要继续展开
`setupNum/setupNodes/setupNets/setupIndexMap/transferOrient` 内部的字段写入、
helper 调用和 id 顺序。

### 9.1 `setupNum()`

端口：

```text
void GPDatabase::setupNum()
```

输入：

- `database`: `GPDatabase` 持有的 raw `db::Database&`。
- rawdb 已经完成 `Database::load()` 和 `Database::setup()`。

调用和计算：

```text
dieInfo = (database.dieLX, database.dieHX, database.dieLY, database.dieHY)
coreInfo = (database.coreLX, database.coreHX, database.coreLY, database.coreHY)
siteW = int(database.siteW)
siteH = database.siteH

num_nodes = database.cells.size()
          + database.iopins.size()
          + database.placeBlockages.size()

num_nets = database.nets.size()

num_pins = 0
for dbnet in database.nets:
  num_pins += dbnet->pins.size()

num_regions = database.regions.size()
num_celltype = database.celltypes.size()
microns = int(database.DBU_Micron)
```

副作用：

```text
nodes.reserve(num_nodes)
pins.reserve(num_pins)
nets.reserve(num_nets)
pin_id2node_id.reserve(num_pins)
pin_id2net_id.reserve(num_pins)
```

下游依赖：

- `setupNodes()` 依赖 `num_nodes` 和 `num_regions`。
- `setupNets()` 依赖 `num_nets` 和 `num_pins`。
- `get*Tensor()` 依赖这些 size 来创建 tensor shape。

审查重点：

- `num_nodes` 包含 `placeBlockages`，即使 timing-only 后续可能不使用 blockage。
- `num_pins` 是按 rawdb net pins 求和，不是按 cell pins 求和。
- `microns` 取 `DBU_Micron`，不是旧注释里的 `LefConvertFactor`。

### 9.2 `setupRegions()`

端口：

```text
void GPDatabase::setupRegions()
```

调用：

```text
for dbregion_id in [0, database.regions.size()):
  addRegion(dbregion_id)
```

`addRegion(dbregion_id)` 做什么：

```text
dbregion = database.regions[dbregion_id]
regions.emplace_back(GPRegion())
region.id = regions.size() - 1
region.name = dbregion->name()
region.ori_db_id = dbregion_id
region.type = dbregion->type()
for rect in dbregion->rects:
  region.addBox(rect.lx, rect.ly, rect.hx, rect.hy)
```

下游依赖：

- `setupNodes()` 里的 `addCellNode()` 会用 `cell->region->id`，并执行
  `regions[node.region_id].addNode(node.id)`。
- 如果 `setupRegions()` 顺序错到 `setupNodes()` 后面，cell node 的 region
  归属会越界或丢失。

### 9.3 `setupNodes()`

端口：

```text
void GPDatabase::setupNodes()
```

输入：

- `database.cells`
- `database.iopins`
- `database.placeBlockages`
- `regions`，必须已经由 `setupRegions()` 建好

第一阶段：分类 rawdb cell/iopin。

```text
for cell_id, cell in database.cells:
  if cell->is_connected:
    if !cell->fixed(): all_mov_ids.push(cell_id)
    else:              all_fix_ids.push(cell_id)
  else:
    if !cell->fixed(): all_float_mov_ids.push(cell_id)
    else:              all_float_fix_ids.push(cell_id)

for iopin_id, iopin in database.iopins:
  if iopin->is_connected:
    all_iopin_ids.push(iopin_id)
  else:
    all_float_iopin_ids.push(iopin_id)
```

第二阶段：按固定顺序创建 gpdb nodes。

```text
Mov:
  for cell_id in all_mov_ids:
    addCellNode(cell_id, "Mov")
  node_types_indices.push(start, end, "Mov")

FloatMov:
  for cell_id in all_float_mov_ids:
    addCellNode(cell_id, "FloatMov")
  node_types_indices.push(start, end, "FloatMov")

Fix:
  for cell_id in all_fix_ids:
    addCellNode(cell_id, "Fix")
  node_types_indices.push(start, end, "Fix")

IOPin:
  for iopin_id in all_iopin_ids:
    addIOPinNode(iopin_id, "IOPin")
  node_types_indices.push(start, end, "IOPin")

Blkg:
  for blkg_id in database.placeBlockages:
    addBlockageNode(blkg_id, "Blkg")
  node_types_indices.push(start, end, "Blkg")

FloatIOPin:
  for iopin_id in all_float_iopin_ids:
    addIOPinNode(iopin_id, "FloatIOPin")
  node_types_indices.push(start, end, "FloatIOPin")

FloatFix:
  for cell_id in all_float_fix_ids:
    addCellNode(cell_id, "FloatFix")
  node_types_indices.push(start, end, "FloatFix")
```

这个顺序是 data loader 的核心 invariant。Python 侧 `PlaceData` 会用
`node_type_indices` 推导：

```text
movable_index = (0, FloatMov.end)
connected_index = (0, IOPin.end)
fixed_index = (FloatMov.end, FloatFix.end)
fixed_connected_index = (fixed_index[0], connected_index[1])
```

`addCellNode(cell_id, node_type)` 写入：

```text
cell = database.cells[cell_id]
node.id = nodes.size() - 1
node.name = cell->name()
node.lx/ly = cell->lx()/ly()
node.width/height = cell->width()/height()
node.orient = cell->orient()
node.node_type = node_type
node.ori_db_id = cell_id
node.region_id = cell->region->id
regions[node.region_id].addNode(node.id)
node.cell_type_name = cell->ctype()->cls + "/" + cell->ctype()->name
if cell->fixed() && cell->ctype()->nonRegularRects().size() > 0:
  node.is_polygon_shape = true
cell->gpdb_id = node.id
node_names.push_back(cell->name())
```

`addIOPinNode(iopin_id, node_type)` 写入：

```text
iopin = database.iopins[iopin_id]
node.id = nodes.size() - 1
node.name = iopin->name
node.lx/ly = iopin->x/y
node.width/height = iopin->width()/height()
node.orient = iopin->orient()
node.node_type = node_type
node.ori_db_id = iopin_id
node.cell_type_name = iopin->name
iopin->gpdb_id = node.id
```

`addBlockageNode(blkg_id, node_type)` 写入：

```text
blockage = database.placeBlockages[blkg_id]
node.id = nodes.size() - 1
node.name = "Blockage_<blkg_id>"
node.lx/ly = blockage.lx/ly
node.width/height = blockage.w()/h()
node.orient = -1
node.node_type = "Blkg"
node.ori_db_id = blkg_id
node.cell_type_name = node.name
```

审查重点：

- `node.id` 必须等于 `nodes` vector 顺序，后续 tensor 全按这个 id 索引。
- `cell->gpdb_id` 和 `iopin->gpdb_id` 是 `setupNets()` 找 parent node 的入口。
- `node_names` 在 `addCellNode()` 中 push；IOPin/blockage 名字后续还会由
  `setupIndexMap()` 统一生成 `node_id2node_name`。
- polygon fixed cell 的 size tensor 后续会被 `getNodeSizeTensor()` 写成 0/0。

### 9.4 `setupNets()`

端口：

```text
void GPDatabase::setupNets()
```

输入：

- `database.nets`
- 每个 `db::Net::pins`
- `cell->gpdb_id` / `iopin->gpdb_id`，来自 `setupNodes()`
- `db::PinType` 的 physical pin box、direction、type/name

当前主路径不是旧的 `addNet()/addPin()` 串行路径，而是 prefix + parallel
direct fill。

第一阶段：计算每个 net 的 pin prefix。

```text
net_pin_start = vector<num_nets + 1>(0)
for dbnet_id in [0, database.nets.size()):
  net_pin_start[dbnet_id + 1] =
    net_pin_start[dbnet_id] + database.nets[dbnet_id]->pins.size()

assert net_pin_start[num_nets] == num_pins
```

第二阶段：resize 输出数组。

```text
nets.resize(num_nets)
pins.resize(num_pins)
net_names.resize(num_nets)
pin_id2node_id.resize(num_pins)
pin_id2net_id.resize(num_pins)
pin_names.resize(num_pins)
```

第三阶段：OpenMP parallel for，按 net 直接填充 `nets/pins/mapping`。

```text
for dbnet_id in parallel [0, database.nets.size()):
  dbnet = database.nets[dbnet_id]
  net = nets[dbnet_id]
  net.id = dbnet_id
  net.name = dbnet->name
  net.reservePins(dbnet->pins.size())
  net.ori_db_id = dbnet_id
  dbnet->gpdb_id = dbnet_id
  net_names[dbnet_id] = dbnet->name

  pin_begin = net_pin_start[dbnet_id]
  for pin_offset in [0, dbnet->pins.size()):
    dbpin = dbnet->pins[pin_offset]
    assert dbpin->is_connected

    is_iopin = dbpin->iopin != nullptr
    pintype = is_iopin ? dbpin->iopin->type : dbpin->type
    node = is_iopin ? nodes[dbpin->iopin->gpdb_id]
                    : nodes[dbpin->cell->gpdb_id]
    node_id = node.id
    pin_id = pin_begin + pin_offset

    pin = pins[pin_id]
    pin.id = pin_id
    if is_iopin:
      pin_names[pin_id] = pintype->name()
      pin.name_ref = pin_names[pin_id]
    else:
      pin_names[pin_id] = node.name + "/" + pintype->name()
      pin.name_ref = pin_names[pin_id]
    pin.macro_name_ref = pintype->name()
    pin.rel_lx/rel_ly = pintype->boundLX/boundLY
    pin.width/height = pintype->getW()/getH()
    pin.direction = pintype->direction()
    pin.type = pintype->type()
    pin.parent_node_id = node_id
    pin.parent_net_id = net.id
    pin.ori_db_info = {node.ori_db_id, is_iopin ? -1 : dbpin->parentCellPinId, net.ori_db_id}

    pin_id2node_id[pin_id] = node_id
    pin_id2net_id[pin_id] = net.id
    net.addPin(pin.id, pintype->direction() == 'o')
    dbpin->gpdb_id = pin.id
```

第四阶段：重建每个 node 的 pin list。

```text
node_pin_counts = zeros(num_nodes)
for pin_id in [0, num_pins):
  node_pin_counts[pin_id2node_id[pin_id]]++

node_pin_start = prefix_sum(node_pin_counts)
flat_node_pins = vector<num_pins>
for pin_id in [0, num_pins):
  node_id = pin_id2node_id[pin_id]
  flat_node_pins[node_pin_cursor[node_id]++] = pin_id

for node_id in parallel [0, num_nodes):
  node.clearPins()
  node.reservePins(node_pin_start[node_id + 1] - node_pin_start[node_id])
  for pos in [node_pin_start[node_id], node_pin_start[node_id + 1]):
    pin_id = flat_node_pins[pos]
    node.addPin(pin_id, pins[pin_id].macro_name)
```

旧 helper 路径仍存在，但当前 `setupNets()` 不走：

```text
addNet(dbnet_id)
  nets.emplace_back(...)
  for dbpin in dbnet->pins:
    addPin(dbpin, pintype, node, net, isIOPin)

addPin(...)
  pins.emplace_back(...)
  node.addPin(...)
  net.addPin(...)
  dbpin->gpdb_id = pin.id
```

审查重点：

- `pin_id = net_pin_start[dbnet_id] + pin_offset`，所以 pin id 顺序是 rawdb net
  顺序下的 net-local pin 展开顺序。
- `pin_id2node_id`、`pin_id2net_id` 在 parallel fill 阶段写好，后续 tensor getter
  直接 clone 这些 vector。
- `node.addPin(...)` 不在 parallel net loop 中做，而是后面用 prefix 统一 rebuild；
  这是避免多个 nets 并行写同一个 node 的 data race。
- `net.addPin(...)` 在 parallel net loop 中是安全的，因为每个 net 只被自己的
  `dbnet_id` iteration 写。

### 9.5 `setupIndexMap()`

端口：

```text
void GPDatabase::setupIndexMap()
```

输入：

- `nodes`
- `pins`
- `pin_id2node_id/pin_id2net_id/pin_names`

执行：

```text
pin_index_maps_ready =
  pin_id2node_id.size() == num_pins &&
  pin_id2net_id.size() == num_pins &&
  pin_names.size() == num_pins

node_id2node_name.resize(num_nodes)
node_id2celltype_name.resize(num_nodes)

if !pin_index_maps_ready:
  pin_id2node_id.resize(num_pins)
  pin_id2net_id.resize(num_pins)
  pin_names.resize(num_pins)
  for pin_id in parallel [0, pins.size()):
    pin = pins[pin_id]
    pin_id2node_id[pin_id] = pin.parent_node_id
    pin_id2net_id[pin_id] = pin.parent_net_id
    pin.moveNameTo(pin_names[pin_id])

for node_id in parallel [0, nodes.size()):
  node = nodes[node_id]
  node_id2node_name[node_id] = node.name
  node_id2celltype_name[node_id] = node.cell_type_name
```

当前 `setupNets()` 主路径已经提前填好 pin mapping，所以通常
`pin_index_maps_ready == true`。这个 fallback 主要服务旧串行 `addNet/addPin`
路径或未来变体。

下游依赖：

- Python `preprocess_design_info()` 的 `node_id2node_name()`、
  `node_id2celltype_name()`。
- `pin_id2node_id_tensor()`、`pin_id2net_id_tensor()`。

审查重点：

- `node_id2node_name[node_id]` 必须按 `nodes` vector 顺序。
- `node_id2celltype_name` 用于 debug 和 `timing_opt` 下的 `node_special_type`。
- 不要把 `node_names` 和 `node_id2node_name` 混为一谈；后者是本函数按 id
  重新生成的稳定数组。

### 9.6 `setupCheckVar()`

端口：

```text
void GPDatabase::setupCheckVar()
```

检查：

```text
nodes.size() == num_nodes
nets.size() == num_nets
pins.size() == num_pins
```

这是 data loader 的基本 size invariant 检查。注释中还有 die boundary 检查，但当前未启用。

### 9.7 `transferOrient()`

端口：

```text
void GPDatabase::transferOrient()
```

目标：

- 只规范化 gpdb node/pin，注释明确“不改变 rawdb”。
- 把非 `N`/`NONE` orient 的 node 转成 `N` 方向下的 size/pin relative geometry。

局部 helper：

```text
getOrientDegreeFlip(orient)
  N  -> (0,   0)
  W  -> (90,  0)
  S  -> (180, 0)
  E  -> (270, 0)
  FN -> (0,   1)
  FW -> (90,  1)
  FS -> (180, 1)
  FE -> (270, 1)
  NONE/unknown -> (0, 0)

getRotatedSizes(rotDegree, width, height)
  0/180 -> (width, height)
  90/270 -> (height, width)

getRotatedPinInfo(rotDegree, srcNodeW, srcNodeH, pinW, pinH, relLx, relLy)
  0:
    rel = (relLx, relLy)
  180:
    rel = (srcNodeW - relLx - pinW,
           srcNodeH - relLy - pinH)
  90:
    rel = (srcNodeH - relLy - pinH,
           relLx)
    swap(pinW, pinH)
  270:
    rel = (relLy,
           srcNodeW - relLx - pinW)
    swap(pinW, pinH)

getFlipYPinRelPos(nodeW, nodeH, relLx, relLy)
  relLx = nodeW - relLx
  relLy = relLy
```

主循环：

```text
for node in parallel nodes:
  srcOrient = node.orient
  if srcOrient != N and srcOrient != NONE:
    srcDegree, srcFlip = getOrientDegreeFlip(srcOrient)
    dstDegree, dstFlip = getOrientDegreeFlip(N)
    rotDegree = (dstDegree - srcDegree + 360) % 360
    flipY = (dstFlip != srcFlip)

    dstNodeWidth, dstNodeHeight =
      getRotatedSizes(rotDegree, node.width, node.height)

    for pin_id in node.pins():
      pin = pins[pin_id]
      dstPinWidth, dstPinHeight, dstRelLx, dstRelLy =
        getRotatedPinInfo(rotDegree, node.width, node.height,
                          pin.width, pin.height, pin.rel_lx, pin.rel_ly)
      pin.width/height = dstPinWidth/dstPinHeight
      pin.rel_lx/rel_ly = dstRelLx/dstRelLy

    node.width/height = dstNodeWidth/dstNodeHeight

    if flipY:
      for pin_id in node.pins():
        pin.rel_lx, pin.rel_ly =
          getFlipYPinRelPos(node.width, node.height, pin.rel_lx, pin.rel_ly)
```

输出/副作用：

- 修改 gpdb `node.width/height`。
- 修改 gpdb `pin.width/height` 和 `pin.rel_lx/rel_ly`。
- 不修改 node `lx/ly`，不修改 rawdb。
- 记录各 orient 转换统计。

下游依赖：

- `getNodeSizeTensor()` 读取转换后的 `node.width/height`。
- `getPinRelLPosTensor()` 和 `getPinSizeTensor()` 读取转换后的 pin geometry。
- 后续 `PlaceData.preprocess_timing()`、`create_timing_rawdb()` 都基于这个规范化几何。

审查重点：

- `transferOrient()` 在 `setupNets()` 和 `setupIndexMap()` 后执行，所以 pins 已经挂到 node 上。
- 这个函数只改 size 和 pin relative geometry，不改 node location；如果以后改成改 location，会影响 `node_cpos_tensor()` 语义。
- `getFlipYPinRelPos()` 当前是 `nodeW - relLx`，没有减 pin width；审查 pin 翻转问题时要从这里开始确认语义。

## 10. `IOParser.preprocess_design_info(...)` 端口

文件：`timer_only/io_parser.py`

输入：

```text
gpdb: C++ GPDatabase pybind object
timing_only: bool
include_names: bool
include_celltype_names: bool
```

执行和 C++ 调用：

```text
dieLX, dieHX, dieLY, dieHY = gpdb.coreInfo()
microns = gpdb.microns()
siteWidth = gpdb.siteWidth()
siteHeight = gpdb.siteHeight()

node_pos = gpdb.node_cpos_tensor()
node_lpos = None if timing_only else gpdb.node_lpos_tensor()
node_size = gpdb.node_size_tensor()
pin_rel_cpos = None if timing_only else gpdb.pin_rel_cpos_tensor()
pin_rel_lpos = gpdb.pin_rel_lpos_tensor()
pin_size = gpdb.pin_size_tensor()

pin_id2node_id = gpdb.pin_id2node_id_tensor()
hyperedge_index, hyperedge_list, hyperedge_list_end = gpdb.hyperedge_info_tensor()
pin_id2net_id = hyperedge_index[1].long().contiguous() if timing_only else None
node2pin_index, node2pin_list, node2pin_list_end = gpdb.node2pin_info_tensor()

if timing_only:
  node_id2region_id, region_boxes, region_boxes_end = None, None, None
else:
  node_id2region_id, region_boxes, region_boxes_end = gpdb.region_info_tensor()

node_type_indices = gpdb.node_type_indices()
node_id2node_name = gpdb.node_id2node_name() if include_names else []
node_id2celltype_name = gpdb.node_id2celltype_name() if include_celltype_names else None
```

输出：`design_info` dict，后续传给 `PlaceData(...)`。

## 11. `GPDatabase::get*Tensor()` 最底层 C++ 端口

文件：`cpp_to_py/io_parser/BindHelper.cpp` 绑定，`cpp_to_py/io_parser/gp/GPDatabase.cpp` 实现。

```text
gpdb.coreInfo()
  -> GPDatabase::getCoreInfo()
  -> 返回 coreLX, coreHX, coreLY, coreHY

gpdb.microns()
  -> GPDatabase::getMicrons()
  -> 返回 database.DBU_Micron

gpdb.node_cpos_tensor()
  -> GPDatabase::getNodeCPosTensor()
  -> torch.zeros({num_nodes,2})
  -> 对每个 node:
       x = node.lx + node.width / 2
       y = node.ly + node.height / 2

gpdb.node_size_tensor()
  -> GPDatabase::getNodeSizeTensor()
  -> torch.zeros({num_nodes,2})
  -> 普通 node 写 width/height；polygon-shape fixed blockage 写 0/0

gpdb.pin_rel_lpos_tensor()
  -> GPDatabase::getPinRelLPosTensor()
  -> torch.zeros({num_pins,2})
  -> pin_rel_lpos = pin.rel_lx, pin.rel_ly

gpdb.pin_size_tensor()
  -> GPDatabase::getPinSizeTensor()
  -> torch.zeros({num_pins,2})
  -> pin width/height

gpdb.pin_id2node_id_tensor()
  -> GPDatabase::getPinId2NodeIdTensor()
  -> torch::from_blob(pin_id2node_id.data(), {num_pins}, int64).clone()

gpdb.pin_id2net_id_tensor()
  -> GPDatabase::getPinId2NetIdTensor()
  -> torch::from_blob(pin_id2net_id.data(), {num_pins}, int64).clone()

gpdb.hyperedge_info_tensor()
  -> GPDatabase::getHyperedgeInfoTensor()
  -> 输出:
       hyperedge_index: [2, num_pins]，第一行 pin_id，第二行 net_id，按 pin_id 排序
       hyperedge_list:  net 顺序展开的 pin_id list
       hyperedge_list_end: 每个 net 的 prefix end

gpdb.node2pin_info_tensor()
  -> GPDatabase::getNode2PinInfoTensor()
  -> 输出:
       node2pin_index: [2, num_pins]，第一行 pin_id，第二行 node_id，按 pin_id 排序
       node2pin_list:  node 顺序展开的 pin_id list
       node2pin_list_end: 每个 node 的 prefix end
```

这些 C++ tensor getter 是 data loader 进入 Python tensor 世界的最底层端口。

## 12. `PlaceData(...)` 构造端口

文件：`timer_only/database.py`

输入：`design_info` dict。

关键字段：

```text
die_info
node_pos            # timing_only 下来自 gpdb.node_cpos_tensor()
node_size
pin_rel_lpos
pin_size
pin_id2node_id
pin_id2net_id       # timing_only 下来自 hyperedge_index[1]
hyperedge_list
hyperedge_list_end
node2pin_list
node2pin_list_end
node_type_indices
movable_index
connected_index
fixed_index
site_info
microns
node_id2node_name
node_id2celltype_name
```

构造时做的关键派生：

```text
node_special_type = zeros(num_nodes_for_special, int32)
if args.timing_opt and node_id2celltype_name exists:
  CORE/BUF -> 1
  CORE/DFF -> 2

dataset_format = "lefdef" or "bookshelf"
design_name = benchmark + "/" + params["design_name"]
movable_connected_index = (movable_index[0], node_type_indices[0][1])
fixed_connected_index = (fixed_index[0], connected_index[1])
num_nodes = node_pos.shape[0]
num_pins = pin_id2node_id.shape[0]
num_nets = hyperedge_list_end.shape[0]
```

## 13. `PlaceData.to_timing_device(device)` 端口

文件：`timer_only/database.py`

输入：`torch.device`。

执行：

```text
self.device = device
self.to(device,
  "die_info",
  "node_pos",
  "node_size",
  "pin_rel_cpos",
  "pin_rel_lpos",
  "pin_size",
  "pin_id2node_id",
  "pin_id2net_id",
  "hyperedge_list",
  "hyperedge_list_end",
  "node2pin_list",
  "node2pin_list_end",
  "node_special_type")
```

输出：`self`。

边界：

- 这里是 Python/Torch tensor device copy，不是 C++ 调用。
- timing-only path 不移动 placement/fence/density 专用字段。

## 14. `PlaceData.preprocess_timing()` 端口

文件：`timer_only/database.py`

执行：

```text
preprocess_timing()
  init_transform_state()
  preshift_timing()
  prescale_by_site_width_timing()
  pre_compute_timing_var()
  logging_timing_statistics()
  return self
```

子函数语义：

```text
init_transform_state()
  die_shift = tensor([0.0, 0.0])
  die_scale = tensor([1.0, 1.0])

preshift_timing()
  die_shift = [die_lx, die_ly]
  die_info = die_info shifted so lower-left becomes 0,0
  node_pos -= die_shift
  self.die_shift += die_shift

prescale_by_site_width_timing()
  scalar_at = site_width
  die_info /= scalar_at
  node_pos /= scalar_at
  node_size /= scalar_at
  if pin_rel_cpos is not None: pin_rel_cpos /= scalar_at
  pin_rel_lpos /= scalar_at
  pin_size /= scalar_at
  die_scale *= scalar_at

pre_compute_timing_var()
  unit_len = [(die_hx - die_lx)/num_bin_x, (die_hy - die_ly)/num_bin_y]
  die_ur = upper-right of die_info
  die_ll = lower-left of die_info
  hpwl_scale = die_scale / site_width
  start_idx = hyperedge_list_end.roll(1); start_idx[0] = 0
  net_to_num_pins = hyperedge_list_end - start_idx
  net_mask = (net_to_num_pins <= args.ignore_net_degree) & (net_to_num_pins >= 2)
  net_weight = ones(num_nets)
  num_macros/num_movable_macros/num_fixed_macros = 0
  total_mov_area_without_filler = 0.0
  bin_area = prod(unit_len)

logging_timing_statistics()
  log #nodes/#nets/#pins/core/site/bin info
  args.include_macros = False
```

输出：`PlaceData`，其中后续 `timer_only.timing_opt.GPUTimer` 会使用：

```text
node_pos
node_size
pin_rel_lpos
pin_size
pin_id2node_id
pin_id2net_id
node2pin_list
node2pin_list_end
hyperedge_list
hyperedge_list_end
net_mask
movable_index
fixed_connected_index
site_width
microns
```

## 15. Handoff 到 GPUTimer

Data loader 到这里结束。下一跳是：

```text
timer_only.timing_opt.GPUTimer(data, rawdb, gpdb, params, timer_args)
```

该 wrapper 会把 data loader 产物传入：

```text
gputimer.create_timing_rawdb(...)
gputimer.create_gputimer(params, rawdb, gpdb, timing_raw_db)
```

这部分属于 `00_POWER_ARCHITECTURE.md` 的 GPUTimer/power 主干。

## 16. Data Loader 不变量

- `params["sdc"]` 必须保留到 `create_gputimer(...)`；`io_parser.start(...)` 不消费 SDC。
- direct RC mode 下不能加入 SPEF，也不能依赖 DEF net wires/blockages 作为 RC 来源。
- `pin_id`、`node_id`、`net_id` 必须与 gpdb 顺序一致；后续 GPUTimer/power tensor 全部按这些 id 索引。
- `hyperedge_list_end` 和 `node2pin_list_end` 是 prefix-end，不是 start；Python 通过 roll 计算 start。
- `node_pos` 在 data loader 里是 center position；GPUTimer wrapper 后面会转换成 lower-left `node_lpos = node_pos - node_size/2`。
- `pin_rel_lpos` 在 GPDB 里是 lower-left relative position；GPUTimer wrapper 后面会加 `pin_size/2` 得到 pin center-related offset。
- `preprocess_timing()` 后所有坐标已经按 `site_width` 缩放。
- `net_mask` 是在 Python `pre_compute_timing_var()` 里按 net degree 生成的，后续 `create_timing_rawdb` 直接消费。

## 17. 人工审查顺序

1. `tools/compare_ispd25_route_power_timing.py::run_xplace_worker`：确认 worker argv 和 stage 顺序。
2. `run_timer.py::getArgs`：确认参数名和默认值。
3. `timer_only/read_platform.py::load_design`：确认 LEF/LIB/DEF/SDC/SPEF/direct RC mode。
4. `timer_only/database.py::load_dataset`：确认 parser flags 和 `timing_only` 分支。
5. `timer_only/io_parser.py::IOParser.read`：确认 Python 到 cpybin 的入口。
6. `cpp_to_py/io_parser/PyBindCppMain.cpp::loadParams/start_all`：确认 `db::setting` 和 C++ load/setup 顺序。
7. `cpp_to_py/common/db/Database.cpp::load/setup`：确认 rawdb 读入和 lite mode。
8. `cpp_to_py/io_parser/gp/GPDatabase.cpp::setup`：确认 gpdb node/net/pin/id 顺序。
9. `timer_only/io_parser.py::preprocess_design_info` 和 `GPDatabase::get*Tensor()`：确认 tensor shape、id mapping、prefix-end 语义。
10. `timer_only/database.py::PlaceData.to_timing_device/preprocess_timing`：确认 device copy、scale/shift、net_mask。
