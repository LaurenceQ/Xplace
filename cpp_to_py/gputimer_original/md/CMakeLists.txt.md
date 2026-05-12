# CMakeLists.txt

## Build Configuration

### Source Discovery
- **Pattern**: `GLOB_RECURSE` collects all `.cpp`, `.hpp`, `.cu` files from current directory
- **CRITICAL**: Must re-run `cmake` after adding new `.cpp` files (glob is evaluated at configure time)

### Library Targets

#### `libgt` (Static Library)
- **Type**: Static library combining CPU and CUDA code
- **Sources**:
  - All `.cpp`, `.hpp` files from gputimer/
  - All `.cu` files from gputimer/
  - GPDatabase.cpp for placement database
  - flute.cpp for FLUTE Steiner tree
- **Properties**:
  - `CUDA_RESOLVE_DEVICE_SYMBOLS ON` — Resolve CUDA symbols at link time
  - `POSITION_INDEPENDENT_CODE ON` — PIC for dynamic linking
- **Dependencies**: xplace_common, io_parser, OpenMP

#### `gputimer` (PyTorch Extension Module)
- **Type**: PyTorch extension (pybind11 module)
- **Source**: PyBindCppMain.cpp (Python bindings)
- **Linking**: Links against `libgt` static library
- **Install**: `${XPLACE_LIB_DIR}` (Python site-packages)

### Dependencies
- **PyTorch**: Headers and libraries (`TORCH_INCLUDE_DIRS`, `TORCH_PYTHON_LIBRARY`)
- **OpenMP**: For parallelization
- **FLUTE MP**: Rectilinear Steiner tree library

### Compiler Options
- `-fPIC` — Position-independent code for shared library
- `--extended-lambda` — CUDA extended lambda support

### Include Paths
- `${PROJECT_SOURCE_DIR}/cpp_to_py` — All modules
- `${TORCH_INCLUDE_DIRS}` — PyTorch headers
- `${FLUTE_MP_INCLUDE_DIR}` — FLUTE headers
