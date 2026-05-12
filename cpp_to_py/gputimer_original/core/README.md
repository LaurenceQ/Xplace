# GPUTimer Core

Implementation of GPU-accelerated static timing analysis. Computes arrival times, required arrival times, slew, and arc delays using CUDA kernels.

## Files

### `GPUTimer.h` / `GPUTimer.cpp` / `GPUTimer.cu`
Main class. Owns raw GPU pointers (`pinAT`, `pinRAT`, `pinSlew`, `pinLoad`, `arcDelay`, etc.) and references to `GTDatabase` and `TimingTorchRawDB`. Implements the full STA flow:
- `initialize()` — allocates GPU arrays, copies liberty LUTs to device
- `levelize()` — topological sort via BFS (see `levelize.cu`)
- `update_rc_timing*()` — RC extraction and load/delay population
- `update_states()` — recompute gate input/output slew from LUTs
- `update_timing()` — forward + backward passes level by level
- `update_endpoints()` — collect endpoint slacks into `endpoint_slacks`

### `gputiming.h`
Shared CUDA device functions and constants:
- `NUM_ATTR = 4` — number of timing corners (early-rise, early-fall, late-rise, late-fall)
- Index convention: `k = el*2 + rf` → `{0,1,2,3}` = `{er,ef,lr,lf}`
- `TimingArc` struct: liberty arc with LUT interpolation coefficients

### `propagate.cu` / `propagate.cpp`
Forward and backward propagation kernels:
- `propagateAT` — computes `pinAT` by accumulating `arcDelay` along arcs, level by level
- `propagateRAT` — backward pass computing `pinRAT`
- **arcDelay layout**: `arcDelay[arc_id * 2 * NUM_ATTR + k]`
  - For net arcs: corners `{er,ef,lr,lf}` sit at `k ∈ {0,3,4,7}` (from `el_rf_rf = (i<<1)+(i&1)`)
  - For cell arcs: `k = el*4 + irf*2 + orf` → `k ∈ {0..7}`

### `levelize.cu`
GPU BFS topological sort:
- `advanceLevel` kernel: marks next-level pins by checking that all fanin arcs are resolved
- Builds `level_list` (flat array of pin IDs per level) and `level_list_end_cpu` (CPU vector of level boundaries)
- Output used by `update_timing()` to process pins in topological order

### `rctree.cu` / `rctree.cpp`
Star-model Elmore delay computation per net:
- `RCTreeNet` kernel: for each net, computes wire resistance/capacitance from pin positions using `wire_resistance_per_micron` / `wire_capacitance_per_micron`
- Populates `pinLoad` (total load including wire + liberty cap), `pinRootDelay` (Elmore delay), `pinImpulse`, `pinWireCap`
- `pinCap` layout: `[pin_id * (NUM_ATTR + 2) + j]` — 6 values per pin (4 corners + 2 defaults)
- FLUTE variant (`rctree_flute`) uses Steiner tree topology instead of star model

### `path.cu` / `path.cpp`
Critical path tracing and criticality computation:
- `explore_path` / `explore_path_deterministic_kernel`: GPU backward trace from endpoints following `at_prefix_pin/arc/attr` pointers
- `report_path()`: traces a single endpoint's critical path, returns pin IDs + delays + arrival times
- `report_criticality()`: computes per-pin criticality weighted by path rank `1/(1+k)²` across top-K paths
- `report_K_path()`: returns top-K critical path pin-ID lists

### `spef.cpp`
SPEF parasitics import:
- `read_spef()` — parses SPEF file via `parser-spef`
- `update_rc_timing_spef()` — maps SPEF net names to GTDatabase net IDs, populates RC tree for each net

### `dump.cpp`
Timing graph export to JSONL:
- `dump_timing_graph(outfile)` — writes one JSON record per line
- Node records: `id`, `name` (from `gtdb.pin_names`), `features[10]`, `labels[17]`
- Edge records:
  - `net_out` (driver→sink): `features=[dx_um, dy_um]`, `labels=[er,ef,lr,lf] delays in ps`
  - `net_in` (sink→driver): `features=[-dx_um, -dy_um]`, same delay labels
  - `cell_out`: `features=[8 arc delays in ps]`, `lut_early`/`lut_late` with LUT breakpoints and table values
- See file header for full feature/label schema

### `utils.cuh`
CUDA utility macros and helpers used across kernels (thread indexing, warp reductions, etc.)

## Data Flow

```
GTDatabase (CPU topology)          TimingTorchRawDB (GPU tensors)
  pin_names, arc_types               pinAT, pinRAT, pinSlew, pinLoad
  timing_arc_from/to_pin_id          arcDelay, x, y, pin_offset_x/y
  primary_inputs/outputs             pin2node_map
  endpoints_id
        |                                     |
        +------------- GPUTimer --------------+
                           |
              levelize() → propagate() → path()
                           |
              dump_timing_graph() → .jsonl
```

## Key Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| `NUM_ATTR` | 4 | Timing corners per pin |
| `NET_DELAY_IDX` | `{0,3,4,7}` | arcDelay offsets for net arc corners |
| `time_to_ps` | `time_unit × 1e12` | Internal time → picoseconds |
| `cap_to_fF` | `cap_unit × 1e15` | Internal cap → femtofarads |

## Lessons Learned: Adding New GPU Methods

### The Correct Pattern (from propagate.cpp)

When adding a new GPU-accelerated method like `propagate_infer_timing()`:

**❌ WRONG - Don't do this:**
- Create wrapper functions in `.h` or `.cu` files
- Add forward declarations to `GPUTimer.h`
- Mix CPU and GPU code organization
- Iterate multiple times on where code lives

**✅ CORRECT - Do this:**

1. **Define CUDA kernels in `.cu` file** (propagate_infer.cu):
   - Device functions: `__device__ void propagateInferAT()`
   - Global kernels: `__global__ void propagatePinInferAT<<<>>>()`
   - Host orchestrator: `void propagate_infer_timing_impl()` that launches kernels

2. **Forward declare the impl in `.cpp` file** (propagate.cpp):
   ```cpp
   void propagate_infer_timing_impl(...);  // forward decl
   ```

3. **Implement the class method in `.cpp` file** (propagate.cpp):
   ```cpp
   void GPUTimer::propagate_infer_timing() {
       propagate_infer_timing_impl(...);  // call impl
   }
   ```

4. **Nothing in `.h` except the method declaration** (already in GPUTimer.h):
   ```cpp
   void propagate_infer_timing();
   ```

### Why this pattern?
- `.cpp` handles CPU-GPU boundary (calling .cu functions)
- `.cu` handles pure GPU code (kernels + launch orchestration)
- `.h` stays clean (no internal implementation details)
- Mirrors existing `update_timing()` / `update_timing_cuda()` pattern exactly

### Key Takeaway
**Always read existing similar code first.** The pattern was already there in `propagate.cpp`. Copy it. Don't invent new structures.
