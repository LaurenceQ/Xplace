# GPUTimer Module Documentation Index

## Overview
GPUTimer is a CUDA-accelerated static timing analysis engine with ML inference support. It loads Liberty timing models, extracts RC from placement, and computes timing slacks.

## Quick Navigation

### Configuration & Build
- **base.h** — CUDA block/thread configuration, corner definitions
- **CMakeLists.txt** — Build system (GLOB_RECURSE for .cu files, linking, Python bindings)
- **PyBindCppMain.cpp** — Python bindings via pybind11, module exports

### Core Data Structures
- **GPUTimer.h** — Main class, GPU memory layout, API methods
- **gputiming.h** — Device utilities (interpolation, LUT lookup)
- **GTDatabase.h** — Timing database (Liberty, netlist, constraints)

### Standard Timing (STA)
- **propagate.cpp** — Wrapper for standard STA
- **propagate.cu** — Forward/backward timing propagation kernels (LUT-based)

### ML Inference Timing
- **read_infer.cu** — Load .infer CSV file, populate GPU arrays with ML-predicted delays
- **propagate_infer.cu** — Propagate ML-predicted delays (critical: net arc transition validity)

### Utilities
- **dump.cpp** — Export timing graph to JSONL for ML training
- **levelize.cu** — Compute topological levels
- **rctree.cu/cpp** — RC extraction via FLUTE/Steiner tree
- **path.cu/cpp** — Critical path reporting
- **spef.cpp** — SPEF file parsing
- **utils.cuh** — CUDA device utilities

## Key Concepts

### Timing Corners (NUM_ATTR = 4)
- **0**: Early-rise (el=0, rf=0)
- **1**: Early-fall (el=0, rf=1)
- **2**: Late-rise (el=1, rf=0)
- **3**: Late-fall (el=1, rf=1)

### GPU Array Layout
- **pinAT[pin_id * 4 + corner]** — Arrival time (ns)
- **pinRAT[pin_id * 4 + corner]** — Required arrival time (ns)
- **arcDelay[arc_id * 8 + idx]** — Arc delay (index 0-7, 2 edges × 4 corners)

### Arc Types
- **0**: Net arc (wire delay, irf==orf required)
- **1**: Cell arc (combinational logic, all transitions valid)

### Net Arc Transition Validity ⚠️ **CRITICAL**
- rise→rise ✓ (indices 0, 4)
- fall→fall ✓ (indices 3, 7)
- rise→fall ✗ (indices 1, 5, stored as NaN)
- fall→rise ✗ (indices 2, 6, stored as NaN)

**Propagation kernels skip NaN delays** — this filters out invalid net arc transitions.

## Data Flow

### Standard STA
```
Design (LEF/DEF/Lib/SDC)
    ↓ [GTDatabase::ExtractTimingGraph()]
Netlist + Timing Graph
    ↓ [GPUTimer::initialize()]
GPU memory allocated, LUT uploaded
    ↓ [GPUTimer::update_rc_timing()]
RC extracted from placement
    ↓ [GPUTimer::update_timing()]
update_states() → propagate.cu kernels
    ↓
AT/RAT computed, slack = RAT - AT
```

### ML Inference
```
Design + ML Model → inference.py
    ↓
TimingPredict/.infer CSV (delays in ns)
    ↓ [GPUTimer::read_infer()]
Parse CSV, D2H copy arcDelay, update delays, H2D copy
    ↓
GPU arcDelay populated with ML predictions
    ↓ [GPUTimer::propagate_infer_timing()]
propagate_infer.cu kernels
    ↓
AT/RAT computed using ML delays
```

### ML Training Data Export
```
Standard STA: update_timing()
    ↓ [GPUTimer::dump_timing_graph()]
JSONL: nodes + net_arcs + cell_arcs
    ↓ [load_graph_from_dump.py]
DGL Heterograph with LUT features
    ↓
ML model training
```

## Important Files

### Must Read First
1. **base.h** — Understand NUM_ATTR, BLOCK_SIZE
2. **GPUTimer.h** — Understand GPU array layout and API
3. **gputiming.h** — Understand LUT query and interpolation

### For Standard Timing
- **propagate.cu** — Main STA kernels
- **rctree.cu** — RC extraction

### For ML Inference
- **read_infer.cu** — CSV parsing and GPU upload (understand net arc NaN handling)
- **propagate_infer.cu** — ML delay propagation (understand irf!=orf skip for net arcs)

### For Training
- **dump.cpp** — JSONL output format
- **gputiming.h** → LUT structure for feature extraction

## Common Tasks

### Run Standard STA
```python
from cpp_to_py import gputimer
gputimer.update_states()
gputimer.update_rc_timing(node_positions)
gputimer.update_timing()
wns = gputimer.report_wns(el=1)  # Late slack
```

### Use ML Inference
```python
gputimer.update_states()
gputimer.read_infer("design.infer")        # Load ML predictions
gputimer.propagate_infer_timing()          # Propagate using ML
wns = gputimer.report_wns(el=1)
```

### Export for ML Training
```python
gputimer.update_timing()
gputimer.dump_timing_graph("timing_graph.json")
# Then: load_graph_from_dump.py → DGL graph
```

## Debugging Tips

### FLT_MAX in slack values
- Likely cause: NaN in timing propagation
- Check: Are arcDelay values populated? Use read_infer debug logs
- Check: CUDA synchronization after H2D memcpy?
- Check: Network arc have irf==orf? (propagate_infer.cu checks via NaN)

### Missing delay updates
- Check read_infer.cu logs for "Updated X net arcs, Y missing"
- Missing arcs indicate pin not found in fanin list
- Check: CSR lookup indices correct?

### Performance issues
- Profile with nvprof
- Check level parallelism (# pins per level)
- Reduce # density bins if OOM
