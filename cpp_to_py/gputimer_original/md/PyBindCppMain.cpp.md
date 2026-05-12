# PyBindCppMain.cpp

## Purpose
Python bindings for GPUTimer CUDA module using pybind11. Exposes C++/CUDA classes and functions to Python.

## Classes Exposed

### GPUTimer
Main timing analysis engine. Methods:
- **Initialization**: `time_unit()`, `read_spef()`, `init()`, `levelize()`
- **RC Extraction**: `update_rc()`, `update_rc_flute()`, `update_rc_spef()`
- **Timing Computation**: `update_states()`, `update_timing()`, `update_endpoints()`
- **Reporting**: `report_wns()`, `report_tns_elw()`, `report_wns_and_tns()`
- **Pin Analysis**: `report_pin_slack()`, `report_pin_at()`, `report_pin_rat()`, `report_pin_slew()`, `report_pin_load()`
- **Path Analysis**: `report_path()`, `report_K_path()`, `report_criticality()`
- **Graph Dump**: `dump_timing_graph(outfile)` — Outputs JSONL timing graph
- **ML Inference**:
  - `read_infer(infile)` — Load ML-predicted delays from .infer CSV
  - `propagate_infer_timing()` — Propagate ML predictions through graph

### TimingTorchRawDB
PyTorch tensor database for design geometry and connectivity.

### GTDatabase
Graph timing database extracted from Liberty/LEF/DEF/SPEF files.

## Module Functions

- `create_gputimer()` — Factory function creating GPUTimer from design parameters
- `create_timing_rawdb()` — Create TimingTorchRawDB from PyTorch tensors

## Return Policies

- `py::return_value_policy::move` — Transfer ownership to Python
- `py::return_value_policy::copy` — Copy data to Python
