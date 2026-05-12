# GPUTimer.h

## Class Structure
Main GPU-accelerated timing analysis engine. Manages GPU memory, timing computation, and result reporting.

## Key GPU Arrays

### Pin Arrays (size: num_pins * NUM_ATTR)
- `pinSlew` — Transition time for each pin/corner (ns)
- `pinLoad` — Load capacitance for each pin/corner (F)
- `pinAT` — Arrival time for each pin/corner (ns)
- `pinRAT` — Required arrival time for each pin/corner (ns)
- `pinCap` — Total capacitance (pin + wire)
- `pinWireCap` — Wire capacitance extracted from RC
- `pinImpulse` — Impulse for slew propagation
- `pinRootDelay` — Root net delay
- `pinRootRes` — Root net resistance

### Arc Arrays (size: num_arcs * 2 * NUM_ATTR)
- `arcDelay` — Delay for each arc and corner (ns)
  - First factor: arc_id * 2 — accounts for early (el=0) and late (el=1) edges
  - Second factor: 4 corners (er, ef, lr, lf)
  - **Format**: arcDelay[arc_id * 8 + corner_index]
- `arcSlew` — Slew at output of arc

### Test Arrays (size: num_tests * NUM_ATTR)
- `testRelatedAT` — AT at test input
- `testRAT` — RAT at test output
- `testConstraint` — Setup/hold constraint value

### Prefix Tracking (size: num_pins * NUM_ATTR)
- `at_prefix_pin` — Previous pin in critical path
- `at_prefix_arc` — Arc used for AT computation
- `at_prefix_attr` — Transition type used

## Graph Structure

### Connectivity (CSR Format)
- `pin_backward_arc_list_end[pin_id]` — Start index of fanin arcs
- `pin_backward_arc_list[i]` — Arc ID at index i
- `pin_forward_arc_list_end[pin_id]` — Start index of fanout arcs
- `pin_forward_arc_list[i]` — Arc ID at index i
- `timing_arc_from_pin_id[arc_id]` — Source pin
- `timing_arc_to_pin_id[arc_id]` — Sink pin
- `arc_types[arc_id]` — 0=net arc, 1=cell arc

### Level-Based Propagation
- `level_list` — Pins ordered by topological level
- `level_list_end_cpu` — Level boundaries (host copy)

## Core Methods

### Initialization & Setup
- `initialize()` — Allocate GPU memory, initialize LUT allocator
- `levelize()` — Compute topological levels
- `read_spef(file)` — Read SPEF parasitic data

### RC Extraction
- `update_rc_timing()` — Compute RC from pin positions
- `update_rc_timing_flute()` — Extract RC via FLUTE Steiner tree
- `update_rc_timing_spef()` — Load RC from SPEF file

### Timing Analysis (Standard)
- `update_states()` — Reset timing arrays to NaN
- `update_timing()` — Standard STA propagation
- `update_endpoints()` — Compute slack at endpoints

### Timing Analysis (ML Inference)
- `read_infer(infile)` — **Load ML-predicted delays from .infer CSV file**
  - Parses node delays (net arc timing)
  - Parses edge delays (cell arc timing)
  - Converts from nanoseconds to internal units
  - Updates GPU arcDelay array via CUDA memcpy
  - All delays stored at indices matching corner encoding
- `propagate_infer_timing()` — **Propagate ML-predicted delays through graph**
  - Forward pass: Compute AT using ML arcDelay
  - Backward pass: Compute RAT
  - Uses library LUT for test constraint setup/hold

### Reporting
- `report_wns()` / `report_tns_elw()` — Worst/total negative slack
- `report_pin_at()` / `report_pin_rat()` — Return PyTorch tensor [num_pins, 4]
- `report_pin_slew()` / `report_pin_load()` — Slew and load tensors
- `report_path()` — Critical path (pin IDs and delays)
- `report_K_path()` — Top K critical paths
- `dump_timing_graph(outfile)` — Output timing graph to JSONL

## Design Parameters

- `num_pins` — Total pins in circuit
- `num_arcs` — Total timing arcs
- `num_tests` — Timing constraint tests
- `num_nets` — Wires in circuit
- `num_nodes` — Cells in circuit
- `num_movable_nodes` — Cells that can be placed
- `clock_period` — Clock period in ns

## Geometry

- `x`, `y` — Current pin positions (from placement)
- `init_x`, `init_y` — Initial positions
- `node_size_x`, `node_size_y` — Cell dimensions
- `pin_offset_x`, `pin_offset_y` — Pin position relative to cell origin
- `pin2node_map` — Which cell owns each pin
- `pin2net_map` — Which net each pin belongs to
- `scale_factor`, `microns` — Coordinate system scaling

## Wire Model

- `wire_resistance_per_micron` — ohm/µm
- `wire_capacitance_per_micron` — F/µm
- `res_unit`, `cap_unit` — Conversion factors to internal units
