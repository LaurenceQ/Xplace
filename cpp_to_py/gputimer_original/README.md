# GPU Timer Module

GPU-accelerated static timing analysis engine for timing-driven placement. Computes path delays, slacks, and timing gradients for optimization.

## Purpose

Provides fast timing analysis on GPU for use during placement iterations. Enables Xplace's timing-driven placement optimization by computing timing metrics and delay gradients for cells.

## Core Components

### GPUTimer Class

**Initialization**:
```python
timer = gputimer.GPUTimer(gtdb, timing_raw_db)
```

### Timing Analysis Methods

#### Fundamental Operations
- **`initialize()`** - Initialize timing graph structures
- **`levelize()`** - Topologically sort timing graph
- **`update_states()`** - Update gate delays based on cell geometry
- **`update_timing()`** - Forward/backward pass to compute delays and slacks

#### Parasitics Update
- **`update_rc()`** - Update RC parasitics from routing capacitance
- **`update_rc_flute()`** - Update RC using FLUTE Steiner tree estimation
- **`update_rc_spef()`** - Update RC from SPEF file

#### Reporting Functions
- **`report_wns()`** - Worst Negative Slack (WNS) across all endpoints
- **`report_tns_elw()`** - Total Negative Slack and Endpoint Late Window
- **`report_wns_and_tns()`** - Combined WNS and TNS report

#### Detailed Information
- **`report_pin_slack(pin_id)`** - Slack at each pin
- **`report_pin_at(pin_id)`** - Arrival time at each pin
- **`report_pin_rat(pin_id)`** - Required arrival time
- **`report_pin_slew(pin_id)`** - Slew/transition time
- **`report_pin_load(pin_id)`** - Load capacitance on pins
- **`report_endpoint_slack(endpoint_id)`** - Path slack by endpoint
- **`report_criticality(pin_id, [endpoint_id])`** - Criticality metric (0-1) for sensitivities

#### Path Reporting
- **`report_path(endpoint_id, num_edges)`** - Complete path details
- **`report_K_path(K, endpoint_id)`** - Top K critical paths
- **`report_criticality_threshold(threshold)`** - Pins above criticality threshold

### Configuration

#### TimingTorchRawDB
Raw timing database built from circuit netlist:
```python
timing_raw_db = gputimer.TimingTorchRawDB(
    pin_capacitance_tensor,  # Pin load capacitances
    pin_resistance_tensor,   # Pin resistances
    cell_delay_tensor,       # Intrinsic gate delays
    cell_slew_tensor,        # Output slew characteristics
    ...  # Additional timing parameters
)
```

## Usage Example

```python
from cpp_to_py import gputimer
import torch

# Parse design and create timing database
rawdb, gpdb = parser.read(params)

# Create timing raw database (from timing data)
timing_raw_db = gputimer.TimingTorchRawDB(...)

# Create GPU timer
timer = gputimer.create_gputimer(
    {"sdc": "constraints.sdc"},  # Optional SDC constraints
    rawdb, gpdb, timing_raw_db
)

# Initialize timing
timer.initialize()
timer.levelize()

# Update placement-dependent values (called each iteration)
node_pos = placement_nodes_pos  # From GP
timer.update_states()  # Update gate delays based on cell sizes
timer.update_rc_flute()  # Estimate RC from placement
timer.update_timing()  # Compute delays and slacks

# Get timing metrics
wns, tns = timer.report_wns_and_tns()
print(f"WNS: {wns} ps, TNS: {tns} ps")

# Get timing gradients for optimization
slack = timer.report_pin_slack()
at = timer.report_pin_at()
criticality = timer.report_criticality()

# Use for timing-driven placement weight computation
```

## Timing Analysis Flow

**Per-Iteration Loop**:
1. Get cell positions from GP
2. Call `update_states()` - compute gate delays based on sizing
3. Call `update_rc_*()` - estimate parasitics from placement
4. Call `update_timing()` - compute delays/slacks
5. Call `report_*()` - extract metrics and gradients
6. Adjust placement weights based on criticality
7. Next GP iteration

## Supported Constraints

**SDC File Support**:
- Create_clock definitions
- set_input_delay / set_output_delay
- set_max_delay / set_min_delay
- Basic constraint parsing

## Timing Metrics

| Metric | Definition |
|--------|-----------|
| **WNS** | Worst Negative Slack = min(slack) across paths; negative = violated |
| **TNS** | Total Negative Slack = sum of negative slacks only; 0 = all paths met |
| **Slack** | Required Time - Arrival Time; positive = timing met |
| **Criticality** | 1 - (slack / max_slack); 1.0 = critical, 0 = non-critical |

## Parasitics Models

Three update modes:

1. **`update_rc()`** - Full RC extraction (slowest, most accurate)
2. **`update_rc_flute()`** - FLUTE-based Steiner tree (fast, good accuracy)
3. **`update_rc_spef()`** - SPEF file (for real extracted parasitics)

Recommended: Use FLUTE during optimization, SPEF for final verification.

## Integration with Placement

**Timing-Driven Optimization**:
```
merged_wl_timing_grad = alpha * wl_grad + beta * timing_grad

where:
  timing_grad = -criticality × gradient_direction
  criticality = report_criticality(pin_id)
```

**Dynamic Weight Adjustment**:
```
alpha(iter) = timing_init_weight × (1 - decay_factor)^iter
beta(iter) = 1 - alpha(iter)
```

## Performance

**On 100K cell design**:
- Initialization: ~2-5 seconds
- Per-iteration timing analysis: 50-200 ms
- Suitable for integration in placement loop (every N iterations)

**Memory Usage**:
- GPU: 2-8 GB depending on circuit size
- Host: 1-2 GB for graph structures

## Limitations

- Simple RC model (not accounting for all coupling capacitance)
- Ignores frequency-dependent effects
- No consideration of power supply noise
- Steiner tree estimation less accurate than full routing

## Related Components

- Input: Cell geometry from `io_parser`
- Gradients feed: `wirelength_timing_cuda` for merged optimization
- Used in `src/core/timing_opt.py` for weight scheduling
- Integrated with `run_placement_nesterov.py` timing loop

## References

- Liberty file format: Cell timing characteristics
- SDC format: Timing constraints
- SPEF format: Detailed RC parasitics
