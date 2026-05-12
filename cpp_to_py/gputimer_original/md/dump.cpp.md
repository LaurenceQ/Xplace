# dump.cpp

## Purpose
Export complete timing graph to JSONL format for ML training. Each line is one JSON record (node/net_arc/cell_arc).

## Output Format

### Node Record
```json
{
  "type": "node",
  "node_id": 42,
  "pin_names": ["u1/q", "u2/a"],
  "features": {
    "is_pi": 0,
    "is_po": 0,
    "is_driver": 1,
    "x": 1234.5,
    "y": 5678.9,
    "num_fanout": 5,
    "total_fanout_cap": 1.5e-12,
    "wns_slack": 0.123,
    "tns_slack": 0.0
  },
  "labels": {
    "pin_0": [0.5, 0.6, 0.55, 0.65],  // [at_er, at_ef, at_lr, at_lf]
    "pin_1": [...]
  }
}
```

### Net Arc Record (Wire)
```json
{
  "type": "net_arc",
  "from": 10,  // from_pin_id
  "to": 20,    // to_pin_id
  "features": {
    "dx": 100.5,   // µm
    "dy": 200.3    // µm
  },
  "labels": [0.001, 0.0011, 0.0012, 0.00115]  // [delay_er, ef, lr, lf]
}
```

### Cell Arc Record
```json
{
  "type": "cell_out",
  "from": 54268,  // from_pin_id
  "to": 38454,    // to_pin_id
  "sdf_cond": "",
  "labels": [201.045837, 219.274277, 201.045837, 219.274277],  // [delay_er, ef, lr, lf]
  "features": {
    "lut_cell_delay_rise": {
      "x": [...],
      "y": [...],
      "table": [...]
    },
    "lut_cell_delay_fall": {...},
    "lut_trans_rise": {...},
    "lut_trans_fall": {...}
  }
}
```

## Time Units

**All times output in nanoseconds (ns)**:
- AT/RAT/Slew/Delay values converted via:
  ```cpp
  value_ns = value_internal * (gtdb.time_unit * 1e9)
  ```

Example: If `time_unit = 1e-12`:
- Internal value 1000 → 1.0 ns

## LUT Structure

Each cell arc includes 4 LUT tables (rise/fall delay, rise/fall slew):

```cpp
{
  "x": [0.1, 0.2, 0.3],     // Input slew axis (ns)
  "y": [0.5e-12, 1.0e-12],  // Output load axis (F)
  "table": [                 // 2D table [y_idx][x_idx]
    [1.5, 1.2, 0.8],
    [2.0, 1.7, 1.3]
  ]
}
```

## Node Features

| Field | Meaning |
|-------|---------|
| `is_pi` | Primary input (1/0) |
| `is_po` | Primary output (1/0) |
| `is_driver` | Has fanout arcs (1/0) |
| `x`, `y` | Cell position (µm) |
| `num_fanout` | # fanout edges |
| `total_fanout_cap` | Sum output capacitances (F) |
| `wns_slack` | Worst negative slack at this pin (ns) |
| `tns_slack` | Total negative slack sum (ns) |

## Pin Label Structure

Per pin (4 values for corners):
- Index 0: AT early-rise (ns)
- Index 1: AT early-fall (ns)
- Index 2: AT late-rise (ns)
- Index 3: AT late-fall (ns)

## Usage

```python
from cpp_to_py import gputimer

# After timing analysis
gputimer.update_timing()

# Export graph
gputimer.dump_timing_graph("output.json")

# In Python: Load and process
import json
with open("output.json") as f:
    for line in f:
        record = json.loads(line)
        if record["type"] == "node":
            print(f"Node {record['node_id']}: {record['pin_names']}")
        elif record["type"] == "cell_out":
            print(f"Cell arc {record['from']}->{record['to']}: delay={record['labels']}")
```

## Integration with ML

This dump format feeds directly into DGL graph loader:
- Nodes store pin features and timing labels
- Edges store arc features (spatial) and delay labels
- LUT tables stored as nested features for ML models to learn

See `TimingPredict/load_graph_from_dump.py` for conversion to DGL heterograph.
