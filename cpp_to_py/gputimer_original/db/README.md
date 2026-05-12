# GPUTimer Database

CPU-resident timing graph topology and GPU tensor storage for timing values. Acts as the bridge between the IO parser (`rawdb`/`gpdb`) and the GPU timing engine.

## Files

### `GTDatabase.h` / `GTDatabase.cpp`

**`GTDatabase`** — CPU-side timing graph and liberty library.

Constructed via `GTDatabase(rawdb, gpdb, timing_raw_db)` then `ExtractTimingGraph()`.

#### CPU Vectors (topology)
| Member | Type | Content |
|--------|------|---------|
| `pin_names` | `vector<string>` | Full pin names; index = pin ID used everywhere |
| `timing_arc_from_pin_id` | `vector<int>` | Source pin of each arc |
| `timing_arc_to_pin_id` | `vector<int>` | Sink pin of each arc |
| `arc_types` | `vector<int>` | 0 = net arc, 1 = cell (liberty) arc |
| `timing_arc_id_map` | `vector<int>` | `[arc_id*2+el]` → liberty timing ID for early/late |
| `primary_inputs` | `vector<int>` | Pin IDs of primary inputs (PI) |
| `primary_outputs` | `vector<int>` | Pin IDs of primary outputs (PO) |
| `endpoints_id` | `vector<int>` | Pin IDs of timing endpoints |

#### Unit / Scale Fields
| Member | Meaning |
|--------|---------|
| `time_unit` | SI seconds per internal time unit (e.g. `1e-12` for ps) |
| `cap_unit` | SI farads per internal cap unit (e.g. `1e-15` for fF) |
| `rawdb` | Reference to `db::Database` (contains `dieHX`, `dieHY`, `DBU_Micron`) |

#### Key Method
- `ExtractTimingGraph()` — builds pin/arc vectors by iterating `gpdb.getPinNames()` and rawdb liberty cell data; assigns `pin_names = gpdb.getPinNames()`
- `readSdc(sdc)` — parses clock period and I/O delay constraints into `clock_period`, PI/PO constraint arrays

---

**`TimingTorchRawDB`** — GPU PyTorch tensors for placement-dependent timing values.

Constructed from tensors passed in from Python (via `gputimer.create_timing_rawdb()`). Stores the live GP state that the timer reads each iteration.

#### GPU Tensors
| Tensor | Shape | Content |
|--------|-------|---------|
| `pinAT` | `[num_pins, NUM_ATTR]` | Arrival time per pin per corner |
| `pinRAT` | `[num_pins, NUM_ATTR]` | Required arrival time |
| `pinSlew` | `[num_pins, NUM_ATTR]` | Transition time (slew) |
| `pinLoad` | `[num_pins, NUM_ATTR]` | Total load capacitance |
| `arcDelay` | `[num_arcs, 2*NUM_ATTR]` | Arc delays (see layout below) |
| `x`, `y` | `[num_nodes]` | Node lower-left X/Y in scaled coords |
| `pin_offset_x/y` | `[num_pins]` | Pin offset from node lower-left |
| `pin2node_map` | `[num_pins]` | Maps each pin ID to its parent node ID |

`NUM_ATTR = 4` corners: early-rise (0), early-fall (1), late-rise (2), late-fall (3).

#### arcDelay Layout
- **Net arcs** (`arc_type=0`): corners `{er,ef,lr,lf}` at offsets `{0,3,4,7}` within `[arc_id*8 .. arc_id*8+7]`
- **Cell arcs** (`arc_type=1`): `index = el*4 + irf*2 + orf` → full `[arc_id*8 .. arc_id*8+7]`

#### Commit / Snapshot Methods
- `commit_from(node_lpos)` — copies new node positions into `x`, `y`; used before each timing update
- `get_curr_cposx/y()` — center position = lower-left + size/2
- `get_curr_lposx/y()` — lower-left position

## Usage Pattern

```python
# Create from Python tensors
timing_raw_db = gputimer.create_timing_rawdb(
    node_lpos_init, node_size, pin_rel_lpos,
    pin_id2node_id, pin_id2net_id,
    node2pin_list, node2pin_list_end,
    hyperedge_list, hyperedge_list_end,
    net_mask,
    num_movable_nodes, scale_factor, microns,
    wire_resistance_per_micron, wire_capacitance_per_micron
)

# Build timing graph
gtdb = gputimer.GTDatabase(rawdb, gpdb, timing_raw_db)
# (GTDatabase.ExtractTimingGraph() called internally by create_gputimer)

# Each placement iteration:
timing_raw_db.commit_from(node_lpos)   # update positions
timer.update_rc_flute(node_lpos)        # recompute RC
timer.update_timing()                   # propagate delays
```
