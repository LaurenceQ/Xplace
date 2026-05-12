# GTDatabase.h

## Purpose
Central timing database aggregating Liberty cell timing, circuit netlist, and timing constraints.

## Core Classes

### Clock
Represents a clock domain.
```cpp
class Clock {
    std::string _name;        // Clock name
    float _period;            // Period (ns)
    int _source_id;           // Driving pin
};
```

### STAPin
Per-pin timing information.
```cpp
class STAPin {
    vector<index_type> timing_arc_in;   // Arcs into this pin
    vector<index_type> timing_arc_out;  // Arcs from this pin
    set<index_type> fanin_pin_ids;      // Fanin pin IDs
    set<index_type> fanout_pin_ids;     // Fanout pin IDs
};
```

### GTDatabase
Main timing database aggregating all components.

## Unit Conversions

```cpp
float res_unit;      // Resistance unit (ohm)
float cap_unit;      // Capacitance unit (F)
float time_unit;     // Time unit (s)
```

Used to convert between file units and internal representation:
- `value_internal = value_file / unit`
- `value_file = value_internal * unit`

**Example**: SPEF in `ohm` and `pF`, database in `kohm` and `F`:
```
res_unit = 1000.0       // 1 kohm
cap_unit = 1e-12        // 1 pF (in F)
time_unit = 1e-12       // 1 ps
```

## Optional Overrides from SDC/SPEF

```cpp
std::optional<float> sdc_res_unit;
std::optional<float> sdc_cap_unit;
std::optional<float> sdc_time_unit;

std::optional<float> spef_res_unit;
std::optional<float> spef_cap_unit;
std::optional<float> spef_time_unit;
```

Priority: SPEF > SDC > Liberty defaults

## Methods

### Graph Extraction
```cpp
void ExtractTimingGraph()
```
Parse Liberty, build timing arc DAG, extract topological levels.

### File I/O
```cpp
void readSpef(const std::string& file)
void readSdc(sdc::SDC& sdc)
```

### Constraint Processing
```cpp
void _read_sdc(sdc::SetInputDelay&)
void _read_sdc(sdc::SetOutputDelay&)
void _read_sdc(sdc::SetDrivingCell&)
void _read_sdc(sdc::SetInputTransition&)
void _read_sdc(sdc::SetLoad&)
void _read_sdc(sdc::CreateClock&)
void _read_sdc(sdc::SetUnits&)
```

### Timing Redundancy
```cpp
bool is_redundant_timing(const TimingArc* timing_arc, Split el)
```
Check if timing arc can be pruned (dominated by another arc).

## Members (Partial)

```cpp
db::Database& rawdb;           // LEF/DEF/Verilog database
gp::GPDatabase& gpdb;          // Placement database
TimingTorchRawDB& timing_raw_db;

std::array<std::shared_ptr<CellLib>, MAX_SPLIT> cell_libs_;
```

## Related Classes

- `CellLib` — Liberty cell library
- `LibertyCell` — Cell definition
- `LibertyPort` — Cell port (pin)
- `TimingArc` — Arc from one port to another
- `Lut` — Lookup table for delay/slew/constraint
