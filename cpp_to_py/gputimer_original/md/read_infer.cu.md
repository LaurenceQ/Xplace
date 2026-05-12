# read_infer.cu

## Purpose
Load ML-predicted timing delays from .infer CSV file and populate GPU arcDelay array for inference-based timing propagation.

## Input File Format

### Section 1: Node Predictions (net arc delays)
```
# Node predictions
node_id,slew_er,slew_ef,slew_lr,slew_lf,net_delay_er,net_delay_ef,net_delay_lr,net_delay_lf
0,0.049687,0.049687,0.049687,0.049687,0.000400,0.000400,0.000400,0.000400
1,0.051234,0.051234,0.051234,0.051234,0.000425,0.000425,0.000425,0.000425
...
```

- **node_id**: Pin index in circuit
- **slew_er/ef/lr/lf**: Predicted slew (transition time) at this pin for 4 corners (ns)
- **net_delay_er/ef/lr/lf**: Predicted net delay from driver to this pin (ns)

### Section 2: Cell Edge Predictions (cell arc delays)
```
# Cell edge predictions
edge_id,from_pin_id,to_pin_id,cell_delay_er,cell_delay_ef,cell_delay_lr,cell_delay_lf
10,20,0.123,0.134,0.234,0.245
11,21,30,0.156,0.167,0.267,0.278
...
```

- **edge_id**: Cell arc index
- **from_pin_id**: Source pin
- **to_pin_id**: Sink pin
- **cell_delay_er/ef/lr/lf**: Predicted cell delay for 4 corners (ns)

## Unit Conversion

**Input**: Nanoseconds (ns)
**Output**: Internal units (relative to `gtdb.time_unit`)

```cpp
float time_to_internal = 1.0f / (gtdb.time_unit * 1e9f);
value_internal = value_ns * time_to_internal;
```

**Example**: If `time_unit = 1e-12` (1 picosecond):
- 1 ns = 1000 ps = 1000 internal units
- `time_to_internal = 1 / (1e-12 * 1e9) = 1000`

## GPU Array Layout

### arcDelay Format
For each arc: `arcDelay[arc_id * 8 + corner_index]`

**For net arcs** (only 4 valid corners):
- Index 0: rise-to-rise early
- Index 3: fall-to-fall early
- Index 4: rise-to-rise late
- Index 7: fall-to-fall late
- Indices 1, 2, 5, 6: Remain **NaN** (invalid for net arcs)

**For cell arcs** (all 8 corners valid):
- Index 0: rise-to-rise early
- Index 1: rise-to-fall early
- Index 2: fall-to-rise early
- Index 3: fall-to-fall early
- Index 4: rise-to-rise late
- Index 5: rise-to-fall late
- Index 6: fall-to-rise late
- Index 7: fall-to-fall late

### pinSlew Format
`pinSlew[pin_id * 4 + corner]` where corner ∈ [0,3] = {er, ef, lr, lf}

## Implementation Steps

### Step 1: Parse CSV File
```cpp
std::ifstream fin(infile);
std::string line;
int section = 0;  // 0=none, 1=node, 2=edge

while (std::getline(fin, line)) {
    if (line.find("# Node predictions") == 0) section = 1;
    if (line.find("# Cell edge predictions") == 0) section = 2;
    if (line[0] == '#') continue;

    // Parse based on section...
}
```

### Step 2: Storage on Host
```cpp
std::vector<std::pair<int, std::array<float, 4>>> slews;
std::vector<std::pair<int, std::array<float, 4>>> net_delays;
std::vector<std::tuple<int, int, std::array<float, 4>>> cell_delays;
```

### Step 3: GPU Memory Transfer (D2H)
```cpp
float* h_arcDelay = new float[num_arcs * 8];
cudaMemcpy(h_arcDelay, arcDelay, num_arcs * 8 * sizeof(float), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

float* h_pinSlew = new float[num_pins * 4];
cudaMemcpy(h_pinSlew, pinSlew, num_pins * 4 * sizeof(float), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
```

### Step 4: Host-Side Updates

**Slew updates:**
```cpp
for (const auto& [pin_id, slew] : slews) {
    for (int corner = 0; corner < 4; corner++) {
        h_pinSlew[pin_id * 4 + corner] = slew[corner] * time_to_internal;
    }
}
```

**Net delay updates** (find fanin net arcs via CSR):
```cpp
int arc_start = gtdb.pin_backward_arc_list_end[pin_id];
int arc_end = gtdb.pin_backward_arc_list_end[pin_id + 1];

for (int i = arc_start; i < arc_end; i++) {
    int arc_id = gtdb.pin_backward_arc_list[i];
    if (gtdb.arc_types[arc_id] != 0) continue;  // Skip non-net

    h_arcDelay[arc_id * 8 + 0] = delays[0] * time_to_internal;  // er
    h_arcDelay[arc_id * 8 + 3] = delays[1] * time_to_internal;  // ef
    h_arcDelay[arc_id * 8 + 4] = delays[2] * time_to_internal;  // lr
    h_arcDelay[arc_id * 8 + 7] = delays[3] * time_to_internal;  // lf
    // Indices 1,2,5,6 remain NaN
}
```

**Cell delay updates** (find matching cell arc):
```cpp
int arc_start = gtdb.pin_forward_arc_list_end[from_pin_id];
int arc_end = gtdb.pin_forward_arc_list_end[from_pin_id + 1];

for (int i = arc_start; i < arc_end; i++) {
    int arc_id = gtdb.pin_forward_arc_list[i];

    if (gtdb.timing_arc_to_pin_id[arc_id] != to_pin_id) continue;
    if (gtdb.arc_types[arc_id] != 1) continue;  // Must be cell

    for (int corner = 0; corner < 4; corner++) {
        int el = corner >> 1;  // Early(0) or Late(1)
        int orf = corner & 1;  // Rise(0) or Fall(1)

        // Map to all 8 indices with this el,orf combination
        for (int i = 0; i < 8; i++) {
            int i_el = i >> 2;
            int i_tel_rf = ((i & 4) >> 1) + (i & 1);
            int i_orf = i_tel_rf & 1;

            if (i_el == el && i_orf == orf) {
                h_arcDelay[arc_id * 8 + i] = delays[corner] * time_to_internal;
            }
        }
    }
}
```

### Step 5: GPU Memory Transfer (H2D)
```cpp
cudaMemcpy(arcDelay, h_arcDelay, num_arcs * 8 * sizeof(float), cudaMemcpyHostToDevice);
cudaDeviceSynchronize();  // CRITICAL: Ensures propagate_infer_timing sees updated data

cudaMemcpy(pinSlew, h_pinSlew, num_pins * 4 * sizeof(float), cudaMemcpyHostToDevice);
cudaDeviceSynchronize();

delete[] h_arcDelay;
delete[] h_pinSlew;
```

## Logging & Verification

- Logs parsed counts: "Parsed X slews, Y net delays, Z cell delays"
- Logs first 3 records from each section
- Logs update statistics: "Updated N slew pins, M net arcs, K cell arcs"
- Logs missing arc warnings (first 10 only)
- Reads back sample values from GPU to verify H2D transfer success

## Error Handling

- Gracefully skips malformed CSV lines (wrong field count)
- Logs missing fanin/fanout arcs as warnings
- Returns early if file cannot be opened
