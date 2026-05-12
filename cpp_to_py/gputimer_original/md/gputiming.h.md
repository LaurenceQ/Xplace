# gputiming.h

## Purpose
Device-side utility functions and LUT allocation for GPU timing computation.

## Interpolation Utilities

### lower_bound() [Device]
Binary search to find insertion point in sorted array.
```cpp
template <typename T>
__device__ int lower_bound(T *arr, int size, T val)
```
Returns index where value should be inserted to maintain sorted order.

### interpolate() [Device]
Linear interpolation between two points.
```cpp
template <typename T>
__device__ float interpolate(T x1, T x2, T y1, T y2, T x)
```
**Formula**: `y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)`

## GPULutAllocator Class

Manages precomputed Liberty LUT data on GPU for fast constraint/delay/slew lookup.

### LUT Structure

**6 LUTs per timing arc**:
1. Cell delay rising edge
2. Cell delay falling edge
3. Transition rising edge
4. Transition falling edge
5. Setup constraint
6. Hold constraint

Each LUT has:
- **X axis**: Input slew (ns)
- **Y axis**: Output capacitance (F)
- **Table**: 2D lookup table [num_y_values][num_x_values]

### Host Allocation

```cpp
void AllocateBatch(vector<TimingArc *> timings)
```

1. Iterate all timing arcs in Liberty
2. For each arc, examine all 6 LUTs
3. Accumulate total array sizes
4. Allocate unified arrays for all LUTs

**Host arrays** (CPU memory):
- `num_x[lut_id]` — X axis size
- `num_y[lut_id]` — Y axis size
- `num_table[lut_id]` — Table size
- `x_array[x_offset[lut_id]:x_offset[lut_id+1]]` — X values
- `y_array[y_offset[lut_id]:y_offset[lut_id+1]]` — Y values
- `table_array[table_offset[lut_id]:table_offset[lut_id+1]]` — Table values
- `allocated[lut_id]` — Whether this LUT is valid

### GPU Upload

```cpp
void CopyToGPU()
void CopyToGPU(GPULutAllocator *d_allocator)
```

Copies all host arrays to GPU. Device pointers:
- `d_num_x`, `d_num_y`, `d_num_table` — Sizes on GPU
- `d_x_array`, `d_y_array`, `d_table_array` — Data on GPU
- `d_x_offset`, `d_y_offset`, `d_table_offset` — Offsets on GPU
- `d_allocated` — Validity flags on GPU

### Timing Metadata

**Per timing arc** (host & GPU):
- `timing_sense[timing_id]` — Positive/negative/None
- `lut_template_var[timing_id]` — Index pair (el, orf)
- `is_rising_edge_triggered[timing_id]` — Clock edge
- `is_falling_edge_triggered[timing_id]` — Clock edge
- `is_constraint[timing_id]` — Whether arc is a test constraint

### Query Interface

```cpp
__device__ float query(int timing_id, int frf, int rf, float sr, float sc, int idx)
```

**Parameters**:
- `timing_id` — Timing arc to query
- `frf` — From pin rise(0) or fall(1)
- `rf` — To pin rise(0) or fall(1)
- `sr` — Input slew (ns)
- `sc` — Output capacitance (F)
- `idx` — LUT type (0=delay, 1=slew, 2=constraint)

**Returns**: Interpolated LUT value

**Algorithm**:
1. Select LUT based on timing_id + frf + rf + idx
2. Binary search X and Y axes to find surrounding values
3. Interpolate 2D table value
4. Return interpolated result
