# propagate_infer.cu

## Purpose
Timing propagation kernel with ML-predicted delays. Reads arcDelay pre-loaded by `read_infer()` and propagates AT/RAT through circuit.

## Critical: Net Arc Transition Validity

**For net arcs, input and output transitions must MATCH:**
- rise → rise ✓ (valid, arcDelay indices 0 and 4)
- fall → fall ✓ (valid, arcDelay indices 3 and 7)
- rise → fall ✗ (INVALID, indices 1 and 5 stored as NaN)
- fall → rise ✗ (INVALID, indices 2 and 6 stored as NaN)

**This is enforced by the NaN check in `propagateInferAT()`** — invalid transitions have NaN in arcDelay and kernel returns early.

**Cell arcs CAN have irf != orf** (e.g., inverters: rise→fall). All 8 indices are valid for cell arcs.

## Kernels

### propagateInferAT() [Device Function]
Forward pass: Compute AT using ML-predicted arcDelay.

```cpp
void propagateInferAT(
    index_type arc_id,           // Arc to propagate through
    index_type from_pin_id,      // Source pin
    index_type to_pin_id,        // Sink pin
    float *pinAt,                // [num_pins * 4] arrival times
    float *arcDelay,             // [num_arcs * 8] ML-predicted delays
    index_type *at_prefix_pin,   // Criticality tracking
    index_type *at_prefix_arc,
    index_type *at_prefix_attr
)
```

**Logic:**
1. Extract corner index i = idx & 0b111 (8 threads per arc)
2. Compute irf (from pin rise/fall), orf (to pin rise/fall)
3. Skip if from_pin AT is NaN
4. **Skip if arcDelay[arc_id * 8 + i] is NaN** ← Handles invalid net transitions
5. Compute AT = from_pin_AT + delay
6. Update to_pin_AT if better

### propagateInferTest() [Device Function]
Forward pass: Compute RAT at timing constraint points using library LUT (not ML).

```cpp
void propagateInferTest(
    index_type arc_id,
    index_type test_id,
    index_type from_pin_id,      // Test input pin
    index_type to_pin_id,        // Test output pin
    ...
    float *pinSlew,              // ML-predicted or LUT slew
    float *testConstraint,       // Library constraint (LUT queried)
    ...
)
```

**Logic:**
1. Query LUT for constraint value based on slew
2. Compute RAT = AT + constraint (setup) or RAT = AT - constraint (hold)

### propagateInferRAT() [Device Function]
Backward pass: Compute RAT from fanout using ML-predicted arcDelay.

```cpp
void propagateInferRAT(
    index_type arc_id,           // Arc to propagate backward through
    int arc_type,                // 0=net, 1=cell
    ...
)
```

**Logic:**
- **Net arc** (arc_type=0): RAT[from] = RAT[to] - delay
- **Cell arc** (arc_type=1):
  - Non-constraint: RAT[from] = RAT[to] - delay
  - Constraint (test): RAT[from] = AT[from] ± slack

Uses shared memory to batch RAT updates with constraint-aware conditions.

### propagatePinInferAT() [Global Kernel]
```cpp
__global__ void propagatePinInferAT(
    index_type *level_list,           // Pins at this level
    int num_pins_level,               // # pins at this level
    ...
)
```

**Launch config**: `num_pins_level * 8` threads (8 per pin for corners)

**Per pin:**
- Iterate fanin arcs via CSR
- Call `propagateInferAT()` for each arc
- Call `propagateInferTest()` if test constraint arc

### propagatePinInferRAT() [Global Kernel]
```cpp
__global__ void propagatePinInferRAT(
    index_type *level_list,           // Pins at this level (reversed)
    int num_pins_level,
    ...
)
```

**Launch config**: `num_pins_level * 8` threads

**Per pin (in reverse topological order):**
- Use shared memory to batch RAT values
- Iterate fanout arcs via CSR
- Call `propagateInferRAT()` for each arc
- Update pinRAT with constraint-aware logic

## Host Implementation

### propagate_infer_timing_impl()
```cpp
void propagate_infer_timing_impl(
    index_type *level_list,
    vector<int> level_list_end_cpu,    // Level boundaries (host)
    ...
)
```

**Forward pass** (levels 1 to N-2):
```
for level in 1...(num_levels-2):
    propagatePinInferAT<<<...>>>(level)
    cudaDeviceSynchronize()
```

**Backward pass** (levels N-3 down to 0):
```
for level in (num_levels-3)...0:
    propagatePinInferRAT<<<...>>>(level)
    cudaDeviceSynchronize()
```

## Key Differences from Standard Timing (propagate.cu)

| Feature | Standard | Inference |
|---------|----------|-----------|
| arcDelay source | LUT query (runtime) | ML prediction (pre-loaded) |
| Test constraint | LUT query (runtime) | LUT query (runtime) |
| Valid transitions | All 8 per arc | Net: 4 valid, Cell: all 8 |
| NaN handling | Skips undefined timings | Skips invalid + NaN arcDelay |

## Synchronization

- `cudaDeviceSynchronize()` after each level in both passes
- Critical: Ensures prior arcs complete before next level starts
