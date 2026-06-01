# DMP Fanout Wave and Arc-Level Backward Plan

Date: 2026-05-29

Scope: proposed DMP timing rewrite for better fanout crossing parallelism and
arc-level backward RAT propagation. This note is intentionally a living plan;
edit it as the design changes.

## Current Baseline

Current forward gate kernel shape:

- `dmpGateKernel` launches one lane per `gate_arc * DMP_PIN_GROUP_SIZE`.
- Each gate arc lane builds a local `DmpDriverWave`.
- It then loops over all net fanouts of the gate output pin.
- For every fanout net arc, it solves load crossing and atomically updates
  `pinSlew` and net `arcDelay`.

Current backward kernel shape:

- `dmpBackwardKernel` launches one lane per `pin * DMP_PIN_GROUP_SIZE`.
- Each pin group serially loops over `pin_forward_arc_list`.
- For gate arcs it writes candidate RATs into shared memory, then lane 0
  updates the source pin RAT.
- In mempool_cluster, backward max time is dominated by L0.

## mempool_cluster Profile Facts

Profile artifact:

`result/codex_mempool_cluster_level_profile_20260528/all_case/xplace_logs/visible/mempool_cluster.direct_route.log`

Key level maxima:

| Metric | Max | Level |
| --- | ---: | ---: |
| pins | 3,703,100 | L3 |
| arcs | 5,150,242 | L3 |
| gate arcs | 4,685,536 | L38 |
| net arcs | 3,703,100 | L3 |
| direct net arcs | 1,082,808 | L1 |
| gate-net pairs | 7,406,260 | L2 |
| pair lanes | 59,250,080 | L2 |
| valid pair lanes | 29,625,040 | L2 |
| gate lanes | 37,484,288 | L38 |

Totals:

- levels: 259
- total pins: 43,896,051
- total arcs: 180,923,409
- total gate arcs: 149,740,072
- total net arcs: 31,183,337
- total direct net arcs: 1,083,502
- total gate-net pairs: 212,932,016
- total valid pair lanes: 851,728,064
- total invalid pair lanes: 851,728,064

## Proposed Forward Parallel Mode

Goal: split gate waveform construction from fanout load crossing.

### Kernel 1: Build Gate Wave

Proposed kernel:

`dmpBuildGateWaveKernel`

Work item:

- one thread per `level_gate_arc * DMP_PIN_GROUP_SIZE`

Responsibilities:

- read `gate_arc_id = level_gate_arc_list[gate_arc_pos]`
- compute `lane`, `el`, `from_attr`, `to_attr`, `input_rf`, `output_rf`
- build `DmpGateArcMeta`
- compute `DmpDriverWave` and gate delay
- store compact driver wave to a level-local scratch array at
  `scratch_slot = gate_arc_pos * DMP_PIN_GROUP_SIZE + lane`
- write gate `arcDelay`
- update gate output pin AT winner
- update gate output pin slew winner

The local heavyweight objects should die before the kernel exits:

- `DmpDriverWave`
- `DmpGateArcMeta`
- `DmpDriverThresholds`
- local delay/slew solve variables

### Kernel 2: Fanout Crossing

Proposed kernel:

`dmpFanoutCrossingKernel`

Work item:

- one thread per `level_net_arc * NUM_ATTR`

Responsibilities:

- read `net_arc_id`
- derive `driver_pin = timing_arc_from_pin_id[net_arc_id]`
- find the level-local group of gate arcs that drive this pin
- enumerate all cell input gate arcs for that driver pin
- read compact wave scratch for each candidate gate arc/lane
- solve crossing for the load pin
- locally choose the best delay/slew for this `(net_arc, attr)`
- write `arcDelay[net_arc]` once
- write/update `pinSlew[load_pin]`
- update load pin AT winner if needed by the forward schedule

Important design point:

This mode avoids storing one wave per gate-net pair. A wave is stored once per
gate arc lane, then reused by every fanout crossing thread for that driver pin.

## New GPU Memory Budget

`DMP_PIN_GROUP_SIZE = 8`.

Allocate by the maximum number of level gate arcs, because only gate arcs
produce driver waves:

```text
wave_lanes = max_level_gate_arcs * 8
           = 4,685,536 * 8
           = 37,484,288 lanes
```

Do not allocate by pin count, net arc count, or total level arc count. The
compact key is the gate arc's level-local ordinal:

```text
scratch_slot = level_gate_arc_pos * DMP_PIN_GROUP_SIZE + lane
```

Fanout crossing must recover the producer wave through this level-local arc
ordinal, not through a global pin id.

### Wave Storage Options

Current `DmpDriverWave` is double-heavy:

- `DmpWaveCoeffs`: 7 doubles
- `t0`, `dt`, `vo_delay`, `vo_upper_time`: 4 doubles
- `vo_slew`: float
- `alg`: int

The raw struct is about 96 bytes per lane.

| Storage | Bytes/lane | Max level gate arcs |
| --- | ---: | ---: |
| compact wave, 12 floats only | 48 B | 1.68 GiB |
| compact wave, 12 floats + `uint8_t meta` | 49 B | 1.71 GiB |
| compact wave, 12 floats + `int32_t meta` | 52 B | 1.82 GiB |
| compact wave aligned to 64B | 64 B | 2.23 GiB |
| compact wave + thresholds | 72 B | 2.51 GiB |
| compact aligned to 80B | 80 B | 2.79 GiB |
| raw `DmpDriverWave` style | 96 B | 3.35 GiB |

Revised recommendation:

- Do not store thresholds or `library_id` in wave scratch.
- Derive driver thresholds in `dmpFanoutCrossingKernel` from
  `gate_arc_id + lane -> timing_id -> timingLibraryId() -> driverLibraryThresholds()`.
- Store only core wave state plus a compact valid/algorithm tag.
- Budget with `uint8_t meta`: about 49 bytes per lane, or 1.71 GiB for the
  mempool_cluster max-gate-arc level.
- Budget with `int32_t meta`: about 52 bytes per lane, or 1.82 GiB for the
  mempool_cluster max-gate-arc level.

Rationale: thresholds are cheap to derive compared with load crossing and do
not need to live across kernels. Storing them costs about 20 bytes per lane,
roughly 0.70 GiB at the mempool_cluster max-gate-arc level.

## Recommended Packed Scratch Layout

Use a compact AoS struct array for bring-up. For this access pattern each
fanout crossing reads nearly the whole wave, so keeping one wave in one compact
record is simpler and has good locality. Do not use `alignas(16)`, `float4`, or
packed 49-byte structs in the first implementation; those either pad the record
to 64 bytes or create unaligned array elements.

Storage contract:

- one scratch slot per `(level_gate_arc_pos, lane)`
- `lane` is the existing DMP timing lane, `0..7`
- slot formula:

  ```text
  slot = level_gate_arc_pos * DMP_PIN_GROUP_SIZE + lane
  ```

- capacity formula:

  ```text
  capacity = max_level_gate_arcs * DMP_PIN_GROUP_SIZE
  ```

- fanout crossing recovers the producer slot through the level-local gate arc
  ordinal stored in the driver-pin grouping; no global-pin-sized wave table is
  allocated

Recommended bring-up type:

- one `DmpCompactDriverWave*` array
- default 4-byte alignment
- `int32_t meta` for valid/algorithm/debug bits

`int32_t meta` costs about 143 MiB at the mempool_cluster max gate level, only
about 107 MiB more than a separate `uint8_t` meta array, and keeps the first
implementation simple. In an AoS struct, `uint8_t meta` does not reduce the
record stride because the struct still rounds up to 4-byte alignment.

```cpp
struct DmpCompactDriverWave {
    float k0;
    float k1;
    float k2;
    float k3;
    float k4;
    float p1;
    float p2;
    float t0;
    float dt;
    float vo_delay;
    float vo_upper_time;
    float vo_slew;
    int32_t meta;

    CUDA_DEV_INLINE void clear();
    CUDA_DEV_INLINE void storeFrom(const DmpDriverWave& wave,
                                   bool valid,
                                   bool ideal_clock_arc);
    CUDA_DEV_INLINE bool valid() const;
    CUDA_DEV_INLINE int alg() const;
    CUDA_DEV_INLINE bool hasValidDriver() const;
    CUDA_DEV_INLINE DmpDriverWave toDriverWave() const;
    CUDA_DEV_INLINE double findLoadCrossingBisection(float elmore,
                                                     float vth,
                                                     float x1,
                                                     float x2,
                                                     int max_iter,
                                                     float x_tol) const;
};

static_assert(sizeof(DmpCompactDriverWave) == 52);
```

The implementation should follow the existing DMP style used by
`DmpGateArcMeta`, `DmpRcParams`, and `DmpDriverWave`: keep the data fields in a
plain compact struct, and put device-side behavior next to the data as
`CUDA_DEV_INLINE`/`__device__` member functions. Avoid scattering raw bit
decoding and crossing formulas across kernels.

Recommended method responsibilities:

- `clear()`: initialize an invalid slot.
- `storeFrom()`: copy the double-heavy `DmpDriverWave` into compact float
  fields and pack `meta`.
- `valid()`, `alg()`, `hasValidDriver()`: hide meta bit layout from kernels.
- `toDriverWave()`: optional debug/bring-up bridge to reuse old helper code
  while validating.
- `findLoadCrossingBisection()`: final float bisection crossing path used by
  fanout crossing.

This is object-oriented in the pragmatic C++ sense: data and operations are
encapsulated together. It should not use CUDA-unfriendly dynamic OOP features
such as virtual functions, inheritance, heap allocation inside device code, or
runtime polymorphism.

Per lane:

- 12 floats = 48 bytes
- one int32 meta = 4 bytes
- total = 52 bytes

If `meta` is a `uint8_t` array:

- 12 floats = 48 bytes
- one byte meta = 1 byte
- total = 49 bytes

This 49-byte layout only applies to a split meta array. It should be treated as
a later memory squeeze, not the first implementation.

`meta` packing:

```text
bits 0-1: alg
bit 2: valid wave
bit 3: ideal clock arc if needed
remaining bits: spare/debug
```

Threshold derivation in fanout crossing:

```cpp
const int el = lane >> 2;
const int timing_id = timing_arc_id_map[producer_arc_id * 2 + el];
const int driver_library_id = timingLibraryId(timing_id);
driverLibraryThresholds(driver_library_id,
                        load_attr,
                        driver_vth,
                        driver_vl,
                        driver_vh,
                        driver_derate);
```

`load_attr` is the same output attribute used by the wave lane:

```cpp
load_attr = (el << 1) | output_rf;
```

This preserves the information needed by `loadDelaySlewFromDriverWave()`:

- crossing thresholds: `vth`, `vl`, `vh`
- slew derate
- `driver_library_id` for `thresholdAdjust()`

Potential later compression:

- Pack `alg` and `valid` into `uint8_t`.
- Split PI and ZERO_C2 waves if `p2/k4` storage becomes worth optimizing.

## Live Range Minimization

Allocate wave scratch once at DMP timing setup:

```text
capacity = max_level_gate_arcs * DMP_PIN_GROUP_SIZE
```

Reuse it for every forward level.

Within each level:

1. `dmpBuildGateWaveKernel` writes only slots `[0, num_level_gate_arcs * 8)`.
2. `dmpFanoutCrossingKernel` reads those slots and writes net delay/slew/AT.
3. Scratch contents become dead when the level finishes.

Do not keep per-level wave scratch across levels. Do not allocate per-pair wave
scratch.

The following should stay local to build-wave threads and never be stored:

- `DmpGateArcMeta`
- LUT query temporary variables
- `DmpRcParams`
- local root solve state
- intermediate `A/B/D`
- local `gate_delay`

The following are worth storing in compact wave scratch:

- wave coefficients needed by load crossing: `k0..k4`, `p1`, `p2`
- time state: `t0`, `dt`, `vo_delay`, `vo_upper_time`, `vo_slew`
- `alg` and valid bit

The following should be derived in the fanout crossing kernel rather than
stored:

- `vth`, `vl`, `vh`, `derate`
- `library_id`
- `timing_id`

## Driver Pin Grouping for Fanout Crossing

To avoid gate-net pair storage, fanout crossing needs to map a net arc's driver
pin to the gate arcs that produced waves for that pin at the current level.

Level-local grouping:

```text
driver_pin_ids[group]
driver_gate_begin[group]
driver_gate_end[group]
group_gate_arc_list[pos]
group_gate_arc_level_pos[pos]
```

`group_gate_arc_level_pos[pos]` is the compact ordinal in the current
`level_gate_arc_list`. The wave slot is derived, not stored separately:

```text
wave_slot = group_gate_arc_level_pos[pos] * DMP_PIN_GROUP_SIZE + lane
```

Options for lookup from `driver_pin` to `group`:

1. Binary search `driver_pin_ids`
   - lowest memory
   - good first implementation
   - slower if many fanout crossing threads repeat the same search

2. Dense `pin_to_driver_group[num_pins]`
   - extra memory: `43,896,051 * 4 = 0.164 GiB`
   - with stamp array: another 0.164 GiB
   - faster lookup

Recommendation:

- First implementation: binary search for correctness and memory simplicity.
- If profiling shows lookup bottleneck: add dense `pin_to_driver_group`.

## Backward Arc-Level Plan

Goal: replace pin-level serial fanout loop with arc-level parallelism.

Current bottleneck:

- mempool_cluster `dmpBackwardKernel`: 258 launches, 1632.567 ms total
- max backward level: 1535.264 ms at L0

### Schedule

Build a backward arc schedule grouped by source pin level:

```text
backward_arc_list
backward_arc_end[level]
```

For each arc, assign it to the level of `timing_arc_from_pin_id[arc_id]`.

Extra memory if stored for all arcs:

```text
num_arcs * int32 = 180,923,409 * 4 = 0.674 GiB
```

Gate/net split is possible, but total storage is similar.

### Winner Buffer

Reuse `pin_at_winner` after forward AT finalization:

- forward no longer needs `pin_at_winner` once all forward levels complete
- reset it to zero
- reinterpret it as `pin_rat_winner`
- no new 1.31 GiB winner buffer needed

### Kernel 1: Backward Arc Propagation

Proposed kernel:

`dmpBackwardArcKernel`

Work item:

- one thread per `backward_arc * DMP_PIN_GROUP_SIZE`

For net arc:

```text
to_slot = to_pin * NUM_ATTR + attr
from_slot = from_pin * NUM_ATTR + attr
candidate = pinRat[to_slot] - arcDelay[net_arc_delay_slot]
atomic update winner[from_slot]
```

For gate arc:

```text
lane -> el/from_attr/to_attr
to_slot = to_pin * NUM_ATTR + to_attr
from_slot = from_pin * NUM_ATTR + from_attr
candidate = pinRat[to_slot] - arcDelay[gate_arc_delay_slot]
skip timing constraint arcs
atomic update winner[from_slot]
```

### Kernel 2: RAT Winner Finalize

Proposed kernel:

`dmpRatWinnerFinalizeKernel`

Work item:

- one thread per `pin * NUM_ATTR` in the current backward level

Responsibilities:

- decode packed RAT winner
- write `pinRat[slot]`
- optionally write RAT predecessor metadata if path reporting needs it later
- reset winner slot to zero

### RAT Winner Semantics

Current code uses:

```cpp
if (isnan(pinRat[from_slot]) || ((pinRat[from_slot] < rat) ^ el)) {
    atomicExch(&pinRat[from_slot], rat);
}
```

This is race-prone. Arc-level rewrite should use packed atomic selection like
AT winner.

Rule:

```text
early RAT: pick max
late RAT:  pick min
```

Use `uint64_t` packed value:

```text
high 32 bits: ordered float key
low 32 bits: payload, e.g. arc_id and attr
```

Then `atomicMax()` selects the correct candidate deterministically.

## Total New Memory Budget

Recommended first implementation with the compact `uint8_t` meta:

| Item | Approx memory |
| --- | ---: |
| forward compact wave scratch, 49 B/lane | 1.71 GiB |
| backward arc schedule, int32 per arc | 0.67 GiB |
| optional dense `pin_to_driver_group` | 0.16 GiB |
| optional group stamp | 0.16 GiB |

Minimum recommended:

```text
1.71 GiB + 0.67 GiB = 2.38 GiB
```

With dense lookup:

```text
2.54 GiB
```

With dense lookup + stamp:

```text
2.70 GiB
```

If `int32_t meta` is used for easier bring-up, the forward scratch is about
1.82 GiB and the minimum total becomes about 2.49 GiB.

Current fixed mempool_cluster run used roughly 26.3 GiB after timing scratch
allocation on a 79.252 GiB GPU, so the recommended plan should still be well
within 80GB memory.

## Implementation Order

1. Add env flag for new path:

   ```text
   DMP_FANOUT_WAVE_SPLIT=1
   ```

2. Build level-local driver pin groups on host as part of forward schedule.

3. Allocate `DmpFanoutWaveScratch` with capacity:

   ```text
   max_level_gate_arcs * DMP_PIN_GROUP_SIZE
   ```

4. Implement `dmpBuildGateWaveKernel`.

5. Implement `dmpFanoutCrossingKernel` with binary search driver group lookup.

6. Validate forward timing against old DMP path:

   - ariane
   - mempool_cluster
   - compare WNS/TNS
   - optionally compare sampled `arcDelay`, `pinSlew`, `pinAT`

7. Profile fanout crossing.

8. If binary search is bottleneck, add dense `pin_to_driver_group`.

9. Implement backward arc-level schedule.

10. Reuse `pin_at_winner` as RAT winner after forward.

11. Implement `dmpBackwardArcKernel` and `dmpRatWinnerFinalizeKernel`.

12. Validate backward:

    - compare WNS/TNS
    - compare sampled `pinRAT` and endpoint slack
    - check deterministic repeatability

13. Remove or gate old backward path only after correctness and performance
    are stable.

## Implementation Result: First Forward Split Prototype

Date: 2026-05-29

Status: code reverted on request after validation because the prototype was
slower than the existing `dmpGateKernel` path. This section is a historical
result note, not an active implementation state.

Implemented an env-gated prototype:

```text
DMP_FANOUT_WAVE_SPLIT=1
```

Code shape:

- `DmpCompactDriverWave` is a compact 52-byte AoS record with device member
  functions for clear/store/valid/alg/float bisection crossing.
- `dmpBuildGateWaveKernel` computes gate delay, gate output slew/AT winner,
  and writes compact wave scratch.
- `dmpFanoutWaveKernel` processes fanout net attrs, enumerates producer gate
  arcs for the driver pin, derives thresholds from `gate_arc_id + lane`, and
  solves float bisection crossing.
- Driver groups are built only when the env flag is enabled; old direct path
  does not allocate these extra grouping arrays.
- Wave scratch is allocated only for forward propagation and freed before
  backward RAT propagation.

Validation:

- `visible/ariane`, split enabled:
  - pass
  - no CUDA illegal memory access
  - WNS/TNS diff versus reference stayed within pass threshold
  - wave scratch: 0.046 GiB
  - forward timing: 504.446 ms
  - build-wave: 36.403 ms
  - fanout-wave: 444.788 ms

- `visible/ariane`, split disabled same build:
  - pass
  - forward timing: 116.013 ms
  - old `dmpGateKernel`: 97.292 ms

- `visible/mempool_cluster`, split enabled:
  - pass
  - no CUDA illegal memory access
  - WNS/TNS diff versus reference stayed within pass threshold
  - wave scratch: 37,484,288 slots, 1,949,182,976 bytes, 1.815 GiB
  - fanout net arcs in split schedule: 30,099,835
  - driver groups: 16,137,322
  - forward timing: 6481.912 ms
  - build-wave: 2692.887 ms
  - fanout-wave: 3690.766 ms
  - backward timing unchanged: 1633.442 ms

Baseline for `visible/mempool_cluster`, split disabled old profile:

- forward timing: 3785.326 ms
- old `dmpGateKernel`: 3686.806 ms
- backward timing: 1632.567 ms

Conclusion:

- The memory model is correct: compact wave scratch matches the planned
  `max_level_gate_arcs * 8 * 52B` budget.
- Correctness is acceptable on ariane and mempool_cluster.
- The first fanout-pin/net-attr parallel mode is slower than the old direct
  gate kernel:
  - ariane forward: 116.013 ms -> 504.446 ms
  - mempool_cluster forward: 3785.326 ms -> 6481.912 ms
- The bottleneck is `dmpFanoutWaveKernel`; the current mapping serializes too
  much work inside each fanout net attr thread and pays fixed float-bisection
  cost for every candidate.

Recommended next change:

- Keep `DMP_FANOUT_WAVE_SPLIT` off by default.
- Keep compact wave scratch code as a correctness prototype.
- Redesign fanout crossing before making it the default:
  - either increase parallelism to candidate/lane level with a compact schedule,
  - or keep the old gate-lane fanout traversal but reuse compact wave only where
    it removes recomputation,
  - and profile bisection iteration count or a guarded Newton/bisection hybrid.

## Open Questions

- Whether `library_id` must be stored in wave scratch or whether
  `thresholdAdjust()` can be rewritten to use only precomputed threshold values.
- Whether fanout crossing should update AT winner directly, or whether a
  separate net winner/finalize stage remains cleaner.
- Whether driver pin grouping should be host-built every schedule build or
  generated on GPU to avoid host memory pressure.
- Whether path reporting requires RAT predecessor metadata after the backward
  rewrite.
- Whether compact wave values are accurate enough as `float`; current
  `DmpDriverWave` stores coefficients as `double`, but load crossing already
  has a float-based evaluator path.
