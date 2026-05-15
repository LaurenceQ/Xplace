
#include "gputiming.h"
#include "utils.cuh"
#include "GPUTimer.h"

#include <cooperative_groups.h>
#include <cstdlib>
#include <vector>

namespace gt {

__device__ void propagateSlew(index_type arc_id,
                              index_type from_pin_id,
                              index_type to_pin_id,
                              float *pinSlew,
                              float *pinLoad,
                              float *pinImpulse,
                              float *pinRootDelay,
                              float *arcDelay,
                              int arc_type,
                              int *timing_arc_id_map,
                              GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    if ((arc_type == 0) && (i < NUM_ATTR)) {
        float si = pinSlew[from_pin_id * NUM_ATTR + i];
        if (isnan(si)) return;
        float imp = pinImpulse[to_pin_id * NUM_ATTR + i];
        float so = si < 0.0 ? -sqrt(si * si + imp * imp) : sqrt(si * si + imp * imp);
        pinSlew[to_pin_id * NUM_ATTR + i] = so;
    } else if (arc_type == 1) {
        int el = i >> 2;
        int fel_rf = i >> 1;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        int irf = fel_rf & 1;
        int orf = tel_rf & 1;
        if ((timing_arc_id_map[arc_id * 2 + el] == -1) || isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf])) return;
        float si = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
        float lc = pinLoad[to_pin_id * NUM_ATTR + tel_rf];
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        float so = d_allocator->query(timing_id, irf, orf, si, lc, 1);
        if (isnan(so)) return;
        if (isnan(pinSlew[to_pin_id * NUM_ATTR + tel_rf]) || ((pinSlew[to_pin_id * NUM_ATTR + tel_rf] > so) ^ el)) {
            atomicExch(&pinSlew[to_pin_id * NUM_ATTR + tel_rf], so);
        }
    }
}

__device__ void propagateDelay(index_type arc_id,
                               index_type from_pin_id,
                               index_type to_pin_id,
                               float *pinSlew,
                               float *pinLoad,
                               float *pinImpulse,
                               float *pinRootDelay,
                               float *arcDelay,
                               int arc_type,
                               int *timing_arc_id_map,
                               GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    if ((arc_type == 0) && (i < NUM_ATTR)) {
        float delay = pinRootDelay[to_pin_id * NUM_ATTR + i];
        int el_rf_rf = (i << 1) + (i & 1);
        arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf] = delay;
    } else if (arc_type == 1) {
        int el = i >> 2;
        int fel_rf = i >> 1;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        int irf = fel_rf & 1;
        int orf = tel_rf & 1;
        if ((timing_arc_id_map[arc_id * 2 + el] == -1) || isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf])) return;
        float si = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
        float lc = pinLoad[to_pin_id * NUM_ATTR + tel_rf];
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        float delay = d_allocator->query(timing_id, irf, orf, si, lc, 0);
        if (isnan(delay)) return;
        arcDelay[arc_id * 2 * NUM_ATTR + i] = delay;
    }
}

__device__ void propagateAT(index_type arc_id,
                            index_type from_pin_id,
                            index_type to_pin_id,
                            float *pinAt,
                            float *arcDelay,
                            index_type *at_prefix_pin,
                            index_type *at_prefix_arc,
                            index_type *at_prefix_attr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    int el = i >> 2;
    int fel_rf = i >> 1;
    int tel_rf = ((i & 0b100) >> 1) + (i & 1);
    int irf = fel_rf & 1;
    int orf = tel_rf & 1;
    if (isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + i])) return;
    float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
    float at = pinAt[from_pin_id * NUM_ATTR + fel_rf] + delay;
    // FIXME: conflict
    if (isnan(pinAt[to_pin_id * NUM_ATTR + tel_rf]) || ((pinAt[to_pin_id * NUM_ATTR + tel_rf] > at) ^ el)) {
        atomicExch(&pinAt[to_pin_id * NUM_ATTR + tel_rf], at);
        at_prefix_pin[to_pin_id * NUM_ATTR + tel_rf] = from_pin_id;
        at_prefix_arc[to_pin_id * NUM_ATTR + tel_rf] = arc_id;
        at_prefix_attr[to_pin_id * NUM_ATTR + tel_rf] = fel_rf;
    }
}

__device__ void propagateTest(index_type arc_id,
                              index_type test_id,
                              index_type from_pin_id,
                              index_type to_pin_id,
                              int *timing_arc_id_map,
                              float *pinSlew,
                              float *pinAt,
                              float *pinRat,
                              float *testRelatedAT,
                              float *testRAT,
                              float *testConstraint,
                              float clock_period,
                              GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    if (i < NUM_ATTR) {
        const int el = i >> 1;
        const int rf = i & 1;
        const int el_rf_rf = (i << 1) + (i & 1);
        if ((timing_arc_id_map[arc_id * 2 + el] == -1) || (isnan(pinSlew[to_pin_id * NUM_ATTR + i]))) return;
        int fel = el ^ 1;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        int frf = d_allocator->d_is_rising_edge_triggered[timing_id] ? 0 : 1;
        if (frf && !d_allocator->d_is_falling_edge_triggered[timing_id]) {
            return;
        }
        const int fel_rf = (fel << 1) + frf;
        if (isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf]) || isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf])) return;

        if (el == 0) {
            testRelatedAT[test_id * NUM_ATTR + i] = pinAt[from_pin_id * NUM_ATTR + fel_rf];
        } else {
            testRelatedAT[test_id * NUM_ATTR + i] = pinAt[from_pin_id * NUM_ATTR + fel_rf] + clock_period;
        }

        float sr = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
        float sc = pinSlew[to_pin_id * NUM_ATTR + i];
        testConstraint[test_id * NUM_ATTR + i] = d_allocator->query(timing_id, frf, rf, sr, sc, 2);

        if (!isnan(testConstraint[test_id * NUM_ATTR + i]) && !isnan(testRelatedAT[test_id * NUM_ATTR + i])) {
            if (el == 0) {
                pinRat[to_pin_id * NUM_ATTR + i] = testRelatedAT[test_id * NUM_ATTR + i] + testConstraint[test_id * NUM_ATTR + i];
            } else {
                pinRat[to_pin_id * NUM_ATTR + i] = testRelatedAT[test_id * NUM_ATTR + i] - testConstraint[test_id * NUM_ATTR + i];
            }
            testRAT[test_id * NUM_ATTR + i] = pinRat[to_pin_id * NUM_ATTR + i];
        }
    }
}

__global__ void propagatePin(index_type *level_list,
                             index_type *pin_backward_arc_list_end,
                             index_type *pin_backward_arc_list,
                             index_type *timing_arc_from_pin_id,
                             int *arc_types,
                             int *arc_id2test_id,
                             float *pinSlew,
                             float *pinLoad,
                             float *pinImpulse,
                             float *pinRootDelay,
                             float *pinAt,
                             float *pinRat,
                             float *testRelatedAT,
                             float *testRAT,
                             float *testConstraint,
                             float *arcDelay,
                             int *timing_arc_id_map,
                             index_type *at_prefix_pin,
                             index_type *at_prefix_arc,
                             index_type *at_prefix_attr,
                             index_type level_start_offset,
                             int num_pins_level,
                             float clock_period,
                             GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_idx = idx >> 3;
    if (pin_idx < num_pins_level) {
        index_type to_pin_id = level_list[level_start_offset + pin_idx];
        for (index_type i = pin_backward_arc_list_end[to_pin_id]; i < pin_backward_arc_list_end[to_pin_id + 1]; i++) {
            index_type arc_id = pin_backward_arc_list[i];
            index_type from_pin_id = timing_arc_from_pin_id[arc_id];
            int arc_type = arc_types[arc_id];
            propagateSlew(arc_id, from_pin_id, to_pin_id, pinSlew, pinLoad, pinImpulse, pinRootDelay, arcDelay, arc_type, timing_arc_id_map, d_allocator);
            propagateDelay(arc_id, from_pin_id, to_pin_id, pinSlew, pinLoad, pinImpulse, pinRootDelay, arcDelay, arc_type, timing_arc_id_map, d_allocator);
            propagateAT(arc_id, from_pin_id, to_pin_id, pinAt, arcDelay, at_prefix_pin, at_prefix_arc, at_prefix_attr);
            int test_id = arc_id2test_id[arc_id];
            if (clock_period > 0 && test_id != -1) {
                propagateTest(arc_id, test_id, from_pin_id, to_pin_id, timing_arc_id_map, pinSlew, pinAt, pinRat, testRelatedAT, testRAT, testConstraint, clock_period, d_allocator);
            }
        }
    }
}

__device__ void propagateRAT(index_type arc_id,
                             int arc_type,
                             index_type from_pin_id,
                             index_type to_pin_id,
                             float *pinAt,
                             float *pinRat,
                             float *arcDelay,
                             int *timing_arc_id_map,
                             float *from_rats,
                             GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    if ((arc_type == 0) && (i < NUM_ATTR)) {
        const int el_rf_rf = (i << 1) + (i & 1);
        const int el = i >> 1;
        if (isnan(pinRat[to_pin_id * NUM_ATTR + i]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf])) return;
        float delay = arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf];
        float rat = pinRat[to_pin_id * NUM_ATTR + i] - delay;
        if (isnan(pinRat[from_pin_id * NUM_ATTR + i]) || ((pinRat[from_pin_id * NUM_ATTR + i] < rat) ^ el)) {
            atomicExch(&pinRat[from_pin_id * NUM_ATTR + i], rat);
        }
    } else if (arc_type == 1) {
        int el = i >> 2;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        if (timing_arc_id_map[arc_id * 2 + el] == -1) return;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        if (d_allocator->d_is_constraint[timing_id]) return;
        if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + i])) return;
        float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
        float rat = pinRat[to_pin_id * NUM_ATTR + tel_rf] - delay;
        from_rats[threadIdx.x] = rat;
    }
}

__global__ void propagatePinBack(index_type *level_list,
                                 index_type *pin_forward_arc_list_end,
                                 index_type *pin_forward_arc_list,
                                 index_type *timing_arc_to_pin_id,
                                 int *arc_types,
                                 int *arc_id2test_id,
                                 float *pinSlew,
                                 float *pinLoad,
                                 float *pinImpulse,
                                 float *pinRootDelay,
                                 float *pinAt,
                                 float *pinRat,
                                 float *testRelatedAT,
                                 float *testConstraint,
                                 float *arcDelay,
                                 int *timing_arc_id_map,
                                 index_type level_start_offset,
                                 int num_pins_level,
                                 float clock_period,
                                 GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_idx = idx >> 3;
    extern __shared__ float from_rats[];

    if (pin_idx < num_pins_level) {
        index_type from_pin_id = level_list[level_start_offset + pin_idx];
        for (index_type i = pin_forward_arc_list_end[from_pin_id]; i < pin_forward_arc_list_end[from_pin_id + 1]; i++) {
            index_type arc_id = pin_forward_arc_list[i];
            index_type to_pin_id = timing_arc_to_pin_id[arc_id];
            int arc_type = arc_types[arc_id];
            if ((threadIdx.x % (2 * NUM_ATTR)) == 0) {
                for (int i = threadIdx.x; i < threadIdx.x + 2 * NUM_ATTR; i++) from_rats[i] = nanf("");
            }
            __syncthreads();

            propagateRAT(arc_id, arc_type, from_pin_id, to_pin_id, pinAt, pinRat, arcDelay, timing_arc_id_map, from_rats, d_allocator);

            __syncthreads();
            if ((threadIdx.x % (2 * NUM_ATTR)) == 0) {
                for (int ti = threadIdx.x; ti < threadIdx.x + 2 * NUM_ATTR; ti++) {
                    const int i = ti & 0b111;
                    if (isnan(from_rats[ti])) continue;
                    int el = i >> 2;
                    int fel_rf = i >> 1;
                    float rat = from_rats[ti];
                    if (isnan(pinRat[from_pin_id * NUM_ATTR + fel_rf]) || ((pinRat[from_pin_id * NUM_ATTR + fel_rf] < rat) ^ el)) {
                        atomicExch(&pinRat[from_pin_id * NUM_ATTR + fel_rf], rat);
                    }
                }
            }
        }
    }
}

void update_timing_cuda(index_type *level_list,
                        vector<int> level_list_end_cpu,
                        index_type *pin_forward_arc_list_end,
                        index_type *pin_forward_arc_list,
                        index_type *timing_arc_to_pin_id,
                        index_type *pin_backward_arc_list_end,
                        index_type *pin_backward_arc_list,
                        index_type *timing_arc_from_pin_id,
                        int *arc_types,
                        int *arc_id2test_id,
                        float *pinSlew,
                        float *pinLoad,
                        float *pinImpulse,
                        float *pinRootDelay,
                        float *pinAt,
                        float *pinRat,
                        float *testRelatedAT,
                        float *testRAT,
                        float *testConstraint,
                        float *arcDelay,
                        int *timing_arc_id_map,
                        index_type *at_prefix_pin,
                        index_type *at_prefix_arc,
                        index_type *at_prefix_attr,
                        float clock_period,
                        GPULutAllocator *d_allocator,
                        int num_pins,
                        bool deterministic) {
    for (int i = 1; i < level_list_end_cpu.size() - 1; i++) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        index_type level_start_offset = level_list_end_cpu[i];
        // printf("==== level %d ======= %d \n", i, num_pins_level);
        propagatePin<<<BLOCK_NUMBER(num_pins_level * 2 * NUM_ATTR), BLOCK_SIZE>>>(level_list,
                                                                                  pin_backward_arc_list_end,
                                                                                  pin_backward_arc_list,
                                                                                  timing_arc_from_pin_id,
                                                                                  arc_types,
                                                                                  arc_id2test_id,
                                                                                  pinSlew,
                                                                                  pinLoad,
                                                                                  pinImpulse,
                                                                                  pinRootDelay,
                                                                                  pinAt,
                                                                                  pinRat,
                                                                                  testRelatedAT,
                                                                                  testRAT,
                                                                                  testConstraint,
                                                                                  arcDelay,
                                                                                  timing_arc_id_map,
                                                                                  at_prefix_pin,
                                                                                  at_prefix_arc,
                                                                                  at_prefix_attr,
                                                                                  level_start_offset,
                                                                                  num_pins_level,
                                                                                  clock_period,
                                                                                  d_allocator);

        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    for (int i = level_list_end_cpu.size() - 3; i >= 0; i--) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        index_type level_start_offset = level_list_end_cpu[i];
        // printf("==== level %d ======= %d \n", i, num_pins_level);
        propagatePinBack<<<BLOCK_NUMBER(num_pins_level * 2 * NUM_ATTR), BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(level_list,
                                                                                                                  pin_forward_arc_list_end,
                                                                                                                  pin_forward_arc_list,
                                                                                                                  timing_arc_to_pin_id,
                                                                                                                  arc_types,
                                                                                                                  arc_id2test_id,
                                                                                                                  pinSlew,
                                                                                                                  pinLoad,
                                                                                                                  pinImpulse,
                                                                                                                  pinRootDelay,
                                                                                                                  pinAt,
                                                                                                                  pinRat,
                                                                                                                  testRelatedAT,
                                                                                                                  testConstraint,
                                                                                                                  arcDelay,
                                                                                                                  timing_arc_id_map,
                                                                                                                  level_start_offset,
                                                                                                                  num_pins_level,
                                                                                                                  clock_period,
                                                                                                                  d_allocator);

        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}

namespace {

__device__ bool g_power_allow_clock_activity_override = true;

__device__ __forceinline__ float power_percent_change(float value, float prev) {
    if (prev == 0.0f) return value == 0.0f ? 0.0f : 1.0f;
    return fabsf(value - prev) / fabsf(prev);
}

__device__ __forceinline__ float power_max_activity_density_from_slew(int pin,
                                                                      const float* pinSlew,
                                                                      float time_unit) {
    if (!pinSlew || pin < 0 || !(time_unit > 0.0f)) return 3.4028234663852886e38f;
    float min_rf_slew = 3.4028234663852886e38f;
    #pragma unroll
    for (int base = 0; base < NUM_ATTR; base += 2) {
        const float rise = pinSlew[pin * NUM_ATTR + base];
        const float fall = pinSlew[pin * NUM_ATTR + base + 1];
        if (isfinite(rise) && isfinite(fall)) {
            const float avg = 0.5f * (rise + fall) * time_unit;
            if (avg > 0.0f && avg < min_rf_slew) min_rf_slew = avg;
        }
    }
    return (min_rf_slew < 3.4028234663852886e38f) ? (1.0f / min_rf_slew)
                                                  : 3.4028234663852886e38f;
}

__device__ bool power_set_activity(int pin,
                                   float new_density,
                                   float new_duty,
                                   int new_origin,
                                   bool force,
                                   const float* pinSlew,
                                   float time_unit,
                                   float* density,
                                   float* duty,
                                   int* origin) {
    if (!force && origin[pin] == 2 && !g_power_allow_clock_activity_override) return false;
    const float prev_density = density[pin];
    const float prev_duty = duty[pin];
    const int prev_origin = origin[pin];
    const float max_density = power_max_activity_density_from_slew(pin, pinSlew, time_unit);
    float d = fminf(fmaxf(new_density, 0.0f), max_density);
    // Match OpenSTA PwrActivity::check(): clip tiny numerical-noise densities.
    if (fabsf(d) < 1.0e-10f) d = 0.0f;
    const float u = fminf(fmaxf(new_duty, 0.0f), 1.0f);
    const bool changed = power_percent_change(d, prev_density) > 0.01f
        || power_percent_change(u, prev_duty) > 0.01f
        || prev_origin != new_origin;
    density[pin] = d;
    duty[pin] = u;
    origin[pin] = new_origin;
    return changed;
}

__device__ void power_enqueue_adjacent(int pin,
                                       const index_type* pin_forward_arc_list_end,
                                       const index_type* pin_forward_arc_list,
                                       const index_type* timing_arc_to_pin_id,
                                       const int* arc_types,
                                       const int* arc_id2test_id,
                                       const uint8_t* is_seq_output_pin,
                                       const int* pin_power_level,
                                       uint8_t* active_level,
                                       int num_power_levels,
                                       int* active) {
    for (index_type i = pin_forward_arc_list_end[pin]; i < pin_forward_arc_list_end[pin + 1]; i++) {
        const int arc = pin_forward_arc_list[i];
        if (arc_id2test_id && arc_id2test_id[arc] != -1) continue;
        const int to_pin = timing_arc_to_pin_id[arc];
        if (to_pin < 0) continue;
        if (arc_types && arc_types[arc] == 1 && is_seq_output_pin && is_seq_output_pin[to_pin]) continue;
        atomicExch(&active[to_pin], 1);
        if (pin_power_level && active_level) {
            const int level = pin_power_level[to_pin];
            if (level >= 0 && level < num_power_levels) active_level[level] = 1;
        }
    }
}

__device__ bool power_set_clock_gate_output(int pin,
                                            const int* clock_gate_clock_for_out,
                                            const int* clock_gate_enable_for_out,
                                            const float* pinSlew,
                                            float time_unit,
                                            float* density,
                                            float* duty,
                                            int* origin) {
    if (!clock_gate_clock_for_out || !clock_gate_enable_for_out) return false;
    const int clk = clock_gate_clock_for_out[pin];
    const int en = clock_gate_enable_for_out[pin];
    if (clk < 0 || en < 0) return false;
    const float out_density = density[clk] * duty[en] + density[en] * duty[clk];
    const float out_duty = duty[clk] * duty[en];
    return power_set_activity(pin, out_density, out_duty, 3, false,
                              pinSlew, time_unit, density, duty, origin);
}

__device__ void power_enqueue_clock_gate_output(int pin,
                                                const int* clock_gate_out_for_input,
                                                const int* pin_power_level,
                                                uint8_t* active_level,
                                                int num_power_levels,
                                                int* active) {
    if (!clock_gate_out_for_input) return;
    const int out_pin = clock_gate_out_for_input[pin];
    if (out_pin < 0) return;
    atomicExch(&active[out_pin], 1);
    if (pin_power_level && active_level) {
        const int level = pin_power_level[out_pin];
        if (level >= 0 && level < num_power_levels) active_level[level] = 1;
    }
}


__device__ bool power_eval_expr_bool(int expr_id,
                                     uint64_t bits,
                                     int force_var,
                                     int force_val,
                                     const int* var_pins,
                                     int var_count,
                                     const GpuPowerExprOpHost* ops,
                                     const int* expr_start,
                                     const int* expr_count,
                                     int8_t& value) {
    if (expr_id < 0) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    int8_t stack[128];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        switch (op.op) {
            case 0: {
                int var = -1;
                for (int i = 0; i < var_count; i++) {
                    if (var_pins[i] == op.arg) { var = i; break; }
                }
                if (var < 0 || sp >= 128) return false;
                int bit = (bits >> var) & 1ULL;
                if (var == force_var) bit = force_val;
                stack[sp++] = static_cast<int8_t>(bit);
                break;
            }
            case 1:
                if (sp >= 128) return false;
                stack[sp++] = 0;
                break;
            case 2:
                if (sp >= 128) return false;
                stack[sp++] = 1;
                break;
            case 3: {
                if (sp < 1) return false;
                const int8_t a = stack[--sp];
                stack[sp++] = a < 0 ? -1 : static_cast<int8_t>(!a);
                break;
            }
            case 4: {
                if (sp < 2) return false;
                const int8_t b = stack[--sp];
                const int8_t a = stack[--sp];
                if (a == 0 || b == 0) stack[sp++] = 0;
                else if (a == 1 && b == 1) stack[sp++] = 1;
                else stack[sp++] = -1;
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const int8_t b = stack[--sp];
                const int8_t a = stack[--sp];
                if (a == 1 || b == 1) stack[sp++] = 1;
                else if (a == 0 && b == 0) stack[sp++] = 0;
                else stack[sp++] = -1;
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const int8_t b = stack[--sp];
                const int8_t a = stack[--sp];
                if (a < 0 || b < 0) stack[sp++] = -1;
                else stack[sp++] = static_cast<int8_t>((a != 0) ^ (b != 0));
                break;
            }
            default:
                return false;
        }
    }
    if (sp != 1 || stack[0] < 0) return false;
    value = stack[0];
    return true;
}

__device__ bool power_eval_expr_activity(int expr_id,
                                         const GpuPowerExprOpHost* ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         const float* pin_density,
                                         const float* pin_duty,
                                         float& out_density,
                                         float& out_duty) {
    if (expr_id < 0) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    int var_pins[16];
    float duties[16];
    float densities[16];
    int var_count = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        if (op.op != 0 || op.arg < 0) continue;
        bool found = false;
        for (int i = 0; i < var_count; i++) {
            if (var_pins[i] == op.arg) { found = true; break; }
        }
        if (!found) {
            if (var_count >= 16) return false;
            var_pins[var_count++] = op.arg;
        }
    }
    for (int i = 0; i < var_count; i++) {
        duties[i] = fminf(fmaxf(pin_duty[var_pins[i]], 0.0f), 1.0f);
        densities[i] = pin_density[var_pins[i]];
    }
    const uint64_t states = 1ULL << var_count;
    float true_duty = 0.0f;
    float false_duty = 0.0f;
    for (uint64_t bits = 0; bits < states; bits++) {
        float prob = 1.0f;
        for (int i = 0; i < var_count; i++)
            prob *= ((bits >> i) & 1ULL) ? duties[i] : (1.0f - duties[i]);
        int8_t value = -1;
        if (!power_eval_expr_bool(expr_id, bits, -1, 0, var_pins, var_count, ops, expr_start, expr_count, value)) return false;
        if (value) true_duty += prob;
        else false_duty += prob;
    }
    // Match OpenSTA's numerically stabler BDD recursion: avoid rounding near-1
    // duties to exactly 1.0 by using the smaller complement side when possible.
    out_duty = (false_duty < true_duty) ? (1.0f - false_duty) : true_duty;
    out_duty = fminf(fmaxf(out_duty, 0.0f), 1.0f);

    out_density = 0.0f;
    for (int var = 0; var < var_count; var++) {
        float diff_true = 0.0f;
        float diff_false = 0.0f;
        for (uint64_t bits = 0; bits < states; bits++) {
            if ((bits >> var) & 1ULL) continue;
            float prob = 1.0f;
            for (int i = 0; i < var_count; i++) {
                if (i == var) continue;
                prob *= ((bits >> i) & 1ULL) ? duties[i] : (1.0f - duties[i]);
            }
            int8_t v0 = -1, v1 = -1;
            if (!power_eval_expr_bool(expr_id, bits, var, 0, var_pins, var_count, ops, expr_start, expr_count, v0)) return false;
            if (!power_eval_expr_bool(expr_id, bits, var, 1, var_pins, var_count, ops, expr_start, expr_count, v1)) return false;
            if (v0 != v1) diff_true += prob;
            else diff_false += prob;
        }
        float diff_duty = (diff_false < diff_true) ? (1.0f - diff_false) : diff_true;
        diff_duty = fminf(fmaxf(diff_duty, 0.0f), 1.0f);
        out_density += densities[var] * diff_duty;
    }
    return isfinite(out_density) && isfinite(out_duty);
}

__global__ void power_seed_pi_kernel(const int* primary_inputs,
                                     int num_primary_inputs,
                                     float default_density,
                                     float clock_density,
                                     const float* pinSlew,
                                     float time_unit,
                                     float* density,
                                     float* duty,
                                     int* origin,
                                     const index_type* pin_forward_arc_list_end,
                                     const index_type* pin_forward_arc_list,
                                     const index_type* timing_arc_to_pin_id,
                                     const int* arc_types,
                                     const int* arc_id2test_id,
                                     const uint8_t* is_seq_output_pin,
                                     const int* pin_power_level,
                                     uint8_t* active_level,
                                     int num_power_levels,
                                     int* active) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primary_inputs) return;
    const int pin = primary_inputs[idx];
    if (pin < 0) return;
    if (power_set_activity(pin, default_density, 0.5f, 1, false, pinSlew, time_unit, density, duty, origin)) {
        power_enqueue_adjacent(pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                               arc_types, arc_id2test_id, is_seq_output_pin,
                               pin_power_level, active_level, num_power_levels, active);
    }
}

__global__ void power_seed_clock_active_kernel(const int* clock_pins,
                                               int num_clock_pins,
                                               float clock_density,
                                               const float* pinSlew,
                                               float time_unit,
                                               float* density,
                                               float* duty,
                                               int* origin,
                                               const index_type* pin_forward_arc_list_end,
                                               const index_type* pin_forward_arc_list,
                                               const index_type* timing_arc_to_pin_id,
                                               const int* arc_types,
                                               const int* arc_id2test_id,
                                               const uint8_t* is_seq_output_pin,
                                               const int* pin_power_level,
                                               uint8_t* active_level,
                                               int num_power_levels,
                                               int* active) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_clock_pins) return;
    const int pin = clock_pins[idx];
    if (pin < 0) return;
    if (power_set_activity(pin, clock_density, 0.5f, 2, true, pinSlew, time_unit, density, duty, origin)) {
        power_enqueue_adjacent(pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                               arc_types, arc_id2test_id, is_seq_output_pin,
                               pin_power_level, active_level, num_power_levels, active);
    }
}


__global__ void power_seed_case_kernel(int n,
                                       const int* case_values,
                                       const float* pinSlew,
                                       float time_unit,
                                       float* density,
                                       float* duty,
                                       int* origin,
                                       const index_type* pin_forward_arc_list_end,
                                       const index_type* pin_forward_arc_list,
                                       const index_type* timing_arc_to_pin_id,
                                       const int* arc_types,
                                       const int* arc_id2test_id,
                                       const uint8_t* is_seq_output_pin,
                                       const int* pin_power_level,
                                       uint8_t* active_level,
                                       int num_power_levels,
                                       int* active) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin >= n || !case_values || case_values[pin] < 0) return;
    if (power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true,
                           pinSlew, time_unit, density, duty, origin)) {
        power_enqueue_adjacent(pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                               arc_types, arc_id2test_id, is_seq_output_pin,
                               pin_power_level, active_level, num_power_levels, active);
    }
}

__global__ void power_seed_seq_feedback_state_kernel(const int* seed_pins,
                                                     int num_seed_pins,
                                                     const int* seed_seqs,
                                                     int num_seed_seqs,
                                                     float default_density,
                                                     const float* pinSlew,
                                                     float time_unit,
                                                     float* density,
                                                     float* duty,
                                                     int* origin,
                                                     int* pending_seq,
                                                     int* pending_seq_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_seed_pins) {
        const int pin = seed_pins[idx];
        if (pin >= 0) {
            power_set_activity(pin, default_density, 0.5f, 1, false,
                               pinSlew, time_unit, density, duty, origin);
        }
    }
    if (idx < num_seed_seqs) {
        const int seq_id = seed_seqs[idx];
        if (seq_id >= 0 && atomicExch(&pending_seq[seq_id], 1) == 0) {
            atomicAdd(pending_seq_count, 1);
        }
    }
}

__global__ void power_visit_level_kernel(const index_type* level_list,
                                         int level_start,
                                         int num_level_pins,
                                         const int* case_values,
                                         const uint8_t* is_load_pin,
                                         const uint8_t* is_driver_pin,
                                         const int* pin2net_map,
                                         const int* net_driver_pin,
                                         const int* flat_net2pin_start_map,
                                         const int* flat_net2pin_map,
                                         const int* pin_func_expr_id,
                                         const int* clock_gate_out_for_input,
                                         const int* clock_gate_clock_for_out,
                                         const int* clock_gate_enable_for_out,
                                         const GpuPowerExprOpHost* expr_ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         float clock_density,
                                         const float* pinSlew,
                                         float time_unit,
                                         float* density,
                                         float* duty,
                                         int* origin,
                                         int* active,
                                         const index_type* pin_forward_arc_list_end,
                                         const index_type* pin_forward_arc_list,
                                         const index_type* timing_arc_to_pin_id,
                                         const int* arc_types,
                                         const int* arc_id2test_id,
                                         const uint8_t* is_seq_output_pin,
                                         const int* pin_seq_list_start,
                                         const int* pin_seq_list,
                                         const int* pin_power_level,
                                         uint8_t* active_level,
                                         int num_power_levels,
                                         int current_level,
                                         int* pending_seq,
                                         int* pending_seq_count) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = level_list[level_start + pos];
    if (pin < 0 || atomicExch(&active[pin], 0) == 0) return;

    bool changed = false;
    if (case_values && case_values[pin] >= 0) {
        changed = power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true,
                                     pinSlew, time_unit, density, duty, origin);
    } else if (is_load_pin[pin]) {
        const int net = pin2net_map[pin];
        const int driver = (net >= 0 && net_driver_pin) ? net_driver_pin[net] : -1;
        if (driver >= 0 && driver != pin) {
            changed = power_set_activity(pin, density[driver], duty[driver], 3, false,
                                         pinSlew, time_unit, density, duty, origin);
        }
    }
    if ((!case_values || case_values[pin] < 0) && is_driver_pin[pin]) {
        const int expr_id = pin_func_expr_id[pin];
        if (expr_id >= 0) {
            float out_density = 0.0f, out_duty = 0.0f;
            if (power_eval_expr_activity(expr_id, expr_ops, expr_start, expr_count,
                                         density, duty, out_density, out_duty)) {
                changed = power_set_activity(pin, out_density, out_duty, 3, false,
                                             pinSlew, time_unit, density, duty, origin) || changed;
            }
        }
        changed = power_set_clock_gate_output(pin, clock_gate_clock_for_out, clock_gate_enable_for_out,
                                              pinSlew, time_unit, density, duty, origin) || changed;
    }
    if (!changed) return;
    if (is_load_pin[pin]) {
        for (int i = pin_seq_list_start[pin]; i < pin_seq_list_start[pin + 1]; i++) {
            const int seq_id = pin_seq_list[i];
            if (seq_id >= 0 && atomicExch(&pending_seq[seq_id], 1) == 0)
                atomicAdd(pending_seq_count, 1);
        }
        power_enqueue_clock_gate_output(pin, clock_gate_out_for_input,
                                        pin_power_level, active_level, num_power_levels, active);
    }
    power_enqueue_adjacent(pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                           arc_types, arc_id2test_id, is_seq_output_pin,
                           pin_power_level, active_level, num_power_levels, active);
}

__global__ void power_seed_seq_kernel(const GpuPowerSeqHost* seqs,
                                      int num_seqs,
                                      const GpuPowerExprOpHost* expr_ops,
                                      const int* expr_start,
                                      const int* expr_count,
                                      float clock_density,
                                      const float* pinSlew,
                                      float time_unit,
                                      float* density,
                                      float* duty,
                                      int* origin,
                                      int* pending_seq,
                                      int* pending_seq_count,
                                      const index_type* pin_forward_arc_list_end,
                                      const index_type* pin_forward_arc_list,
                                      const index_type* timing_arc_to_pin_id,
                                      const int* arc_types,
                                      const int* arc_id2test_id,
                                      const uint8_t* is_seq_output_pin,
                                      const int* pin_power_level,
                                      uint8_t* active_level,
                                      int num_power_levels,
                                      int* active) {
    const int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_id >= num_seqs) return;
    if (atomicExch(&pending_seq[seq_id], 0) == 0) return;
    atomicSub(pending_seq_count, 1);
    const auto seq = seqs[seq_id];
    float in_density = 0.0f, in_duty = 0.0f;
    float clk_density = clock_density, clk_duty = 0.5f;
    if (!power_eval_expr_activity(seq.data_expr_id, expr_ops, expr_start, expr_count,
                                  density, duty, in_density, in_duty)) return;
    power_eval_expr_activity(seq.clk_expr_id, expr_ops, expr_start, expr_count,
                             density, duty, clk_density, clk_duty);
    float out_density = in_density;
    float out_duty = in_duty;
    if (in_density > clk_density / 2.0f) {
        out_density = seq.is_latch ? in_density * clk_duty
                                   : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
    }
    if (seq.q_pin >= 0) {
        if (power_set_activity(seq.q_pin, out_density, out_duty, 3, false, pinSlew, time_unit, density, duty, origin)) {
            power_enqueue_adjacent(seq.q_pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                   arc_types, arc_id2test_id, is_seq_output_pin,
                                   pin_power_level, active_level, num_power_levels, active);
        }
    }
    if (seq.qn_pin >= 0) {
        const float qn_duty = 1.0f - out_duty;
        if (power_set_activity(seq.qn_pin, out_density, qn_duty, 3, false, pinSlew, time_unit, density, duty, origin)) {
            power_enqueue_adjacent(seq.qn_pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                   arc_types, arc_id2test_id, is_seq_output_pin,
                                   pin_power_level, active_level, num_power_levels, active);
        }
    }
}


__device__ bool power_process_pin_frontier(int pin,
                                           const int* case_values,
                                           const uint8_t* is_load_pin,
                                           const uint8_t* is_driver_pin,
                                           const int* pin2net_map,
                                           const int* net_driver_pin,
                                           const int* flat_net2pin_start_map,
                                           const int* flat_net2pin_map,
                                           const int* pin_func_expr_id,
                                           const int* clock_gate_clock_for_out,
                                           const int* clock_gate_enable_for_out,
                                           const GpuPowerExprOpHost* expr_ops,
                                           const int* expr_start,
                                           const int* expr_count,
                                           float clock_density,
                                           const float* pinSlew,
                                           float time_unit,
                                           float* density,
                                           float* duty,
                                           int* origin,
                                           const int* pin_seq_list_start,
                                           const int* pin_seq_list,
                                           int* pending_seq,
                                           int* pending_seq_count) {
    if (case_values && case_values[pin] >= 0) {
        return power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true,
                                  pinSlew, time_unit, density, duty, origin);
    }
    bool changed = false;
    if (is_load_pin[pin]) {
        const int net = pin2net_map[pin];
        const int driver = (net >= 0 && net_driver_pin) ? net_driver_pin[net] : -1;
        if (driver >= 0 && driver != pin) {
            changed = power_set_activity(pin, density[driver], duty[driver], 3, false,
                                         pinSlew, time_unit, density, duty, origin);
        }
    }
    if (is_driver_pin[pin]) {
        const int expr_id = pin_func_expr_id[pin];
        if (expr_id >= 0) {
            float out_density = 0.0f, out_duty = 0.0f;
            if (power_eval_expr_activity(expr_id, expr_ops, expr_start, expr_count,
                                         density, duty, out_density, out_duty)) {
                changed = power_set_activity(pin, out_density, out_duty, 3, false,
                                             pinSlew, time_unit, density, duty, origin) || changed;
            }
        }
        changed = power_set_clock_gate_output(pin, clock_gate_clock_for_out, clock_gate_enable_for_out,
                                              pinSlew, time_unit, density, duty, origin) || changed;
    }
    if (changed && is_load_pin[pin]) {
        for (int i = pin_seq_list_start[pin]; i < pin_seq_list_start[pin + 1]; i++) {
            const int seq_id = pin_seq_list[i];
            if (seq_id >= 0 && atomicExch(&pending_seq[seq_id], 1) == 0)
                atomicAdd(pending_seq_count, 1);
        }
    }
    return changed;
}


__device__ void power_enqueue_pin_level_queue(int pin,
                                              const int* pin_power_level,
                                              const int* level_offsets,
                                              int num_power_levels,
                                              int* level_queue,
                                              int* level_counts,
                                              int* queued,
                                              int* overflow) {
    if (pin < 0 || !pin_power_level || !level_offsets) return;
    const int level = pin_power_level[pin];
    if (level < 0 || level >= num_power_levels) return;
    if (atomicExch(&queued[pin], 1) == 0) {
        const int pos = atomicAdd(&level_counts[level], 1);
        const int cap = level_offsets[level + 1] - level_offsets[level];
        if (pos < cap) level_queue[level_offsets[level] + pos] = pin;
        else if (overflow) atomicExch(overflow, 1);
    }
}

__device__ void power_enqueue_adjacent_level_queue(int pin,
                                                   const index_type* pin_forward_arc_list_end,
                                                   const index_type* pin_forward_arc_list,
                                                   const index_type* timing_arc_to_pin_id,
                                                   const int* arc_types,
                                                   const int* arc_id2test_id,
                                                   const uint8_t* is_seq_output_pin,
                                                   const int* pin_power_level,
                                                   const int* level_offsets,
                                                   int num_power_levels,
                                                   int* level_queue,
                                                   int* level_counts,
                                                   int* queued,
                                                   int* overflow) {
    for (index_type i = pin_forward_arc_list_end[pin]; i < pin_forward_arc_list_end[pin + 1]; i++) {
        const int arc = pin_forward_arc_list[i];
        if (arc_id2test_id && arc_id2test_id[arc] != -1) continue;
        const int to_pin = timing_arc_to_pin_id[arc];
        if (to_pin < 0) continue;
        if (arc_types && arc_types[arc] == 1 && is_seq_output_pin && is_seq_output_pin[to_pin]) continue;
        power_enqueue_pin_level_queue(to_pin, pin_power_level, level_offsets, num_power_levels,
                                      level_queue, level_counts, queued, overflow);
    }
}

__device__ void power_enqueue_clock_gate_output_level_queue(int pin,
                                                            const int* clock_gate_out_for_input,
                                                            const int* pin_power_level,
                                                            const int* level_offsets,
                                                            int num_power_levels,
                                                            int* level_queue,
                                                            int* level_counts,
                                                            int* queued,
                                                            int* overflow) {
    if (!clock_gate_out_for_input) return;
    const int out_pin = clock_gate_out_for_input[pin];
    if (out_pin < 0) return;
    power_enqueue_pin_level_queue(out_pin, pin_power_level, level_offsets, num_power_levels,
                                  level_queue, level_counts, queued, overflow);
}

__global__ void power_seed_case_level_queue_kernel(int n,
                                                   const int* case_values,
                                                   const float* pinSlew,
                                                   float time_unit,
                                                   float* density,
                                                   float* duty,
                                                   int* origin,
                                                   const index_type* pin_forward_arc_list_end,
                                                   const index_type* pin_forward_arc_list,
                                                   const index_type* timing_arc_to_pin_id,
                                                   const int* arc_types,
                                                   const int* arc_id2test_id,
                                                   const uint8_t* is_seq_output_pin,
                                                   const int* pin_power_level,
                                                   const int* level_offsets,
                                                   int num_power_levels,
                                                   int* level_queue,
                                                   int* level_counts,
                                                   int* queued,
                                                   int* overflow) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin >= n || !case_values || case_values[pin] < 0) return;
    if (power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true,
                           pinSlew, time_unit, density, duty, origin)) {
        power_enqueue_adjacent_level_queue(pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                           arc_types, arc_id2test_id, is_seq_output_pin,
                                           pin_power_level, level_offsets, num_power_levels,
                                           level_queue, level_counts, queued, overflow);
    }
}

__global__ void power_seed_pi_level_queue_kernel(const int* primary_inputs,
                                                 int num_primary_inputs,
                                                 float default_density,
                                                 float clock_density,
                                                 const float* pinSlew,
                                                 float time_unit,
                                                 float* density,
                                                 float* duty,
                                                 int* origin,
                                                 const index_type* pin_forward_arc_list_end,
                                                 const index_type* pin_forward_arc_list,
                                                 const index_type* timing_arc_to_pin_id,
                                                 const int* arc_types,
                                                 const int* arc_id2test_id,
                                                 const uint8_t* is_seq_output_pin,
                                                 const int* pin_power_level,
                                                 const int* level_offsets,
                                                 int num_power_levels,
                                                 int* level_queue,
                                                 int* level_counts,
                                                 int* queued,
                                                 int* overflow) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primary_inputs) return;
    const int pin = primary_inputs[idx];
    if (pin < 0) return;
    if (power_set_activity(pin, default_density, 0.5f, 1, false, pinSlew, time_unit, density, duty, origin)) {
        power_enqueue_adjacent_level_queue(pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                           arc_types, arc_id2test_id, is_seq_output_pin,
                                           pin_power_level, level_offsets, num_power_levels,
                                           level_queue, level_counts, queued, overflow);
    }
}

__global__ void power_seed_clock_level_queue_kernel(const int* clock_pins,
                                                    int num_clock_pins,
                                                    float clock_density,
                                                    const float* pinSlew,
                                                    float time_unit,
                                                    float* density,
                                                    float* duty,
                                                    int* origin,
                                                    const index_type* pin_forward_arc_list_end,
                                                    const index_type* pin_forward_arc_list,
                                                    const index_type* timing_arc_to_pin_id,
                                                    const int* arc_types,
                                                    const int* arc_id2test_id,
                                                    const uint8_t* is_seq_output_pin,
                                                    const int* pin_power_level,
                                                    const int* level_offsets,
                                                    int num_power_levels,
                                                    int* level_queue,
                                                    int* level_counts,
                                                    int* queued,
                                                    int* overflow) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_clock_pins) return;
    const int pin = clock_pins[idx];
    if (pin < 0) return;
    if (power_set_activity(pin, clock_density, 0.5f, 2, true, pinSlew, time_unit, density, duty, origin)) {
        power_enqueue_adjacent_level_queue(pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                           arc_types, arc_id2test_id, is_seq_output_pin,
                                           pin_power_level, level_offsets, num_power_levels,
                                           level_queue, level_counts, queued, overflow);
    }
}

__global__ void power_activity_level_queue_persistent_kernel(int n,
                                                             int num_power_levels,
                                                             const int* level_offsets,
                                                             const int* case_values,
                                                             const uint8_t* is_load_pin,
                                                             const uint8_t* is_driver_pin,
                                                             const int* pin2net_map,
                                                             const int* net_driver_pin,
                                                             const int* flat_net2pin_start_map,
                                                             const int* flat_net2pin_map,
                                                             const int* pin_func_expr_id,
                                                             const int* clock_gate_out_for_input,
                                                             const int* clock_gate_clock_for_out,
                                                             const int* clock_gate_enable_for_out,
                                                             const GpuPowerExprOpHost* expr_ops,
                                                             const int* expr_start,
                                                             const int* expr_count,
                                                             float clock_density,
                                                             const float* pinSlew,
                                                             float time_unit,
                                                             float* density,
                                                             float* duty,
                                                             int* origin,
                                                             const index_type* pin_forward_arc_list_end,
                                                             const index_type* pin_forward_arc_list,
                                                             const index_type* timing_arc_to_pin_id,
                                                             const int* arc_types,
                                                             const int* arc_id2test_id,
                                                             const uint8_t* is_seq_output_pin,
                                                             const int* pin_seq_list_start,
                                                             const int* pin_seq_list,
                                                             const GpuPowerSeqHost* seqs,
                                                             int num_seqs,
                                                             const int* pin_power_level,
                                                             int* pending_seq,
                                                             int* pending_seq_count,
                                                             int* level_queue,
                                                             int* level_counts,
                                                             int* queued,
                                                             int max_seq_passes,
                                                             int* overflow) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int pass = 0; pass < max_seq_passes; ++pass) {
        for (int level = 0; level < num_power_levels; ++level) {
            grid.sync();
            const int count = level_counts[level];
            const int offset = level_offsets[level];
            for (int idx = tid; idx < count; idx += stride) {
                const int pin = level_queue[offset + idx];
                if (pin < 0 || pin >= n) continue;
                atomicExch(&queued[pin], 0);
                const bool changed = power_process_pin_frontier(
                    pin, case_values, is_load_pin, is_driver_pin, pin2net_map, net_driver_pin, flat_net2pin_start_map, flat_net2pin_map,
                    pin_func_expr_id, clock_gate_clock_for_out, clock_gate_enable_for_out,
                    expr_ops, expr_start, expr_count, clock_density,
                    pinSlew, time_unit, density, duty, origin, pin_seq_list_start, pin_seq_list, pending_seq, pending_seq_count);
                if (changed) {
                    if (is_load_pin[pin]) {
                        power_enqueue_clock_gate_output_level_queue(pin, clock_gate_out_for_input,
                                                                    pin_power_level, level_offsets, num_power_levels,
                                                                    level_queue, level_counts, queued, overflow);
                    }
                    power_enqueue_adjacent_level_queue(pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                                       arc_types, arc_id2test_id, is_seq_output_pin,
                                                       pin_power_level, level_offsets, num_power_levels,
                                                       level_queue, level_counts, queued, overflow);
                }
            }
            grid.sync();
            if (tid == 0) level_counts[level] = 0;
        }
        grid.sync();
        const int pending = *pending_seq_count;
        if (pending <= 0) break;
        for (int seq_id = tid; seq_id < num_seqs; seq_id += stride) {
            if (atomicExch(&pending_seq[seq_id], 0) == 0) continue;
            atomicSub(pending_seq_count, 1);
            const auto seq = seqs[seq_id];
            float in_density = 0.0f, in_duty = 0.0f;
            float clk_density = clock_density, clk_duty = 0.5f;
            if (!power_eval_expr_activity(seq.data_expr_id, expr_ops, expr_start, expr_count,
                                          density, duty, in_density, in_duty)) continue;
            power_eval_expr_activity(seq.clk_expr_id, expr_ops, expr_start, expr_count,
                                     density, duty, clk_density, clk_duty);
            float out_density = in_density;
            float out_duty = in_duty;
            if (in_density > clk_density / 2.0f) {
                out_density = seq.is_latch ? in_density * clk_duty
                                           : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
            }
            if (seq.q_pin >= 0) {
                if (power_set_activity(seq.q_pin, out_density, out_duty, 3, false, pinSlew, time_unit, density, duty, origin)) {
                    power_enqueue_adjacent_level_queue(seq.q_pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                                       arc_types, arc_id2test_id, is_seq_output_pin,
                                                       pin_power_level, level_offsets, num_power_levels,
                                                       level_queue, level_counts, queued, overflow);
                }
            }
            if (seq.qn_pin >= 0) {
                const float qn_duty = 1.0f - out_duty;
                if (power_set_activity(seq.qn_pin, out_density, qn_duty, 3, false, pinSlew, time_unit, density, duty, origin)) {
                    power_enqueue_adjacent_level_queue(seq.qn_pin, pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                                       arc_types, arc_id2test_id, is_seq_output_pin,
                                                       pin_power_level, level_offsets, num_power_levels,
                                                       level_queue, level_counts, queued, overflow);
                }
            }
        }
    }
}

__global__ void power_pack_output_kernel(int n,
                                         const float* density,
                                         const float* duty,
                                         const int* origin,
                                         float* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx * 3 + 0] = density[idx];
    out[idx * 3 + 1] = duty[idx];
    out[idx * 3 + 2] = static_cast<float>(origin[idx]);
}


__global__ void power_switching_kernel(int n,
                                       int num_nodes,
                                       const uint8_t* is_driver_pin,
                                       const uint8_t* is_cell_pin,
                                       const int* pin2node_map,
                                       const float* pinLoad,
                                       const double* dmp_C1,
                                       const double* dmp_C2,
                                       float cap_unit,
                                       float voltage,
                                       const float* activity_out,
                                       float* inst_switching,
                                       float* pin_switching) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin >= n) return;
    float sw = 0.0f;
    if (is_driver_pin[pin] && (!is_cell_pin || is_cell_pin[pin])) {
        float load_internal = 0.0f;
        // OpenSTA Power::findSwitchingPower asks GraphDelayCalc::loadCap(..., MinMax::max())
        // and then takes max over rise/fall.  In the flattened attr layout, max/late
        // rise/fall are attrs 2 and 3 (early/min are attrs 0 and 1).
        for (int i = 2; i < NUM_ATTR; i++) {
            float v = pinLoad ? pinLoad[pin * NUM_ATTR + i] : 0.0f;
            if (dmp_C1 && dmp_C2) {
                const double dv = dmp_C1[pin * NUM_ATTR + i] + dmp_C2[pin * NUM_ATTR + i];
                v = isfinite(dv) && dv > 0.0 ? static_cast<float>(dv) : 0.0f;
            }
            if (isfinite(v)) load_internal = fmaxf(load_internal, v);
        }
        const float density = activity_out[pin * 3 + 0];
        if (load_internal > 0.0f && density > 0.0f && voltage > 0.0f) {
            const float load_f = load_internal * cap_unit;
            sw = 0.5f * load_f * voltage * voltage * density;
        }
        const int node = pin2node_map[pin];
        if (node >= 0 && node < num_nodes && sw != 0.0f) atomicAdd(&inst_switching[node], sw);
    }
    if (pin_switching) pin_switching[pin] = sw;
}

__device__ float power_expr_duty_cuda(int expr_id,
                                      const GpuPowerExprOpHost* ops,
                                      const int* expr_start,
                                      const int* expr_count,
                                      const float* pin_density,
                                      const float* pin_duty) {
    float d = 0.0f, u = 0.0f;
    if (!power_eval_expr_activity(expr_id, ops, expr_start, expr_count, pin_density, pin_duty, d, u)) return 0.0f;
    return fminf(fmaxf(u, 0.0f), 1.0f);
}

__device__ float power_expr_diff_duty_cuda(int expr_id,
                                           int diff_pin,
                                           const GpuPowerExprOpHost* ops,
                                           const int* expr_start,
                                           const int* expr_count,
                                           const float* pin_duty) {
    if (expr_id < 0 || diff_pin < 0) return 0.0f;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return 0.0f;
    int var_pins[16];
    float duties[16];
    int var_count = 0;
    int diff_var = -1;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        if (op.op != 0 || op.arg < 0) continue;
        bool found = false;
        for (int i = 0; i < var_count; i++) {
            if (var_pins[i] == op.arg) { found = true; break; }
        }
        if (!found) {
            if (var_count >= 16) return 0.0f;
            var_pins[var_count] = op.arg;
            if (op.arg == diff_pin) diff_var = var_count;
            var_count++;
        }
    }
    if (diff_var < 0) return 0.0f;
    for (int i = 0; i < var_count; i++) duties[i] = fminf(fmaxf(pin_duty[var_pins[i]], 0.0f), 1.0f);
    const uint64_t states = 1ULL << var_count;
    float diff_duty = 0.0f;
    for (uint64_t bits = 0; bits < states; bits++) {
        if ((bits >> diff_var) & 1ULL) continue;
        float prob = 1.0f;
        for (int i = 0; i < var_count; i++) {
            if (i == diff_var) continue;
            prob *= ((bits >> i) & 1ULL) ? duties[i] : (1.0f - duties[i]);
        }
        int8_t v0 = -1, v1 = -1;
        if (!power_eval_expr_bool(expr_id, bits, diff_var, 0, var_pins, var_count, ops, expr_start, expr_count, v0)) return 0.0f;
        if (!power_eval_expr_bool(expr_id, bits, diff_var, 1, var_pins, var_count, ops, expr_start, expr_count, v1)) return 0.0f;
        if (v0 != v1) diff_duty += prob;
    }
    return fminf(fmaxf(diff_duty, 0.0f), 1.0f);
}

__device__ float power_internal_row_duty(const GpuPowerInternalHost& row,
                                         const GpuPowerExprOpHost* expr_ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         const float* density,
                                         const float* duty) {
    if (row.duty_mode == 0) return 1.0f;
    if (row.duty_mode == 1) return power_expr_duty_cuda(row.duty_expr_id, expr_ops, expr_start, expr_count, density, duty);
    if (row.duty_mode == 2) return power_expr_diff_duty_cuda(row.duty_expr_id, row.duty_pin, expr_ops, expr_start, expr_count, duty);
    if (row.duty_mode == 3) return 0.5f;
    return 0.0f;
}

__global__ void power_internal_denom_kernel(const GpuPowerInternalHost* rows,
                                            int num_rows,
                                            const GpuPowerExprOpHost* expr_ops,
                                            const int* expr_start,
                                            const int* expr_count,
                                            const float* density,
                                            const float* duty,
                                            float* denom) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    const auto row = rows[idx];
    if (row.kind != 1 || row.denom_group < 0 || row.from_pin < 0) return;
    const float d = power_internal_row_duty(row, expr_ops, expr_start, expr_count, density, duty);
    const float numer = density[row.from_pin] * d;
    if (isfinite(numer) && numer != 0.0f) atomicAdd(&denom[row.denom_group], numer);
}

__global__ void power_internal_contrib_kernel(const GpuPowerInternalHost* rows,
                                              int num_rows,
                                              int num_nodes,
                                              const GpuPowerExprOpHost* expr_ops,
                                              const int* expr_start,
                                              const int* expr_count,
                                              const float* density,
                                              const float* duty,
                                              const float* pinSlew,
                                              const float* power_clock_slews,
                                              const double* dmp_C1,
                                              const double* dmp_C2,
                                              const float* denom,
                                              GPUPowerLutAllocator* power_allocator,
                                              float fallback_energy_unit,
                                              float* inst_internal,
                                              float* row_internal) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (row_internal) row_internal[idx] = 0.0f;
    const auto row = rows[idx];
    if (row.node_id < 0 || row.node_id >= num_nodes || row.to_pin < 0 || !power_allocator) return;
    const float row_duty = power_internal_row_duty(row, expr_ops, expr_start, expr_count, density, duty);
    float weight = (row.kind == 0) ? row_duty : 1.0f;
    if (row.kind == 1) {
        if (row.from_pin < 0 || row.denom_group < 0) return;
        const float numer = density[row.from_pin] * row_duty;
        const float den = denom[row.denom_group];
        if (!(den != 0.0f) || !isfinite(den) || !isfinite(numer)) return;
        weight = numer / den;
    }
    float load_internal = 0.0f;
    if (row.kind == 1 && dmp_C1 && dmp_C2) {
        for (int attr = 2; attr < NUM_ATTR; ++attr) {
            const double dv = dmp_C1[row.to_pin * NUM_ATTR + attr] + dmp_C2[row.to_pin * NUM_ATTR + attr];
            if (isfinite(dv) && dv > 0.0) load_internal = fmaxf(load_internal, static_cast<float>(dv));
        }
    }
    const int slew_pin = row.kind == 0 ? row.to_pin : row.from_pin;
    float slew_r = 0.0f, slew_f = 0.0f;
    if (slew_pin >= 0 && pinSlew) {
        auto slew_value = [&](int attr) {
            float value = pinSlew[slew_pin * NUM_ATTR + attr];
            if (power_clock_slews) {
                const float clock_slew = power_clock_slews[slew_pin * NUM_ATTR + attr];
                if (isfinite(clock_slew)) value = clock_slew;
            }
            return value;
        };
        if (row.kind == 0 || row.positive_unate) {
            slew_r = slew_value(2);
            slew_f = slew_value(3);
        } else {
            slew_r = slew_value(3);
            slew_f = slew_value(2);
        }
    }
    float energy = 0.0f;
    int rf_count = 0;
    if (isfinite(slew_r)) {
        const float e = power_allocator->query_internal_power(row.internal_power_id, 0, slew_r, load_internal);
        if (isfinite(e)) { energy += e; rf_count++; }
    }
    if (isfinite(slew_f)) {
        const float e = power_allocator->query_internal_power(row.internal_power_id, 1, slew_f, load_internal);
        if (isfinite(e)) { energy += e; rf_count++; }
    }
    if (rf_count == 0) return;
    energy /= static_cast<float>(rf_count);
    const float energy_unit = (isfinite(row.energy_unit) && row.energy_unit > 0.0f)
        ? row.energy_unit
        : fallback_energy_unit;
    const float p = weight * energy * energy_unit * density[row.to_pin];
    if (isfinite(p)) {
        if (row_internal) row_internal[idx] = p;
        if (p != 0.0f && inst_internal) atomicAdd(&inst_internal[row.node_id], p);
    }
}

__global__ void power_leakage_row_kernel(const GpuPowerLeakageRowHost* rows,
                                         int num_rows,
                                         const GpuPowerExprOpHost* expr_ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         const float* density,
                                         const float* duty,
                                         float* group_cond_leakage,
                                         float* group_cond_duty_sum,
                                         int* group_cond_count,
                                         float* row_weighted) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (row_weighted) row_weighted[idx] = 0.0f;
    const auto row = rows[idx];
    if (row.group_id < 0) return;
    float cond_duty = 1.0f;
    if (row.when_expr_id >= 0) {
        cond_duty = power_expr_duty_cuda(row.when_expr_id, expr_ops, expr_start, expr_count, density, duty);
    }
    const float weighted = row.leakage * cond_duty;
    if (isfinite(weighted)) {
        atomicAdd(&group_cond_leakage[row.group_id], weighted);
        if (row.leakage > 0.0f) atomicAdd(&group_cond_duty_sum[row.group_id], cond_duty);
        atomicAdd(&group_cond_count[row.group_id], 1);
        if (row_weighted) row_weighted[idx] = weighted;
    }
}

__global__ void power_leakage_summary_kernel(const GpuPowerLeakageGroupHost* groups,
                                             int num_groups,
                                             const float* group_cond_leakage,
                                             const float* group_cond_duty_sum,
                                             const int* group_cond_count,
                                             int num_nodes,
                                             float* inst_leakage) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_groups) return;
    const auto group = groups[idx];
    if (group.node_id < 0 || group.node_id >= num_nodes) return;
    float leakage = 0.0f;
    if (group_cond_count[idx] > 0) {
        float fallback_duty = 1.0f - group_cond_duty_sum[idx];
        leakage = group_cond_leakage[idx] + group.cell_leakage * fallback_duty;
    } else {
        leakage = group.cell_leakage;
    }
    if (isfinite(leakage) && leakage != 0.0f) atomicAdd(&inst_leakage[group.node_id], leakage);
}

}  // namespace

void clear_power_cuda_error() { cudaGetLastError(); }

void run_power_activity_cuda_launcher(int n,
                                      const std::vector<int>& level_list_end_cpu,
                                      index_type* d_level_list,
                                      const int* d_pin_power_level,
                                      index_type* d_pin_forward_arc_list_end,
                                      index_type* d_pin_forward_arc_list,
                                      index_type* d_timing_arc_to_pin_id,
                                      int* d_arc_types,
                                      int* d_arc_id2test_id,
                                      const int* d_pin2net_map,
                                      const int* d_net_driver_pin,
                                      const int* d_flat_net2pin_start_map,
                                      const int* d_flat_net2pin_map,
                                      uint8_t* d_is_load_pin,
                                      uint8_t* d_is_driver_pin,
                                      uint8_t* d_is_cell_pin,
                                      uint8_t* d_is_seq_output_pin,
                                      int* d_clock_gate_out_for_input,
                                      int* d_clock_gate_clock_for_out,
                                      int* d_clock_gate_enable_for_out,
                                      int* d_primary_inputs,
                                      int num_primary_inputs,
                                      int* d_case_values,
                                      int* d_clock_pins,
                                      int num_clock_pins,
                                      GpuPowerExprOpHost* d_expr_ops,
                                      int* d_expr_start,
                                      int* d_expr_count,
                                      int* d_pin_func_expr_id,
                                      GpuPowerSeqHost* d_seqs,
                                      int num_seqs,
                                      int* d_pin_seq_list_start,
                                      int* d_pin_seq_list,
                                      int* d_feedback_seed_pins,
                                      int num_feedback_seed_pins,
                                      int* d_feedback_seed_seqs,
                                      int num_feedback_seed_seqs,
                                      float default_density,
                                      float clock_density,
                                      float time_unit,
                                      int max_activity_passes,
                                      float* d_out,
                                      int num_nodes,
                                      const int* d_pin2node_map,
                                      const float* d_pinLoad,
                                      const double* d_dmp_C1,
                                      const double* d_dmp_C2,
                                      const float* d_pinSlew,
                                      const float* d_power_clock_slews,
                                      bool allow_clock_activity_override,
                                      GpuPowerInternalHost* d_internal_rows,
                                      int num_internal_rows,
                                      int num_internal_denom_groups,
                                      GPUPowerLutAllocator* d_power_allocator,
                                      float cap_unit,
                                      float voltage,
                                      float* d_inst_switching,
                                      float* d_pin_switching,
                                      float* d_inst_internal,
                                      float* d_internal_row_power,
                                      GpuPowerLeakageRowHost* d_leakage_rows,
                                      int num_leakage_rows,
                                      GpuPowerLeakageGroupHost* d_leakage_groups,
                                      int num_leakage_groups,
                                      float* d_inst_leakage,
                                      float* d_leakage_row_power) {
    float* d_density = nullptr;
    float* d_duty = nullptr;
    int* d_origin = nullptr;
    int* d_active = nullptr;
    uint8_t* d_active_level = nullptr;
    int* d_pending_seq = nullptr;
    int* d_pending_seq_count = nullptr;
    const int num_power_levels = std::max(0, static_cast<int>(level_list_end_cpu.size()) - 1);
    cudaMalloc(&d_density, sizeof(float) * n);
    cudaMalloc(&d_duty, sizeof(float) * n);
    cudaMalloc(&d_origin, sizeof(int) * n);
    cudaMalloc(&d_active, sizeof(int) * n);
    cudaMalloc(&d_active_level, sizeof(uint8_t) * std::max(1, num_power_levels));
    cudaMalloc(&d_pending_seq, sizeof(int) * std::max(1, num_seqs));
    cudaMalloc(&d_pending_seq_count, sizeof(int));
    cudaMemset(d_density, 0, sizeof(float) * n);
    cudaMemset(d_duty, 0, sizeof(float) * n);
    cudaMemset(d_origin, 0, sizeof(int) * n);
    cudaMemset(d_active, 0, sizeof(int) * n);
    cudaMemset(d_active_level, 0, sizeof(uint8_t) * std::max(1, num_power_levels));
    cudaMemset(d_pending_seq, 0, sizeof(int) * std::max(1, num_seqs));
    cudaMemset(d_pending_seq_count, 0, sizeof(int));
    cudaMemcpyToSymbol(g_power_allow_clock_activity_override,
                       &allow_clock_activity_override,
                       sizeof(bool));

    bool use_frontier = false;
    if (const char* env_frontier = std::getenv("XPLACE_POWER_ACTIVITY_FRONTIER"))
        use_frontier = std::atoi(env_frontier) != 0;
    cudaDeviceProp prop{};
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);
    if (use_frontier && !prop.cooperativeLaunch) {
        fprintf(stderr, "[power_frontier] cooperative launch unsupported; falling back to level scan\n");
        use_frontier = false;
    }
    int max_comb_sweeps = 1000;
    if (const char* env_comb_sweeps = std::getenv("XPLACE_POWER_ACTIVITY_MAX_COMB_SWEEPS"))
        max_comb_sweeps = std::max(1, std::atoi(env_comb_sweeps));

    const int num_feedback_seed_items = std::max(num_feedback_seed_pins, num_feedback_seed_seqs);
    if (num_feedback_seed_items > 0) {
        power_seed_seq_feedback_state_kernel<<<BLOCK_NUMBER(num_feedback_seed_items), BLOCK_SIZE>>>(
            d_feedback_seed_pins, num_feedback_seed_pins,
            d_feedback_seed_seqs, num_feedback_seed_seqs,
            default_density, d_pinSlew, time_unit, d_density, d_duty, d_origin,
            d_pending_seq, d_pending_seq_count);
        cudaDeviceSynchronize();
    }

    if (use_frontier) {
        int *d_level_offsets = nullptr, *d_level_queue = nullptr, *d_level_counts = nullptr;
        int *d_queued = nullptr, *d_overflow = nullptr;
        cudaMalloc(&d_level_offsets, sizeof(int) * std::max(1, num_power_levels + 1));
        cudaMalloc(&d_level_queue, sizeof(int) * std::max(1, n));
        cudaMalloc(&d_level_counts, sizeof(int) * std::max(1, num_power_levels));
        cudaMalloc(&d_queued, sizeof(int) * std::max(1, n));
        cudaMalloc(&d_overflow, sizeof(int));
        if (num_power_levels + 1 > 0) {
            cudaMemcpy(d_level_offsets, level_list_end_cpu.data(), sizeof(int) * (num_power_levels + 1), cudaMemcpyHostToDevice);
        }
        cudaMemset(d_level_queue, 0, sizeof(int) * std::max(1, n));
        cudaMemset(d_level_counts, 0, sizeof(int) * std::max(1, num_power_levels));
        cudaMemset(d_queued, 0, sizeof(int) * std::max(1, n));
        cudaMemset(d_overflow, 0, sizeof(int));

        if (d_case_values) {
            power_seed_case_level_queue_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                n, d_case_values, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_level_offsets, num_power_levels,
                d_level_queue, d_level_counts, d_queued, d_overflow);
        }
        if (num_primary_inputs > 0) {
            power_seed_pi_level_queue_kernel<<<BLOCK_NUMBER(num_primary_inputs), BLOCK_SIZE>>>(
                d_primary_inputs, num_primary_inputs, default_density, clock_density, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_level_offsets, num_power_levels,
                d_level_queue, d_level_counts, d_queued, d_overflow);
        }
        if (num_clock_pins > 0) {
            power_seed_clock_level_queue_kernel<<<BLOCK_NUMBER(num_clock_pins), BLOCK_SIZE>>>(
                d_clock_pins, num_clock_pins, clock_density, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_level_offsets, num_power_levels,
                d_level_queue, d_level_counts, d_queued, d_overflow);
        }
        cudaDeviceSynchronize();

        int blocks_per_sm = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, power_activity_level_queue_persistent_kernel, BLOCK_SIZE, 0);
        int coop_blocks = std::max(1, prop.multiProcessorCount * std::max(1, blocks_per_sm));
        int num_power_levels_arg = num_power_levels;
        void* args[] = {
            &n,
            &num_power_levels_arg,
            &d_level_offsets,
            &d_case_values,
            &d_is_load_pin,
            &d_is_driver_pin,
            &d_pin2net_map,
            &d_net_driver_pin,
            &d_flat_net2pin_start_map,
            &d_flat_net2pin_map,
            &d_pin_func_expr_id,
            &d_clock_gate_out_for_input,
            &d_clock_gate_clock_for_out,
            &d_clock_gate_enable_for_out,
            &d_expr_ops,
            &d_expr_start,
            &d_expr_count,
            &clock_density,
            &d_pinSlew,
            &time_unit,
            &d_density,
            &d_duty,
            &d_origin,
            &d_pin_forward_arc_list_end,
            &d_pin_forward_arc_list,
            &d_timing_arc_to_pin_id,
            &d_arc_types,
            &d_arc_id2test_id,
            &d_is_seq_output_pin,
            &d_pin_seq_list_start,
            &d_pin_seq_list,
            &d_seqs,
            &num_seqs,
            &d_pin_power_level,
            &d_pending_seq,
            &d_pending_seq_count,
            &d_level_queue,
            &d_level_counts,
            &d_queued,
            &max_activity_passes,
            &d_overflow
        };
        cudaError_t launch_err = cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(power_activity_level_queue_persistent_kernel),
            dim3(coop_blocks), dim3(BLOCK_SIZE), args, 0, nullptr);
        if (launch_err != cudaSuccess) {
            fprintf(stderr, "[power_frontier] cooperative level-queue launch failed: %s\n", cudaGetErrorString(launch_err));
        }
        cudaDeviceSynchronize();
        int overflow = 0;
        cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost);
        if (overflow) fprintf(stderr, "[power_frontier] level queue overflow detected; results may be incomplete\n");
        cudaFree(d_level_offsets);
        cudaFree(d_level_queue);
        cudaFree(d_level_counts);
        cudaFree(d_queued);
        cudaFree(d_overflow);
    } else {
        if (d_case_values) {
            power_seed_case_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                n, d_case_values, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_active_level, num_power_levels, d_active);
        }
        if (num_primary_inputs > 0) {
            power_seed_pi_kernel<<<BLOCK_NUMBER(num_primary_inputs), BLOCK_SIZE>>>(
                d_primary_inputs, num_primary_inputs, default_density, clock_density, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_active_level, num_power_levels, d_active);
        }
        if (num_clock_pins > 0) {
            power_seed_clock_active_kernel<<<BLOCK_NUMBER(num_clock_pins), BLOCK_SIZE>>>(
                d_clock_pins, num_clock_pins, clock_density, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_active_level, num_power_levels, d_active);
        }
        cudaDeviceSynchronize();

        std::vector<uint8_t> active_levels(std::max(1, num_power_levels));
        auto run_bfs = [&]() {
            for (int level = 0; level + 1 < static_cast<int>(level_list_end_cpu.size()); level++) {
                const int start = level_list_end_cpu[level];
                const int count = level_list_end_cpu[level + 1] - start;
                if (count <= 0) continue;
                if (d_active_level && level >= 0 && level < num_power_levels) {
                    cudaMemsetAsync(d_active_level + level, 0, sizeof(uint8_t));
                }
                power_visit_level_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                    d_level_list, start, count, d_case_values, d_is_load_pin, d_is_driver_pin,
                    d_pin2net_map, d_net_driver_pin, d_flat_net2pin_start_map, d_flat_net2pin_map,
                    d_pin_func_expr_id, d_clock_gate_out_for_input,
                    d_clock_gate_clock_for_out, d_clock_gate_enable_for_out,
                    d_expr_ops, d_expr_start, d_expr_count, clock_density,
                    d_pinSlew, time_unit, d_density, d_duty, d_origin, d_active,
                    d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                    d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                    d_pin_seq_list_start, d_pin_seq_list,
                    d_pin_power_level, d_active_level, num_power_levels, level,
                    d_pending_seq, d_pending_seq_count);
            }
            cudaDeviceSynchronize();
            if (num_power_levels <= 0) return false;
            cudaMemcpy(active_levels.data(), d_active_level, sizeof(uint8_t) * num_power_levels,
                       cudaMemcpyDeviceToHost);
            for (int level = 0; level < num_power_levels; ++level) {
                if (active_levels[level]) return true;
            }
            return false;
        };

        const bool print_pass_stats = std::getenv("XPLACE_POWER_PRINT_PASS_STATS") != nullptr;
        int total_comb_sweeps = 0;
        auto drain_bfs = [&]() {
            int sweeps = 0;
            for (int sweep = 0; sweep < max_comb_sweeps; ++sweep) {
                sweeps++;
                if (!run_bfs()) break;
            }
            total_comb_sweeps += sweeps;
            return sweeps;
        };

        drain_bfs();
        int pending_count = 0;
        cudaMemcpy(&pending_count, d_pending_seq_count, sizeof(int), cudaMemcpyDeviceToHost);
        int seq_passes = 0;
        for (int pass = 1; pending_count > 0 && pass < max_activity_passes; pass++) {
            seq_passes = pass;
            if (num_seqs > 0) {
                power_seed_seq_kernel<<<BLOCK_NUMBER(num_seqs), BLOCK_SIZE>>>(
                    d_seqs, num_seqs, d_expr_ops, d_expr_start, d_expr_count, clock_density,
                    d_pinSlew, time_unit, d_density, d_duty, d_origin, d_pending_seq, d_pending_seq_count,
                    d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                    d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                    d_pin_power_level, d_active_level, num_power_levels, d_active);
                cudaDeviceSynchronize();
            }
            const int comb_sweeps = drain_bfs();
            cudaMemcpy(&pending_count, d_pending_seq_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (print_pass_stats && std::getenv("XPLACE_POWER_PRINT_PASS_STATS_VERBOSE")) {
                fprintf(stderr,
                        "[power_activity_pass] pass=%d pending=%d comb_sweeps=%d\n",
                        pass, pending_count, comb_sweeps);
            }
        }
        if (print_pass_stats) {
            fprintf(stderr,
                    "[power_activity_passes] seq_passes=%d final_pending=%d total_comb_sweeps=%d max_seq_passes=%d max_comb_sweeps=%d\n",
                    seq_passes, pending_count, total_comb_sweeps, max_activity_passes, max_comb_sweeps);
        }
    }

    power_pack_output_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(n, d_density, d_duty, d_origin, d_out);
    cudaDeviceSynchronize();
    if (d_inst_switching && d_pin_switching && d_pin2node_map && d_pinLoad) {
        cudaMemset(d_inst_switching, 0, sizeof(float) * num_nodes);
        cudaMemset(d_pin_switching, 0, sizeof(float) * n);
        power_switching_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
            n, num_nodes, d_is_driver_pin, d_is_cell_pin, d_pin2node_map, d_pinLoad, d_dmp_C1, d_dmp_C2, cap_unit, voltage,
            d_out, d_inst_switching, d_pin_switching);
        cudaDeviceSynchronize();
    }
    if ((d_inst_internal || d_internal_row_power) && d_internal_rows && num_internal_rows > 0 && d_power_allocator) {
        if (d_inst_internal) cudaMemset(d_inst_internal, 0, sizeof(float) * num_nodes);
        if (d_internal_row_power) cudaMemset(d_internal_row_power, 0, sizeof(float) * num_internal_rows);
        float* d_denom = nullptr;
        cudaMalloc(&d_denom, sizeof(float) * std::max(1, num_internal_denom_groups));
        cudaMemset(d_denom, 0, sizeof(float) * std::max(1, num_internal_denom_groups));
        power_internal_denom_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_internal_rows, num_internal_rows, d_expr_ops, d_expr_start, d_expr_count,
            d_density, d_duty, d_denom);
        cudaDeviceSynchronize();
        power_internal_contrib_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_internal_rows, num_internal_rows, num_nodes, d_expr_ops, d_expr_start, d_expr_count,
            d_density, d_duty, d_pinSlew, d_power_clock_slews, d_dmp_C1, d_dmp_C2, d_denom, d_power_allocator, cap_unit, d_inst_internal, d_internal_row_power);
        cudaDeviceSynchronize();
        cudaFree(d_denom);
    }
    if ((d_inst_leakage || d_leakage_row_power) && d_leakage_groups && num_leakage_groups > 0) {
        if (d_inst_leakage) cudaMemset(d_inst_leakage, 0, sizeof(float) * num_nodes);
        if (d_leakage_row_power && num_leakage_rows > 0) cudaMemset(d_leakage_row_power, 0, sizeof(float) * num_leakage_rows);
        float* d_group_cond_leakage = nullptr;
        float* d_group_cond_duty_sum = nullptr;
        int* d_group_cond_count = nullptr;
        cudaMalloc(&d_group_cond_leakage, sizeof(float) * num_leakage_groups);
        cudaMalloc(&d_group_cond_duty_sum, sizeof(float) * num_leakage_groups);
        cudaMalloc(&d_group_cond_count, sizeof(int) * num_leakage_groups);
        cudaMemset(d_group_cond_leakage, 0, sizeof(float) * num_leakage_groups);
        cudaMemset(d_group_cond_duty_sum, 0, sizeof(float) * num_leakage_groups);
        cudaMemset(d_group_cond_count, 0, sizeof(int) * num_leakage_groups);
        if (d_leakage_rows && num_leakage_rows > 0) {
            power_leakage_row_kernel<<<BLOCK_NUMBER(num_leakage_rows), BLOCK_SIZE>>>(
                d_leakage_rows, num_leakage_rows, d_expr_ops, d_expr_start, d_expr_count,
                d_density, d_duty, d_group_cond_leakage, d_group_cond_duty_sum, d_group_cond_count, d_leakage_row_power);
            cudaDeviceSynchronize();
        }
        if (d_inst_leakage) {
            power_leakage_summary_kernel<<<BLOCK_NUMBER(num_leakage_groups), BLOCK_SIZE>>>(
                d_leakage_groups, num_leakage_groups, d_group_cond_leakage, d_group_cond_duty_sum, d_group_cond_count,
                num_nodes, d_inst_leakage);
            cudaDeviceSynchronize();
        }
        cudaFree(d_group_cond_leakage);
        cudaFree(d_group_cond_duty_sum);
        cudaFree(d_group_cond_count);
    }

    cudaFree(d_density);
    cudaFree(d_duty);
    cudaFree(d_origin);
    cudaFree(d_active);
    cudaFree(d_active_level);
    cudaFree(d_pending_seq);
    cudaFree(d_pending_seq_count);
}

}  // namespace gt
