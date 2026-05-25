
#include "gputiming.h"
#include "utils.cuh"
#include "GPUTimer.h"

#include <algorithm>
#include <cctype>
#include <cooperative_groups.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gt {

static bool read_power_bool_env_host(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env) return default_value;
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value.empty() || value == "0" || value == "false" || value == "no");
}

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
__device__ float g_power_min_activity_density = 1.0e-10f;
__device__ float g_power_min_activity_duty = 0.0f;
__device__ bool g_power_disable_activity_slew_cap = false;
__device__ float g_power_seq_clock_limit_rel_tol = 0.0f;
__device__ float g_power_seq_pending_min_density = 0.0f;
__device__ float g_power_activity_clock_density_cap = 3.4028234663852886e38f;
__device__ int g_power_direct_ordered_seq_seed = 0;
__device__ int g_power_require_known_seq_data = 0;

__device__ __forceinline__ float power_percent_change(float value, float prev) {
    if (prev == 0.0f) return value == 0.0f ? 0.0f : 1.0f;
    return fabsf(value - prev) / fabsf(prev);
}

__device__ __forceinline__ float power_clamp_activity_duty(float duty) {
    float u = fminf(fmaxf(duty, 0.0f), 1.0f);
    const float eps = fmaxf(g_power_min_activity_duty, 0.0f);
    if (eps > 0.0f) {
        if (u < eps) u = 0.0f;
        else if ((1.0f - u) < eps) u = 1.0f;
    }
    return u;
}

__device__ __forceinline__ bool power_should_mark_pending_seq(float density) {
    return density >= fmaxf(g_power_seq_pending_min_density, 0.0f);
}

__device__ __forceinline__ float power_max_activity_density_from_slew(int pin,
                                                                      const float* pinSlew,
                                                                      float time_unit) {
    if (g_power_disable_activity_slew_cap) return 3.4028234663852886e38f;
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

__device__ __forceinline__ bool power_seq_density_exceeds_clock_limit(float in_density,
                                                                      float clk_density) {
    const float limit = clk_density * 0.5f;
    return in_density > limit * (1.0f + fmaxf(g_power_seq_clock_limit_rel_tol, 0.0f));
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
    const float max_density = force
        ? g_power_activity_clock_density_cap
        : fminf(power_max_activity_density_from_slew(pin, pinSlew, time_unit),
                g_power_activity_clock_density_cap);
    float d = fminf(fmaxf(new_density, 0.0f), max_density);
    if (fabsf(d) < g_power_min_activity_density) d = 0.0f;
    const float u = power_clamp_activity_duty(new_duty);
    const bool value_changed = power_percent_change(d, prev_density) > 0.01f
        || power_percent_change(u, prev_duty) > 0.01f;
    const bool changed = value_changed || prev_origin != new_origin;
    density[pin] = d;
    duty[pin] = u;
    origin[pin] = new_origin;
    return changed;
}

__device__ void power_enqueue_adjacent(int pin,
                                       const uint8_t* is_load_pin,
                                       const int* pin2net_map,
                                       const int* net_driver_pin,
                                       const int* flat_net2pin_start_map,
                                       const int* flat_net2pin_map,
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
    if (is_load_pin && pin2net_map && net_driver_pin && flat_net2pin_start_map && flat_net2pin_map) {
        const int net = pin2net_map[pin];
        if (net >= 0 && net_driver_pin[net] == pin) {
            const int start = flat_net2pin_start_map[net];
            const int end = flat_net2pin_start_map[net + 1];
            for (int pos = start; pos < end; ++pos) {
                const int sink = flat_net2pin_map[pos];
                if (sink < 0 || sink == pin || !is_load_pin[sink]) continue;
                atomicExch(&active[sink], 1);
                if (pin_power_level && active_level) {
                    const int level = pin_power_level[sink];
                    if (level >= 0 && level < num_power_levels) active_level[level] = 1;
                }
            }
        }
    }
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

__device__ void power_activate_pin(int pin,
                                   const int* pin_power_level,
                                   uint8_t* active_level,
                                   int num_power_levels,
                                   int* active) {
    if (pin < 0 || !active) return;
    atomicExch(&active[pin], 1);
    if (pin_power_level && active_level) {
        const int level = pin_power_level[pin];
        if (level >= 0 && level < num_power_levels) active_level[level] = 1;
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
    if (origin && origin[clk] == 0 && origin[en] == 0) return false;
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

__global__ void power_snapshot_level_active_kernel(const int* level_list,
                                                   int level_start,
                                                   int num_level_pins,
                                                   int* active,
                                                   uint8_t* visit_active) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = level_list[level_start + pos];
    if (pin < 0) return;
    visit_active[pin] = static_cast<uint8_t>(atomicExch(&active[pin], 0) != 0);
}

__global__ void power_snapshot_level_active_list_kernel(const int* level_list,
                                                        int level_start,
                                                        int num_level_pins,
                                                        int* active,
                                                        uint8_t* visit_active,
                                                        int* active_count,
                                                        int* active_pins) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = level_list[level_start + pos];
    if (pin < 0) return;
    const bool is_active = atomicExch(&active[pin], 0) != 0;
    if (visit_active) visit_active[pin] = static_cast<uint8_t>(is_active);
    if (!is_active || !active_count || !active_pins) return;
    const int out_pos = atomicAdd(active_count, 1);
    active_pins[out_pos] = pin;
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
            case 7:
                if (sp >= 128) return false;
                stack[sp++] = -1;
                break;
            default:
                return false;
        }
    }
    if (sp != 1 || stack[0] < 0) return false;
    value = stack[0];
    return true;
}

constexpr int POWER_BDD_MAX_VARS = 32;
constexpr int POWER_BDD_MAX_NODES = 256;
constexpr int POWER_BDD_MAX_APPLY_CACHE = 1024;

struct PowerBddNodeCuda {
    int var = -1;
    int low = 0;
    int high = 0;
};

struct PowerBddApplyCacheCuda {
    int op = -1;
    int left = 0;
    int right = 0;
    int result = 0;
};

struct PowerBddContextCuda {
    PowerBddNodeCuda nodes[POWER_BDD_MAX_NODES];
    PowerBddApplyCacheCuda apply_cache[POWER_BDD_MAX_APPLY_CACHE];
    int node_count = 0;
    int apply_count = 0;
    int var_pins[POWER_BDD_MAX_VARS];
    int var_keys[POWER_BDD_MAX_VARS];
    uint8_t var_has_pin[POWER_BDD_MAX_VARS];
    float var_duties[POWER_BDD_MAX_VARS];
    float var_densities[POWER_BDD_MAX_VARS];
    int var_count = 0;
    bool ok = true;
};

__device__ __forceinline__ int power_bdd_edge_id(int edge) { return edge >> 1; }
__device__ __forceinline__ bool power_bdd_edge_inv(int edge) { return (edge & 1) != 0; }
__device__ __forceinline__ int power_bdd_not(int edge) { return edge ^ 1; }

__device__ int power_bdd_make_node(PowerBddContextCuda& ctx, int var, int low, int high) {
    if (low == high) return low;
    bool result_inv = false;
    // Mirror CUDD's complemented-edge normalization: the then/high edge is
    // stored regular and a complement is moved onto the returned edge.
    if (power_bdd_edge_inv(high)) {
        low = power_bdd_not(low);
        high = power_bdd_not(high);
        result_inv = true;
    }
    for (int i = 0; i < ctx.node_count; i++) {
        const auto& node = ctx.nodes[i];
        if (node.var == var && node.low == low && node.high == high) {
            const int edge = (i + 1) << 1;
            return result_inv ? power_bdd_not(edge) : edge;
        }
    }
    if (ctx.node_count >= POWER_BDD_MAX_NODES) {
        ctx.ok = false;
        return 1;
    }
    const int id = ++ctx.node_count;
    ctx.nodes[id - 1] = PowerBddNodeCuda{var, low, high};
    const int edge = id << 1;
    return result_inv ? power_bdd_not(edge) : edge;
}

__device__ int power_bdd_top_var(const PowerBddContextCuda& ctx, int edge) {
    const int id = power_bdd_edge_id(edge);
    return id == 0 ? 0x3fffffff : ctx.nodes[id - 1].var;
}

__device__ int power_bdd_cof_top(const PowerBddContextCuda& ctx, int edge, int var, bool high_child) {
    const int id = power_bdd_edge_id(edge);
    if (id == 0 || ctx.nodes[id - 1].var != var) return edge;
    const int child = high_child ? ctx.nodes[id - 1].high : ctx.nodes[id - 1].low;
    return power_bdd_edge_inv(edge) ? power_bdd_not(child) : child;
}

__device__ int power_bdd_apply(PowerBddContextCuda& ctx, int op, int left, int right) {
    if (op >= 0 && op <= 2 && right < left) {
        const int tmp = left;
        left = right;
        right = tmp;
    }
    for (int i = 0; i < ctx.apply_count; i++) {
        const auto& cache = ctx.apply_cache[i];
        if (cache.op == op && cache.left == left && cache.right == right) return cache.result;
    }

    int result = 1;
    const int left_id = power_bdd_edge_id(left);
    const int right_id = power_bdd_edge_id(right);
    if (left_id == 0 && right_id == 0) {
        const bool left_value = !power_bdd_edge_inv(left);
        const bool right_value = !power_bdd_edge_inv(right);
        bool value = false;
        if (op == 0) value = left_value && right_value;
        else if (op == 1) value = left_value || right_value;
        else value = left_value != right_value;
        result = value ? 0 : 1;
    } else {
        const int var = min(power_bdd_top_var(ctx, left), power_bdd_top_var(ctx, right));
        const int low = power_bdd_apply(ctx, op,
                                        power_bdd_cof_top(ctx, left, var, false),
                                        power_bdd_cof_top(ctx, right, var, false));
        const int high = power_bdd_apply(ctx, op,
                                         power_bdd_cof_top(ctx, left, var, true),
                                         power_bdd_cof_top(ctx, right, var, true));
        result = power_bdd_make_node(ctx, var, low, high);
    }

    if (ctx.apply_count < POWER_BDD_MAX_APPLY_CACHE) {
        ctx.apply_cache[ctx.apply_count++] = PowerBddApplyCacheCuda{op, left, right, result};
    }
    return result;
}

__device__ int power_bdd_restrict(PowerBddContextCuda& ctx, int edge, int target_var, bool high_child) {
    const int id = power_bdd_edge_id(edge);
    if (id == 0) return edge;
    const auto node = ctx.nodes[id - 1];
    if (node.var > target_var) return edge;
    int result = edge;
    if (node.var == target_var) {
        result = high_child ? node.high : node.low;
    } else {
        const int low = power_bdd_restrict(ctx, node.low, target_var, high_child);
        const int high = power_bdd_restrict(ctx, node.high, target_var, high_child);
        result = power_bdd_make_node(ctx, node.var, low, high);
    }
    return power_bdd_edge_inv(edge) ? power_bdd_not(result) : result;
}

__device__ float power_bdd_eval_duty(const PowerBddContextCuda& ctx, int edge) {
    const int id = power_bdd_edge_id(edge);
    if (id == 0) return power_bdd_edge_inv(edge) ? 0.0f : 1.0f;
    const auto node = ctx.nodes[id - 1];
    if (node.var >= 0 && node.var < ctx.var_count && !ctx.var_has_pin[node.var])
        return 0.0f;
    const float duty0 = power_bdd_eval_duty(ctx, node.low);
    const float duty1 = power_bdd_eval_duty(ctx, node.high);
    const float var_duty = ctx.var_duties[node.var];
    const double result_d =
        static_cast<double>(duty0) * (1.0 - static_cast<double>(var_duty)) +
        static_cast<double>(duty1) * static_cast<double>(var_duty);
    float result = static_cast<float>(result_d);
    if (power_bdd_edge_inv(edge)) result = 1.0f - result;
    return fminf(fmaxf(result, 0.0f), 1.0f);
}

__device__ int power_bdd_ensure_var(PowerBddContextCuda& ctx,
                                    int var_key,
                                    int pin,
                                    const float* pin_density,
                                    const float* pin_duty,
                                    bool zero_density = false) {
    for (int i = 0; i < ctx.var_count; i++) {
        if (ctx.var_keys[i] == var_key && zero_density)
            ctx.var_densities[i] = 0.0f;
        if (ctx.var_keys[i] == var_key) return i;
    }
    if (ctx.var_count >= POWER_BDD_MAX_VARS) {
        ctx.ok = false;
        return -1;
    }
    const int var = ctx.var_count++;
    ctx.var_keys[var] = var_key;
    ctx.var_pins[var] = pin;
    ctx.var_has_pin[var] = pin >= 0 ? 1 : 0;
    ctx.var_duties[var] = pin >= 0 ? power_clamp_activity_duty(pin_duty[pin]) : 0.0f;
    ctx.var_densities[var] = (pin >= 0 && !zero_density && pin_density) ? pin_density[pin] : 0.0f;
    return var;
}

__device__ int power_expr_resolve_pin_arg(int arg,
                                          const int* node_port_pin_start,
                                          const int* node_port_pin_list,
                                          int node_id) {
    if (arg >= 0) return arg;
    if (arg == -1 || !node_port_pin_start || !node_port_pin_list || node_id < 0) return -1;
    const int port_id = -2 - arg;
    const int start = node_port_pin_start[node_id];
    const int end = node_port_pin_start[node_id + 1];
    if (port_id < 0 || start + port_id < start || start + port_id >= end) return -1;
    return node_port_pin_list[start + port_id];
}

__device__ bool power_bdd_build_expr(int expr_id,
                                     const GpuPowerExprOpHost* ops,
                                     const int* expr_start,
                                     const int* expr_count,
                                     const float* pin_density,
                                     const float* pin_duty,
                                     PowerBddContextCuda& ctx,
                                     int& root,
                                     const int* node_port_pin_start = nullptr,
                                     const int* node_port_pin_list = nullptr,
                                     int node_id = -1) {
    if (expr_id < 0) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    int pre_keys[POWER_BDD_MAX_VARS];
    int pre_pins[POWER_BDD_MAX_VARS];
    uint8_t pre_zero_density[POWER_BDD_MAX_VARS];
    int pre_count = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        int pin = -1;
        int var_key = -1;
        bool zero_density = op.zero_density != 0;
        if (op.op == 0) {
            pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                             node_port_pin_list, node_id);
            if (pin < 0 && op.arg > -2) return false;
            if (pin < 0) continue;
            var_key = op.var_key >= 0 ? op.var_key : pin;
        } else if (op.op == 7) {
            var_key = op.var_key;
            zero_density = true;
        } else {
            continue;
        }
        if (var_key < 0) return false;
        int pos = -1;
        for (int i = 0; i < pre_count; i++) {
            if (pre_keys[i] == var_key) {
                pos = i;
                break;
            }
        }
        if (pos < 0) {
            if (pre_count >= POWER_BDD_MAX_VARS) return false;
            pos = pre_count++;
            pre_keys[pos] = var_key;
            pre_pins[pos] = pin;
            pre_zero_density[pos] = zero_density ? 1 : 0;
        } else if (zero_density) {
            pre_zero_density[pos] = 1;
        }
    }
    for (int i = 1; i < pre_count; i++) {
        const int key = pre_keys[i];
        const int pin = pre_pins[i];
        const uint8_t zero_density = pre_zero_density[i];
        int j = i - 1;
        while (j >= 0 && pre_keys[j] > key) {
            pre_keys[j + 1] = pre_keys[j];
            pre_pins[j + 1] = pre_pins[j];
            pre_zero_density[j + 1] = pre_zero_density[j];
            j--;
        }
        pre_keys[j + 1] = key;
        pre_pins[j + 1] = pin;
        pre_zero_density[j + 1] = zero_density;
    }
    for (int i = 0; i < pre_count; i++) {
        if (power_bdd_ensure_var(ctx, pre_keys[i], pre_pins[i], pin_density, pin_duty,
                                 pre_zero_density[i] != 0) < 0 || !ctx.ok)
            return false;
    }
    int stack[128];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        switch (op.op) {
            case 0: {
                const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                           node_port_pin_list, node_id);
                if (sp >= 128) return false;
                if (pin < 0 && op.arg > -2) return false;
                if (pin < 0) {
                    stack[sp++] = 1;
                    break;
                }
                const int var_key = op.var_key >= 0 ? op.var_key : pin;
                const int var = power_bdd_ensure_var(ctx, var_key, pin, pin_density, pin_duty,
                                                     op.zero_density != 0);
                if (var < 0 || !ctx.ok) return false;
                stack[sp++] = power_bdd_make_node(ctx, var, 1, 0);
                break;
            }
            case 7: {
                if (sp >= 128) return false;
                const int var_key = op.var_key >= 0 ? op.var_key : -1;
                const int var = power_bdd_ensure_var(ctx, var_key, -1, pin_density, pin_duty,
                                                     true);
                if (var < 0 || !ctx.ok) return false;
                stack[sp++] = power_bdd_make_node(ctx, var, 1, 0);
                break;
            }
            case 1:
                if (sp >= 128) return false;
                stack[sp++] = 1;
                break;
            case 2:
                if (sp >= 128) return false;
                stack[sp++] = 0;
                break;
            case 3: {
                if (sp < 1) return false;
                stack[sp - 1] = power_bdd_not(stack[sp - 1]);
                break;
            }
            case 4: {
                if (sp < 2) return false;
                const int right = stack[--sp];
                const int left = stack[--sp];
                stack[sp++] = power_bdd_apply(ctx, 0, left, right);
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const int right = stack[--sp];
                const int left = stack[--sp];
                stack[sp++] = power_bdd_apply(ctx, 1, left, right);
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const int right = stack[--sp];
                const int left = stack[--sp];
                stack[sp++] = power_bdd_apply(ctx, 2, left, right);
                break;
            }
            default:
                return false;
        }
        if (!ctx.ok) return false;
    }
    if (sp != 1) return false;
    root = stack[0];
    return ctx.ok;
}

__device__ bool power_eval_expr_activity(int expr_id,
                                         const GpuPowerExprOpHost* ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         const float* pin_density,
                                         const float* pin_duty,
                                         float& out_density,
                                         float& out_duty,
                                         const int* node_port_pin_start = nullptr,
                                         const int* node_port_pin_list = nullptr,
                                         int node_id = -1) {
    PowerBddContextCuda ctx;
    int root = 1;
    if (!power_bdd_build_expr(expr_id, ops, expr_start, expr_count,
                              pin_density, pin_duty, ctx, root,
                              node_port_pin_start, node_port_pin_list, node_id)) {
        return false;
    }
    out_duty = power_bdd_eval_duty(ctx, root);
    out_density = 0.0f;

    int order[POWER_BDD_MAX_VARS];
    for (int i = 0; i < ctx.var_count; i++) order[i] = i;
    for (int i = 1; i < ctx.var_count; i++) {
        const int item = order[i];
        int j = i - 1;
        while (j >= 0 && ctx.var_keys[order[j]] > ctx.var_keys[item]) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = item;
    }
    for (int idx = 0; idx < ctx.var_count; idx++) {
        const int var = order[idx];
        if (!ctx.var_has_pin[var]) continue;
        const int low = power_bdd_restrict(ctx, root, var, false);
        const int high = power_bdd_restrict(ctx, root, var, true);
        const int diff = power_bdd_apply(ctx, 2, low, high);
        const float diff_duty = power_bdd_eval_duty(ctx, diff);
        out_density += ctx.var_densities[var] * diff_duty;
    }
    return isfinite(out_density) && isfinite(out_duty);
}

__device__ bool power_expr_has_known_activity_input(int expr_id,
                                                    const GpuPowerExprOpHost* ops,
                                                    const int* expr_start,
                                                    const int* expr_count,
                                                    const int* origin,
                                                    const int* node_port_pin_start = nullptr,
                                                    const int* node_port_pin_list = nullptr,
                                                    int node_id = -1) {
    if (expr_id < 0) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    bool has_pin_arg = false;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        if (op.op != 0) continue;
        const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                   node_port_pin_list, node_id);
        if (pin < 0) continue;
        has_pin_arg = true;
        if (!origin || origin[pin] != 0 || op.zero_density != 0) return true;
    }
    return !has_pin_arg;
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
                                     const uint8_t* is_load_pin,
                                     const int* pin2net_map,
                                     const int* net_driver_pin,
                                     const int* flat_net2pin_start_map,
                                     const int* flat_net2pin_map,
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
        power_enqueue_adjacent(pin, is_load_pin, pin2net_map, net_driver_pin,
                               flat_net2pin_start_map, flat_net2pin_map,
                               pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                               arc_types, arc_id2test_id, is_seq_output_pin,
                               pin_power_level, active_level, num_power_levels, active);
    }
}

__global__ void power_seed_clock_active_kernel(const int* clock_pins,
                                               int num_clock_pins,
                                               float clock_density,
                                               const float* clock_pin_densities,
                                               const float* clock_pin_duties,
                                               const uint8_t* clock_pin_enqueue,
                                               const float* pinSlew,
                                               float time_unit,
                                               float* density,
                                               float* duty,
                                               int* origin,
                                               const uint8_t* is_load_pin,
                                               const int* pin2net_map,
                                               const int* net_driver_pin,
                                               const int* flat_net2pin_start_map,
                                               const int* flat_net2pin_map,
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
    const float pin_density = clock_pin_densities ? clock_pin_densities[idx] : clock_density;
    const float pin_duty = clock_pin_duties ? clock_pin_duties[idx] : 0.5f;
    const bool enqueue = !clock_pin_enqueue || clock_pin_enqueue[idx] != 0;
    if (power_set_activity(pin, pin_density, pin_duty, 2, true, pinSlew, time_unit, density, duty, origin)
        && enqueue) {
        power_enqueue_adjacent(pin, is_load_pin, pin2net_map, net_driver_pin,
                               flat_net2pin_start_map, flat_net2pin_map,
                               pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
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
                                       const uint8_t* is_load_pin,
                                       const int* pin2net_map,
                                       const int* net_driver_pin,
                                       const int* flat_net2pin_start_map,
                                       const int* flat_net2pin_map,
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
        power_enqueue_adjacent(pin, is_load_pin, pin2net_map, net_driver_pin,
                               flat_net2pin_start_map, flat_net2pin_map,
                               pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
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

__device__ bool power_process_level_pin(int pin,
                                        const int* case_values,
                                        const uint8_t* is_load_pin,
                                        const uint8_t* is_driver_pin,
                                        const int* pin2net_map,
                                        const int* net_driver_pin,
                                        const int* flat_net2pin_start_map,
                                        const int* flat_net2pin_map,
                                        const int* pin_func_expr_id,
                                        const int* missing_func_out_start,
                                        const int* missing_func_out_list,
                                        const float* seq_pin_density,
                                        const float* seq_pin_duty,
                                        const uint8_t* seq_pin_valid,
                                        const int* clock_gate_out_for_input,
                                        const int* clock_gate_clock_for_out,
                                        const int* clock_gate_enable_for_out,
                                        const GpuPowerExprOpHost* expr_ops,
                                        const int* expr_start,
                                        const int* expr_count,
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
                                        int* pending_seq_count,
                                        bool defer_pending_seq) {
    bool changed = false;
    if (case_values && case_values[pin] >= 0) {
        changed = power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true,
                                     pinSlew, time_unit, density, duty, origin);
    } else if (is_load_pin[pin]) {
        const int net = pin2net_map[pin];
        const int driver = (net >= 0 && net_driver_pin) ? net_driver_pin[net] : -1;
        if (driver >= 0 && driver != pin && (!origin || origin[driver] != 0)) {
            changed = power_set_activity(pin, density[driver], duty[driver], 3, false,
                                         pinSlew, time_unit, density, duty, origin);
        }
    }
    if ((!case_values || case_values[pin] < 0) && is_driver_pin[pin]) {
        if (seq_pin_valid && seq_pin_valid[pin]) {
            changed = power_set_activity(pin, seq_pin_density[pin], seq_pin_duty[pin], 3, false,
                                         pinSlew, time_unit, density, duty, origin) || changed;
        } else {
            const int expr_id = pin_func_expr_id[pin];
            if (expr_id >= 0) {
                float out_density = 0.0f, out_duty = 0.0f;
                if (power_eval_expr_activity(expr_id, expr_ops, expr_start, expr_count,
                                             density, duty, out_density, out_duty)) {
                    changed = power_set_activity(pin, out_density, out_duty, 3, false,
                                                 pinSlew, time_unit, density, duty, origin) || changed;
                }
            }
        }
        changed = power_set_clock_gate_output(pin, clock_gate_clock_for_out, clock_gate_enable_for_out,
                                              pinSlew, time_unit, density, duty, origin) || changed;
    }
    if (!changed) return false;
    if (is_load_pin[pin]) {
        if (!defer_pending_seq) {
            if (power_should_mark_pending_seq(density[pin])) {
                for (int i = pin_seq_list_start[pin]; i < pin_seq_list_start[pin + 1]; i++) {
                    const int seq_id = pin_seq_list[i];
                    if (seq_id >= 0 && atomicExch(&pending_seq[seq_id], 1) == 0)
                        atomicAdd(pending_seq_count, 1);
                }
            }
        }
        power_enqueue_clock_gate_output(pin, clock_gate_out_for_input,
                                        pin_power_level, active_level, num_power_levels, active);
        if (missing_func_out_start && missing_func_out_list) {
            for (int i = missing_func_out_start[pin]; i < missing_func_out_start[pin + 1]; ++i) {
                const int out_pin = missing_func_out_list[i];
                if (out_pin < 0) continue;
                const int expr_id = pin_func_expr_id[out_pin];
                if (expr_id < 0) continue;
                float out_density = 0.0f, out_duty = 0.0f;
                if (!power_eval_expr_activity(expr_id, expr_ops, expr_start, expr_count,
                                              density, duty, out_density, out_duty))
                    continue;
                if (power_set_activity(out_pin, out_density, out_duty, 3, false,
                                       pinSlew, time_unit, density, duty, origin)) {
                    power_enqueue_adjacent(out_pin, is_load_pin, pin2net_map, net_driver_pin,
                                           flat_net2pin_start_map, flat_net2pin_map,
                                           pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                           arc_types, arc_id2test_id, is_seq_output_pin,
                                           pin_power_level, active_level, num_power_levels, active);
                }
            }
        }
    }
    power_enqueue_adjacent(pin, is_load_pin, pin2net_map, net_driver_pin,
                           flat_net2pin_start_map, flat_net2pin_map,
                           pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                           arc_types, arc_id2test_id, is_seq_output_pin,
                           pin_power_level, active_level, num_power_levels, active);
    return true;
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
                                         const int* missing_func_out_start,
                                         const int* missing_func_out_list,
                                         const float* seq_pin_density,
                                         const float* seq_pin_duty,
                                         const uint8_t* seq_pin_valid,
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
                                         uint8_t* visit_active,
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
                                         int* pending_seq_count,
                                         bool defer_pending_seq) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = level_list[level_start + pos];
    if (pin < 0 || !visit_active || visit_active[pin] == 0) return;
    visit_active[pin] = 0;
    (void)clock_density;
    power_process_level_pin(
        pin, case_values, is_load_pin, is_driver_pin,
        pin2net_map, net_driver_pin, flat_net2pin_start_map, flat_net2pin_map,
        pin_func_expr_id, missing_func_out_start, missing_func_out_list,
        seq_pin_density, seq_pin_duty, seq_pin_valid,
        clock_gate_out_for_input, clock_gate_clock_for_out, clock_gate_enable_for_out,
        expr_ops, expr_start, expr_count, pinSlew, time_unit,
        density, duty, origin, active,
        pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
        arc_types, arc_id2test_id, is_seq_output_pin,
        pin_seq_list_start, pin_seq_list,
        pin_power_level, active_level, num_power_levels, current_level,
        pending_seq, pending_seq_count, defer_pending_seq);
}

__global__ void power_visit_level_serial_kernel(const index_type* level_list,
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
                                                const int* missing_func_out_start,
                                                const int* missing_func_out_list,
                                                const float* seq_pin_density,
                                                const float* seq_pin_duty,
                                                const uint8_t* seq_pin_valid,
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
                                                int* pending_seq_count,
                                                bool defer_pending_seq) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void)clock_density;
    for (int pos = num_level_pins - 1; pos >= 0; --pos) {
        const int pin = level_list[level_start + pos];
        if (pin < 0) continue;
        if (atomicExch(&active[pin], 0) == 0) continue;
        power_process_level_pin(
            pin, case_values, is_load_pin, is_driver_pin,
            pin2net_map, net_driver_pin, flat_net2pin_start_map, flat_net2pin_map,
            pin_func_expr_id, missing_func_out_start, missing_func_out_list,
            seq_pin_density, seq_pin_duty, seq_pin_valid,
            clock_gate_out_for_input, clock_gate_clock_for_out, clock_gate_enable_for_out,
            expr_ops, expr_start, expr_count, pinSlew, time_unit,
            density, duty, origin, active,
            pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
            arc_types, arc_id2test_id, is_seq_output_pin,
            pin_seq_list_start, pin_seq_list,
            pin_power_level, active_level, num_power_levels, current_level,
            pending_seq, pending_seq_count, defer_pending_seq);
    }
}

__global__ void power_visit_active_list_serial_kernel(const int* active_pins,
                                                      const int* active_count,
                                                      const int* case_values,
                                                      const uint8_t* is_load_pin,
                                                      const uint8_t* is_driver_pin,
                                                      const int* pin2net_map,
                                                      const int* net_driver_pin,
                                                      const int* flat_net2pin_start_map,
                                                      const int* flat_net2pin_map,
                                                      const int* pin_func_expr_id,
                                                      const int* missing_func_out_start,
                                                      const int* missing_func_out_list,
                                                      const float* seq_pin_density,
                                                      const float* seq_pin_duty,
                                                      const uint8_t* seq_pin_valid,
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
                                                      uint8_t* visit_active,
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
                                                      int* pending_seq_count,
                                                      bool defer_pending_seq) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !active_pins || !active_count) return;
    (void)clock_density;
    for (int idx = *active_count - 1; idx >= 0; --idx) {
        const int pin = active_pins[idx];
        if (pin < 0) continue;
        if (visit_active && visit_active[pin] == 0) continue;
        if (visit_active) visit_active[pin] = 0;
        power_process_level_pin(
            pin, case_values, is_load_pin, is_driver_pin,
            pin2net_map, net_driver_pin, flat_net2pin_start_map, flat_net2pin_map,
            pin_func_expr_id, missing_func_out_start, missing_func_out_list,
            seq_pin_density, seq_pin_duty, seq_pin_valid,
            clock_gate_out_for_input, clock_gate_clock_for_out, clock_gate_enable_for_out,
            expr_ops, expr_start, expr_count, pinSlew, time_unit,
            density, duty, origin, active,
            pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
            arc_types, arc_id2test_id, is_seq_output_pin,
            pin_seq_list_start, pin_seq_list,
            pin_power_level, active_level, num_power_levels, current_level,
            pending_seq, pending_seq_count, defer_pending_seq);
    }
}

__global__ void power_mark_pending_seq_changes_kernel(int n,
                                                      const uint8_t* is_load_pin,
                                                      const float* prev_density,
                                                      const float* prev_duty,
                                                      const int* prev_origin,
                                                      const float* density,
                                                      const float* duty,
                                                      const int* origin,
                                                      const int* pin_seq_list_start,
                                                      const int* pin_seq_list,
                                                      int* pending_seq,
                                                      int* pending_seq_count) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin >= n || !is_load_pin || !is_load_pin[pin]) return;
    if (!pin_seq_list_start || !pin_seq_list || !pending_seq || !pending_seq_count)
        return;
    const bool changed = power_percent_change(density[pin], prev_density[pin]) > 0.01f
        || power_percent_change(duty[pin], prev_duty[pin]) > 0.01f
        || origin[pin] != prev_origin[pin];
    if (!changed) return;
    if (!power_should_mark_pending_seq(density[pin])) return;
    for (int i = pin_seq_list_start[pin]; i < pin_seq_list_start[pin + 1]; i++) {
        const int seq_id = pin_seq_list[i];
        if (seq_id >= 0 && atomicExch(&pending_seq[seq_id], 1) == 0)
            atomicAdd(pending_seq_count, 1);
    }
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
                                      float* seq_pin_density,
                                      float* seq_pin_duty,
                                      uint8_t* seq_pin_valid,
                                      int* pending_seq,
                                      int* pending_seq_count,
                                      const uint8_t* is_load_pin,
                                      const int* pin2net_map,
                                      const int* net_driver_pin,
                                      const int* flat_net2pin_start_map,
                                      const int* flat_net2pin_map,
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
    if ((g_power_require_known_seq_data
         && !power_expr_has_known_activity_input(seq.data_expr_id, expr_ops, expr_start, expr_count, origin))
        || !power_eval_expr_activity(seq.data_expr_id, expr_ops, expr_start, expr_count,
                                  density, duty, in_density, in_duty)) return;
    power_eval_expr_activity(seq.clk_expr_id, expr_ops, expr_start, expr_count,
                             density, duty, clk_density, clk_duty);
    float out_density = in_density;
    float out_duty = in_duty;
    if (power_seq_density_exceeds_clock_limit(in_density, clk_density)) {
        out_density = seq.is_latch ? in_density * clk_duty
                                   : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
    }
    if (seq.q_pin >= 0) {
        seq_pin_density[seq.q_pin] = out_density;
        seq_pin_duty[seq.q_pin] = out_duty;
        seq_pin_valid[seq.q_pin] = 1;
        power_activate_pin(seq.q_pin, pin_power_level, active_level, num_power_levels, active);
    }
    if (seq.qn_pin >= 0) {
        const float qn_duty = 1.0f - out_duty;
        seq_pin_density[seq.qn_pin] = out_density;
        seq_pin_duty[seq.qn_pin] = qn_duty;
        seq_pin_valid[seq.qn_pin] = 1;
        power_activate_pin(seq.qn_pin, pin_power_level, active_level, num_power_levels, active);
    }
}

__global__ void power_seed_seq_ordered_kernel(const GpuPowerSeqHost* seqs,
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
                                              float* seq_pin_density,
                                              float* seq_pin_duty,
                                              uint8_t* seq_pin_valid,
                                              int* pending_seq,
                                              int* pending_seq_count,
                                              const int* pin_power_level,
                                              uint8_t* active_level,
                                              int num_power_levels,
                                              int* active) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int seq_id = 0; seq_id < num_seqs; ++seq_id) {
        if (pending_seq[seq_id] == 0) continue;
        pending_seq[seq_id] = 0;
        const auto seq = seqs[seq_id];
        float in_density = 0.0f, in_duty = 0.0f;
        float clk_density = clock_density, clk_duty = 0.5f;
        if ((g_power_require_known_seq_data
             && !power_expr_has_known_activity_input(seq.data_expr_id, expr_ops, expr_start, expr_count, origin))
            || !power_eval_expr_activity(seq.data_expr_id, expr_ops, expr_start, expr_count,
                                      density, duty, in_density, in_duty))
            continue;
        power_eval_expr_activity(seq.clk_expr_id, expr_ops, expr_start, expr_count,
                                 density, duty, clk_density, clk_duty);
        float out_density = in_density;
        float out_duty = in_duty;
        if (power_seq_density_exceeds_clock_limit(in_density, clk_density)) {
            out_density = seq.is_latch ? in_density * clk_duty
                                       : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
        }
        if (seq.q_pin >= 0) {
            if (g_power_direct_ordered_seq_seed) {
                power_set_activity(seq.q_pin, out_density, out_duty, 3, false,
                                   pinSlew, time_unit, density, duty, origin);
            }
            seq_pin_density[seq.q_pin] = out_density;
            seq_pin_duty[seq.q_pin] = out_duty;
            seq_pin_valid[seq.q_pin] = 1;
            power_activate_pin(seq.q_pin, pin_power_level, active_level, num_power_levels, active);
        }
        if (seq.qn_pin >= 0) {
            const float qn_duty = 1.0f - out_duty;
            if (g_power_direct_ordered_seq_seed) {
                power_set_activity(seq.qn_pin, out_density, qn_duty, 3, false,
                                   pinSlew, time_unit, density, duty, origin);
            }
            seq_pin_density[seq.qn_pin] = out_density;
            seq_pin_duty[seq.qn_pin] = qn_duty;
            seq_pin_valid[seq.qn_pin] = 1;
            power_activate_pin(seq.qn_pin, pin_power_level, active_level, num_power_levels, active);
        }
    }
    if (pending_seq_count) *pending_seq_count = 0;
}

__global__ void power_seed_seq_id_list_ordered_kernel(const GpuPowerSeqHost* seqs,
                                                      const int* seq_ids,
                                                      int num_seq_ids,
                                                      const GpuPowerExprOpHost* expr_ops,
                                                      const int* expr_start,
                                                      const int* expr_count,
                                                      float clock_density,
                                                      const float* pinSlew,
                                                      float time_unit,
                                                      float* density,
                                                      float* duty,
                                                      int* origin,
                                                      float* seq_pin_density,
                                                      float* seq_pin_duty,
                                                      uint8_t* seq_pin_valid,
                                                      int* pending_seq,
                                                      int* pending_seq_count,
                                                      const int* pin_power_level,
                                                      uint8_t* active_level,
                                                      int num_power_levels,
                                                      int* active) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !seq_ids) return;
    for (int idx = 0; idx < num_seq_ids; ++idx) {
        const int seq_id = seq_ids[idx];
        if (seq_id < 0) continue;
        if (pending_seq && pending_seq[seq_id] == 0) continue;
        if (pending_seq) pending_seq[seq_id] = 0;
        const auto seq = seqs[seq_id];
        float in_density = 0.0f, in_duty = 0.0f;
        float clk_density = clock_density, clk_duty = 0.5f;
        if ((g_power_require_known_seq_data
             && !power_expr_has_known_activity_input(seq.data_expr_id, expr_ops, expr_start, expr_count, origin))
            || !power_eval_expr_activity(seq.data_expr_id, expr_ops, expr_start, expr_count,
                                      density, duty, in_density, in_duty))
            continue;
        power_eval_expr_activity(seq.clk_expr_id, expr_ops, expr_start, expr_count,
                                 density, duty, clk_density, clk_duty);
        float out_density = in_density;
        float out_duty = in_duty;
        if (power_seq_density_exceeds_clock_limit(in_density, clk_density)) {
            out_density = seq.is_latch ? in_density * clk_duty
                                       : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
        }
        if (seq.q_pin >= 0) {
            if (g_power_direct_ordered_seq_seed) {
                power_set_activity(seq.q_pin, out_density, out_duty, 3, false,
                                   pinSlew, time_unit, density, duty, origin);
            }
            seq_pin_density[seq.q_pin] = out_density;
            seq_pin_duty[seq.q_pin] = out_duty;
            seq_pin_valid[seq.q_pin] = 1;
            power_activate_pin(seq.q_pin, pin_power_level, active_level, num_power_levels, active);
        }
        if (seq.qn_pin >= 0) {
            const float qn_duty = 1.0f - out_duty;
            if (g_power_direct_ordered_seq_seed) {
                power_set_activity(seq.qn_pin, out_density, qn_duty, 3, false,
                                   pinSlew, time_unit, density, duty, origin);
            }
            seq_pin_density[seq.qn_pin] = out_density;
            seq_pin_duty[seq.qn_pin] = qn_duty;
            seq_pin_valid[seq.qn_pin] = 1;
            power_activate_pin(seq.qn_pin, pin_power_level, active_level, num_power_levels, active);
        }
    }
    if (pending_seq_count) *pending_seq_count = 0;
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
                                           const float* seq_pin_density,
                                           const float* seq_pin_duty,
                                           const uint8_t* seq_pin_valid,
                                           const int* pin_seq_list_start,
                                           const int* pin_seq_list,
                                           int* pending_seq,
                                           int* pending_seq_count,
                                           int* pending_seq_list,
                                           int* pending_seq_list_count,
                                           int pending_seq_list_cap) {
	    bool changed = false;
	    if (case_values && case_values[pin] >= 0) {
	        changed = power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true,
	                                     pinSlew, time_unit, density, duty, origin);
	        return changed;
	    }
    if (is_load_pin[pin]) {
        const int net = pin2net_map[pin];
        const int driver = (net >= 0 && net_driver_pin) ? net_driver_pin[net] : -1;
	        if (driver >= 0 && driver != pin && (!origin || origin[driver] != 0)) {
	            changed = power_set_activity(pin, density[driver], duty[driver], 3, false,
	                                         pinSlew, time_unit, density, duty, origin);
	        }
	    }
    if (is_driver_pin[pin]) {
        if (seq_pin_valid && seq_pin_valid[pin]) {
            changed = power_set_activity(pin, seq_pin_density[pin], seq_pin_duty[pin], 3, false,
                                         pinSlew, time_unit, density, duty, origin) || changed;
        } else {
            const int expr_id = pin_func_expr_id[pin];
            if (expr_id >= 0) {
                float out_density = 0.0f, out_duty = 0.0f;
                if (power_eval_expr_activity(expr_id, expr_ops, expr_start, expr_count,
                                             density, duty, out_density, out_duty)) {
                    changed = power_set_activity(pin, out_density, out_duty, 3, false,
                                                 pinSlew, time_unit, density, duty, origin) || changed;
                }
            }
        }
        changed = power_set_clock_gate_output(pin, clock_gate_clock_for_out, clock_gate_enable_for_out,
                                              pinSlew, time_unit, density, duty, origin) || changed;
    }
    if (changed && is_load_pin[pin] && power_should_mark_pending_seq(density[pin])) {
        for (int i = pin_seq_list_start[pin]; i < pin_seq_list_start[pin + 1]; i++) {
            const int seq_id = pin_seq_list[i];
            if (seq_id >= 0 && atomicExch(&pending_seq[seq_id], 1) == 0) {
                atomicAdd(pending_seq_count, 1);
                if (pending_seq_list && pending_seq_list_count && pending_seq_list_cap > 0) {
                    const int pos = atomicAdd(pending_seq_list_count, 1);
                    if (pos < pending_seq_list_cap) pending_seq_list[pos] = seq_id;
                }
            }
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
                                                   const uint8_t* is_load_pin,
                                                   const int* pin2net_map,
                                                   const int* net_driver_pin,
                                                   const int* flat_net2pin_start_map,
                                                   const int* flat_net2pin_map,
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
    if (is_load_pin && pin2net_map && net_driver_pin && flat_net2pin_start_map && flat_net2pin_map) {
        const int net = pin2net_map[pin];
        if (net >= 0 && net_driver_pin[net] == pin) {
            const int start = flat_net2pin_start_map[net];
            const int end = flat_net2pin_start_map[net + 1];
            for (int pos = start; pos < end; ++pos) {
                const int sink = flat_net2pin_map[pos];
                if (sink < 0 || sink == pin || !is_load_pin[sink]) continue;
                power_enqueue_pin_level_queue(sink, pin_power_level, level_offsets, num_power_levels,
                                              level_queue, level_counts, queued, overflow);
            }
        }
    }
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
                                                   const uint8_t* is_load_pin,
                                                   const int* pin2net_map,
                                                   const int* net_driver_pin,
                                                   const int* flat_net2pin_start_map,
                                                   const int* flat_net2pin_map,
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
        power_enqueue_adjacent_level_queue(pin, is_load_pin, pin2net_map, net_driver_pin,
                                           flat_net2pin_start_map, flat_net2pin_map,
                                           pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
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
                                                 const uint8_t* is_load_pin,
                                                 const int* pin2net_map,
                                                 const int* net_driver_pin,
                                                 const int* flat_net2pin_start_map,
                                                 const int* flat_net2pin_map,
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
        power_enqueue_adjacent_level_queue(pin, is_load_pin, pin2net_map, net_driver_pin,
                                           flat_net2pin_start_map, flat_net2pin_map,
                                           pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                           arc_types, arc_id2test_id, is_seq_output_pin,
                                           pin_power_level, level_offsets, num_power_levels,
                                           level_queue, level_counts, queued, overflow);
    }
}

__global__ void power_seed_clock_level_queue_kernel(const int* clock_pins,
                                                    int num_clock_pins,
                                                    float clock_density,
                                                    const float* clock_pin_densities,
                                                    const float* clock_pin_duties,
                                                    const uint8_t* clock_pin_enqueue,
                                                    const float* pinSlew,
                                                    float time_unit,
                                                    float* density,
                                                    float* duty,
                                                    int* origin,
                                                    const uint8_t* is_load_pin,
                                                    const int* pin2net_map,
                                                    const int* net_driver_pin,
                                                    const int* flat_net2pin_start_map,
                                                    const int* flat_net2pin_map,
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
    const float pin_density = clock_pin_densities ? clock_pin_densities[idx] : clock_density;
    const float pin_duty = clock_pin_duties ? clock_pin_duties[idx] : 0.5f;
    const bool enqueue = !clock_pin_enqueue || clock_pin_enqueue[idx] != 0;
    if (power_set_activity(pin, pin_density, pin_duty, 2, true, pinSlew, time_unit, density, duty, origin)
        && enqueue) {
        power_enqueue_adjacent_level_queue(pin, is_load_pin, pin2net_map, net_driver_pin,
                                           flat_net2pin_start_map, flat_net2pin_map,
                                           pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
                                           arc_types, arc_id2test_id, is_seq_output_pin,
                                           pin_power_level, level_offsets, num_power_levels,
                                           level_queue, level_counts, queued, overflow);
    }
}

__global__ void power_seed_roots_level_queue_ordered_kernel(
    int n,
    const int* case_values,
    const int* primary_inputs,
    int num_primary_inputs,
    float default_density,
    const int* clock_pins,
    int num_clock_pins,
    float clock_density,
    const float* clock_pin_densities,
    const float* clock_pin_duties,
    const uint8_t* clock_pin_enqueue,
    const float* pinSlew,
    float time_unit,
    float* density,
    float* duty,
    int* origin,
    const uint8_t* is_load_pin,
    const int* pin2net_map,
    const int* net_driver_pin,
    const int* flat_net2pin_start_map,
    const int* flat_net2pin_map,
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
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (case_values) {
        for (int pin = 0; pin < n; ++pin) {
            if (case_values[pin] < 0) continue;
            if (power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true,
                                   pinSlew, time_unit, density, duty, origin)) {
                power_enqueue_adjacent_level_queue(pin, is_load_pin, pin2net_map, net_driver_pin,
                                                   flat_net2pin_start_map, flat_net2pin_map,
                                                   pin_forward_arc_list_end, pin_forward_arc_list,
                                                   timing_arc_to_pin_id, arc_types, arc_id2test_id,
                                                   is_seq_output_pin, pin_power_level, level_offsets,
                                                   num_power_levels, level_queue, level_counts,
                                                   queued, overflow);
            }
        }
    }
    for (int idx = 0; idx < num_primary_inputs; ++idx) {
        const int pin = primary_inputs ? primary_inputs[idx] : -1;
        if (pin < 0 || pin >= n) continue;
        if (power_set_activity(pin, default_density, 0.5f, 1, false,
                               pinSlew, time_unit, density, duty, origin)) {
            power_enqueue_adjacent_level_queue(pin, is_load_pin, pin2net_map, net_driver_pin,
                                               flat_net2pin_start_map, flat_net2pin_map,
                                               pin_forward_arc_list_end, pin_forward_arc_list,
                                               timing_arc_to_pin_id, arc_types, arc_id2test_id,
                                               is_seq_output_pin, pin_power_level, level_offsets,
                                               num_power_levels, level_queue, level_counts,
                                               queued, overflow);
        }
    }
    for (int idx = 0; idx < num_clock_pins; ++idx) {
        const int pin = clock_pins ? clock_pins[idx] : -1;
        if (pin < 0 || pin >= n) continue;
        const float pin_density = clock_pin_densities ? clock_pin_densities[idx] : clock_density;
        const float pin_duty = clock_pin_duties ? clock_pin_duties[idx] : 0.5f;
        const bool enqueue = !clock_pin_enqueue || clock_pin_enqueue[idx] != 0;
        if (power_set_activity(pin, pin_density, pin_duty, 2, true,
                               pinSlew, time_unit, density, duty, origin) && enqueue) {
            power_enqueue_adjacent_level_queue(pin, is_load_pin, pin2net_map, net_driver_pin,
                                               flat_net2pin_start_map, flat_net2pin_map,
                                               pin_forward_arc_list_end, pin_forward_arc_list,
                                               timing_arc_to_pin_id, arc_types, arc_id2test_id,
                                               is_seq_output_pin, pin_power_level, level_offsets,
                                               num_power_levels, level_queue, level_counts,
                                               queued, overflow);
        }
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
                                                             float* seq_pin_density,
                                                             float* seq_pin_duty,
                                                             uint8_t* seq_pin_valid,
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
                    pinSlew, time_unit, density, duty, origin,
                    seq_pin_density, seq_pin_duty, seq_pin_valid,
                    pin_seq_list_start, pin_seq_list, pending_seq, pending_seq_count,
                    nullptr, nullptr, 0);
                if (changed) {
                    if (is_load_pin[pin]) {
                        power_enqueue_clock_gate_output_level_queue(pin, clock_gate_out_for_input,
                                                                    pin_power_level, level_offsets, num_power_levels,
                                                                    level_queue, level_counts, queued, overflow);
                    }
                    power_enqueue_adjacent_level_queue(pin, is_load_pin, pin2net_map, net_driver_pin,
                                                       flat_net2pin_start_map, flat_net2pin_map,
                                                       pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id,
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
            if ((g_power_require_known_seq_data
                 && !power_expr_has_known_activity_input(seq.data_expr_id, expr_ops, expr_start, expr_count, origin))
                || !power_eval_expr_activity(seq.data_expr_id, expr_ops, expr_start, expr_count,
                                          density, duty, in_density, in_duty)) continue;
            power_eval_expr_activity(seq.clk_expr_id, expr_ops, expr_start, expr_count,
                                     density, duty, clk_density, clk_duty);
            float out_density = in_density;
            float out_duty = in_duty;
            if (power_seq_density_exceeds_clock_limit(in_density, clk_density)) {
                out_density = seq.is_latch ? in_density * clk_duty
                                           : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
            }
            if (seq.q_pin >= 0) {
                seq_pin_density[seq.q_pin] = out_density;
                seq_pin_duty[seq.q_pin] = out_duty;
                seq_pin_valid[seq.q_pin] = 1;
                power_enqueue_pin_level_queue(seq.q_pin, pin_power_level, level_offsets, num_power_levels,
                                              level_queue, level_counts, queued, overflow);
            }
            if (seq.qn_pin >= 0) {
                const float qn_duty = 1.0f - out_duty;
                seq_pin_density[seq.qn_pin] = out_density;
                seq_pin_duty[seq.qn_pin] = qn_duty;
                seq_pin_valid[seq.qn_pin] = 1;
                power_enqueue_pin_level_queue(seq.qn_pin, pin_power_level, level_offsets, num_power_levels,
                                              level_queue, level_counts, queued, overflow);
            }
        }
    }
}

__global__ void power_activity_level_queue_ordered_kernel(int n,
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
                                                          float* seq_pin_density,
                                                          float* seq_pin_duty,
                                                          uint8_t* seq_pin_valid,
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
                                                          int* pending_seq_list,
                                                          int* pending_seq_list_count,
                                                          int* level_queue,
                                                          int* level_counts,
                                                          int* queued,
                                                          int max_seq_passes,
                                                          int* overflow) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    auto drain = [&]() {
        bool any = true;
        while (any) {
            any = false;
            for (int level = 0; level < num_power_levels; ++level) {
                const int offset = level_offsets[level];
                while (level_counts[level] > 0) {
                    any = true;
                    const int idx = --level_counts[level];
                    const int pin = level_queue[offset + idx];
                    if (pin < 0 || pin >= n) continue;
                    queued[pin] = 0;
                    const bool changed = power_process_pin_frontier(
                        pin, case_values, is_load_pin, is_driver_pin,
                        pin2net_map, net_driver_pin, flat_net2pin_start_map, flat_net2pin_map,
                        pin_func_expr_id, clock_gate_clock_for_out, clock_gate_enable_for_out,
                        expr_ops, expr_start, expr_count, clock_density,
                        pinSlew, time_unit, density, duty, origin,
                        seq_pin_density, seq_pin_duty, seq_pin_valid,
                        pin_seq_list_start, pin_seq_list, pending_seq, pending_seq_count,
                        pending_seq_list, pending_seq_list_count, num_seqs);
                    if (!changed) continue;
                    if (is_load_pin[pin]) {
                        power_enqueue_clock_gate_output_level_queue(pin, clock_gate_out_for_input,
                                                                    pin_power_level, level_offsets,
                                                                    num_power_levels, level_queue,
                                                                    level_counts, queued, overflow);
                    }
                    power_enqueue_adjacent_level_queue(pin, is_load_pin, pin2net_map, net_driver_pin,
                                                       flat_net2pin_start_map, flat_net2pin_map,
                                                       pin_forward_arc_list_end, pin_forward_arc_list,
                                                       timing_arc_to_pin_id, arc_types, arc_id2test_id,
                                                       is_seq_output_pin, pin_power_level, level_offsets,
                                                       num_power_levels, level_queue, level_counts,
                                                       queued, overflow);
                }
            }
        }
    };

    drain();
    for (int pass = 1; pass < max_seq_passes; ++pass) {
        if (*pending_seq_count <= 0) break;
        const int pending_items = pending_seq_list_count ? *pending_seq_list_count : 0;
        const int seq_scan_count = pending_items > 0 ? pending_items : num_seqs;
        for (int idx = 0; idx < seq_scan_count; ++idx) {
            const int seq_id = pending_items > 0 ? pending_seq_list[idx] : idx;
            if (seq_id < 0 || seq_id >= num_seqs) continue;
            if (pending_seq[seq_id] == 0) continue;
            pending_seq[seq_id] = 0;
            *pending_seq_count -= 1;
            const auto seq = seqs[seq_id];
            float in_density = 0.0f, in_duty = 0.0f;
            float clk_density = clock_density, clk_duty = 0.5f;
            if ((g_power_require_known_seq_data
                 && !power_expr_has_known_activity_input(seq.data_expr_id, expr_ops, expr_start, expr_count, origin))
                || !power_eval_expr_activity(seq.data_expr_id, expr_ops, expr_start, expr_count,
                                          density, duty, in_density, in_duty)) {
                continue;
            }
            power_eval_expr_activity(seq.clk_expr_id, expr_ops, expr_start, expr_count,
                                     density, duty, clk_density, clk_duty);
            float out_density = in_density;
            float out_duty = in_duty;
            if (power_seq_density_exceeds_clock_limit(in_density, clk_density)) {
                out_density = seq.is_latch ? in_density * clk_duty
                                           : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
            }
            if (seq.q_pin >= 0) {
                seq_pin_density[seq.q_pin] = out_density;
                seq_pin_duty[seq.q_pin] = out_duty;
                seq_pin_valid[seq.q_pin] = 1;
                power_enqueue_pin_level_queue(seq.q_pin, pin_power_level, level_offsets,
                                              num_power_levels, level_queue, level_counts,
                                              queued, overflow);
            }
            if (seq.qn_pin >= 0) {
                const float qn_duty = 1.0f - out_duty;
                seq_pin_density[seq.qn_pin] = out_density;
                seq_pin_duty[seq.qn_pin] = qn_duty;
                seq_pin_valid[seq.qn_pin] = 1;
                power_enqueue_pin_level_queue(seq.qn_pin, pin_power_level, level_offsets,
                                              num_power_levels, level_queue, level_counts,
                                              queued, overflow);
            }
        }
        if (pending_seq_list_count) *pending_seq_list_count = 0;
        drain();
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

__global__ void power_unpack_precomputed_activity_kernel(int n,
                                                         const float* activity,
                                                         float* density,
                                                         float* duty,
                                                         int* origin) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    density[idx] = activity[idx * 3 + 0];
    duty[idx] = activity[idx * 3 + 1];
    origin[idx] = static_cast<int>(activity[idx * 3 + 2]);
}

__global__ void power_unpack_activity_density_duty_kernel(int n,
                                                          const float* activity,
                                                          float* density,
                                                          float* duty) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    density[idx] = activity[idx * 3 + 0];
    duty[idx] = activity[idx * 3 + 1];
}

__global__ void power_switching_kernel(int n,
                                       int num_nodes,
                                       const uint8_t* is_driver_pin,
                                       const uint8_t* is_cell_pin,
                                       const int* pin2node_map,
                                       const float* pinLoad,
                                       const float* dmp_C1,
                                       const float* dmp_C2,
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
        if (inst_switching && node >= 0 && node < num_nodes && sw != 0.0f) atomicAdd(&inst_switching[node], sw);
    }
    if (pin_switching) pin_switching[pin] = sw;
}

__device__ float power_expr_duty_cuda(int expr_id,
                                      const GpuPowerExprOpHost* ops,
                                      const int* expr_start,
                                      const int* expr_count,
                                      const float* pin_density,
                                      const float* pin_duty,
                                      const int* node_port_pin_start = nullptr,
                                      const int* node_port_pin_list = nullptr,
                                      int node_id = -1) {
    float d = 0.0f, u = 0.0f;
    if (!power_eval_expr_activity(expr_id, ops, expr_start, expr_count, pin_density, pin_duty, d, u,
                                  node_port_pin_start, node_port_pin_list, node_id)) return 0.0f;
    return fminf(fmaxf(u, 0.0f), 1.0f);
}

__device__ float power_expr_diff_duty_cuda(int expr_id,
                                           int diff_pin,
                                           const GpuPowerExprOpHost* ops,
                                           const int* expr_start,
                                           const int* expr_count,
                                           const float* pin_duty) {
    if (expr_id < 0 || diff_pin < 0) return 0.0f;
    PowerBddContextCuda ctx;
    int root = 1;
    if (!power_bdd_build_expr(expr_id, ops, expr_start, expr_count,
                              nullptr, pin_duty, ctx, root)) {
        return 0.0f;
    }
    int diff_var = -1;
    for (int var = 0; var < ctx.var_count; var++) {
        if (ctx.var_pins[var] == diff_pin) {
            diff_var = var;
            break;
        }
    }
    if (diff_var < 0) return 0.0f;
    const int low = power_bdd_restrict(ctx, root, diff_var, false);
    const int high = power_bdd_restrict(ctx, root, diff_var, true);
    const int diff = power_bdd_apply(ctx, 2, low, high);
    return power_bdd_eval_duty(ctx, diff);
}

__device__ float power_internal_row_duty(const GpuPowerInternalHost& row,
                                         const GpuPowerExprOpHost* expr_ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         const float* density,
                                         const float* duty,
                                         const int* node_port_pin_start,
                                         const int* node_port_pin_list) {
    if (row.duty_mode == 0) return 1.0f;
    if (row.duty_mode == 1) return power_expr_duty_cuda(row.duty_expr_id, expr_ops, expr_start, expr_count,
                                                        density, duty, node_port_pin_start,
                                                        node_port_pin_list, row.node_id);
    if (row.duty_mode == 2) return power_expr_diff_duty_cuda(row.duty_expr_id, row.duty_pin, expr_ops, expr_start, expr_count, duty);
    if (row.duty_mode == 3) return 0.5f;
    return 0.0f;
}

__global__ void power_internal_denom_kernel(const GpuPowerInternalHost* rows,
                                            int num_rows,
                                            const GpuPowerExprOpHost* expr_ops,
                                            const int* expr_start,
                                            const int* expr_count,
                                            const int* node_port_pin_start,
                                            const int* node_port_pin_list,
                                            const float* density,
                                            const float* duty,
                                            float* denom) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    const auto row = rows[idx];
    if (row.kind != 1 || row.denom_group < 0 || row.from_pin < 0) return;
    const float d = power_internal_row_duty(row, expr_ops, expr_start, expr_count, density, duty,
                                            node_port_pin_start, node_port_pin_list);
    const float numer = density[row.from_pin] * d;
    if (isfinite(numer) && numer != 0.0f) atomicAdd(&denom[row.denom_group], numer);
}

__global__ void power_internal_contrib_kernel(const GpuPowerInternalHost* rows,
                                              int num_rows,
                                              int num_nodes,
                                              const GpuPowerExprOpHost* expr_ops,
                                              const int* expr_start,
                                              const int* expr_count,
                                              const int* node_port_pin_start,
                                              const int* node_port_pin_list,
                                              const float* density,
                                              const float* duty,
                                              const float* pinSlew,
                                              const float* power_clock_slews,
                                              const float* dmp_C1,
                                              const float* dmp_C2,
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
    const float row_duty = power_internal_row_duty(row, expr_ops, expr_start, expr_count, density, duty,
                                                   node_port_pin_start, node_port_pin_list);
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
                                         const int* node_port_pin_start,
                                         const int* node_port_pin_list,
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
        cond_duty = power_expr_duty_cuda(row.when_expr_id, expr_ops, expr_start, expr_count,
                                         density, duty, node_port_pin_start,
                                         node_port_pin_list, row.node_id);
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

void check_power_cuda_error(const char* label) {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::string where = label ? label : "unknown";
        cudaGetLastError();
        throw std::runtime_error("[power] CUDA error at " + where + ": " +
                                 cudaGetErrorString(sync_err));
    }
    cudaError_t last_err = cudaGetLastError();
    if (last_err != cudaSuccess) {
        std::string where = label ? label : "unknown";
        throw std::runtime_error("[power] CUDA error at " + where + ": " +
                                 cudaGetErrorString(last_err));
    }
}

static void power_cuda_call(cudaError_t err, const char* label) {
    if (err == cudaSuccess) return;
    std::string where = label ? label : "unknown";
    throw std::runtime_error("[power] CUDA call failed at " + where + ": " +
                             cudaGetErrorString(err));
}

static void unpack_power_activity_density_duty(int n,
                                               const float* d_precomputed_activity,
                                               float** d_density,
                                               float** d_duty) {
    *d_density = nullptr;
    *d_duty = nullptr;
    if (n <= 0 || !d_precomputed_activity) return;
    cudaMalloc(d_density, sizeof(float) * n);
    cudaMalloc(d_duty, sizeof(float) * n);
    power_unpack_activity_density_duty_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
        n, d_precomputed_activity, *d_density, *d_duty);
    cudaDeviceSynchronize();
}

void run_power_internal_denom_chunk_cuda_launcher(int n,
                                                  const float* d_precomputed_activity,
                                                  GpuPowerInternalHost* d_internal_rows,
                                                  int num_internal_rows,
                                                  GpuPowerExprOpHost* d_expr_ops,
                                                  int* d_expr_start,
                                                  int* d_expr_count,
                                                  int* d_node_port_pin_start,
                                                  int* d_node_port_pin_list,
                                                  float* d_denom) {
    if (n <= 0 || !d_precomputed_activity || !d_internal_rows || num_internal_rows <= 0 || !d_denom)
        return;
    float* d_density = nullptr;
    float* d_duty = nullptr;
    unpack_power_activity_density_duty(n, d_precomputed_activity, &d_density, &d_duty);
    if (d_density && d_duty) {
        power_internal_denom_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_internal_rows, num_internal_rows, d_expr_ops, d_expr_start, d_expr_count,
            d_node_port_pin_start, d_node_port_pin_list, d_density, d_duty, d_denom);
        cudaDeviceSynchronize();
    }
    cudaFree(d_density);
    cudaFree(d_duty);
}

void run_power_internal_contrib_chunk_cuda_launcher(int n,
                                                    int num_nodes,
                                                    const float* d_precomputed_activity,
                                                    GpuPowerInternalHost* d_internal_rows,
                                                    int num_internal_rows,
                                                    GpuPowerExprOpHost* d_expr_ops,
                                                    int* d_expr_start,
                                                    int* d_expr_count,
                                                    int* d_node_port_pin_start,
                                                    int* d_node_port_pin_list,
                                                    const float* d_pinSlew,
                                                    const float* d_power_clock_slews,
                                                    const float* d_dmp_C1,
                                                    const float* d_dmp_C2,
                                                    const float* d_denom,
                                                    GPUPowerLutAllocator* d_power_allocator,
                                                    float cap_unit,
                                                    float* d_inst_internal,
                                                    float* d_internal_row_power) {
    if (n <= 0 || !d_precomputed_activity || !d_internal_rows || num_internal_rows <= 0 ||
        !d_denom || !d_power_allocator)
        return;
    float* d_density = nullptr;
    float* d_duty = nullptr;
    unpack_power_activity_density_duty(n, d_precomputed_activity, &d_density, &d_duty);
    if (d_density && d_duty) {
        power_internal_contrib_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_internal_rows, num_internal_rows, num_nodes, d_expr_ops, d_expr_start, d_expr_count,
            d_node_port_pin_start, d_node_port_pin_list, d_density, d_duty, d_pinSlew, d_power_clock_slews, d_dmp_C1, d_dmp_C2,
            d_denom, d_power_allocator, cap_unit, d_inst_internal, d_internal_row_power);
        cudaDeviceSynchronize();
    }
    cudaFree(d_density);
    cudaFree(d_duty);
}

void run_power_leakage_rows_chunk_cuda_launcher(int n,
                                                const float* d_precomputed_activity,
                                                GpuPowerLeakageRowHost* d_leakage_rows,
                                                int num_leakage_rows,
                                                GpuPowerExprOpHost* d_expr_ops,
                                                int* d_expr_start,
                                                int* d_expr_count,
                                                int* d_node_port_pin_start,
                                                int* d_node_port_pin_list,
                                                float* d_group_cond_leakage,
                                                float* d_group_cond_duty_sum,
                                                int* d_group_cond_count,
                                                float* d_leakage_row_power) {
    if (n <= 0 || !d_precomputed_activity || !d_leakage_rows || num_leakage_rows <= 0 ||
        !d_group_cond_leakage || !d_group_cond_duty_sum || !d_group_cond_count)
        return;
    float* d_density = nullptr;
    float* d_duty = nullptr;
    unpack_power_activity_density_duty(n, d_precomputed_activity, &d_density, &d_duty);
    if (d_density && d_duty) {
        power_leakage_row_kernel<<<BLOCK_NUMBER(num_leakage_rows), BLOCK_SIZE>>>(
            d_leakage_rows, num_leakage_rows, d_expr_ops, d_expr_start, d_expr_count,
            d_node_port_pin_start, d_node_port_pin_list, d_density, d_duty, d_group_cond_leakage, d_group_cond_duty_sum,
            d_group_cond_count, d_leakage_row_power);
        cudaDeviceSynchronize();
    }
    cudaFree(d_density);
    cudaFree(d_duty);
}

void run_power_leakage_summary_chunk_cuda_launcher(GpuPowerLeakageGroupHost* d_leakage_groups,
                                                   int num_leakage_groups,
                                                   float* d_group_cond_leakage,
                                                   float* d_group_cond_duty_sum,
                                                   int* d_group_cond_count,
                                                   int num_nodes,
                                                   float* d_inst_leakage) {
    if (!d_leakage_groups || num_leakage_groups <= 0 || !d_group_cond_leakage ||
        !d_group_cond_duty_sum || !d_group_cond_count || !d_inst_leakage)
        return;
    power_leakage_summary_kernel<<<BLOCK_NUMBER(num_leakage_groups), BLOCK_SIZE>>>(
        d_leakage_groups, num_leakage_groups, d_group_cond_leakage,
        d_group_cond_duty_sum, d_group_cond_count, num_nodes, d_inst_leakage);
    check_power_cuda_error("leakage summary chunk");
}

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
                                      const float* d_clock_pin_densities,
                                      const float* d_clock_pin_duties,
                                      const uint8_t* d_clock_pin_enqueue,
                                      GpuPowerExprOpHost* d_expr_ops,
                                      int* d_expr_start,
                                      int* d_expr_count,
                                      int* d_node_port_pin_start,
                                      int* d_node_port_pin_list,
                                      int* d_pin_func_expr_id,
                                      int* d_missing_func_out_start,
                                      int* d_missing_func_out_list,
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
                                      int* d_trace_pins,
                                      int num_trace_pins,
                                      const float* d_precomputed_activity,
                                      float* d_out,
                                      int num_nodes,
                                      const int* d_pin2node_map,
                                      const float* d_pinLoad,
                                      const float* d_dmp_C1,
                                      const float* d_dmp_C2,
                                      const float* d_pinSlew,
                                      const float* d_power_clock_slews,
                                      bool allow_clock_activity_override,
                                      float min_activity_density,
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
    float* d_prev_density = nullptr;
    float* d_prev_duty = nullptr;
    float* d_seq_pin_density = nullptr;
    float* d_seq_pin_duty = nullptr;
    int* d_origin = nullptr;
    int* d_prev_origin = nullptr;
    int* d_active = nullptr;
    uint8_t* d_active_level = nullptr;
    uint8_t* d_visit_active = nullptr;
    uint8_t* d_seq_pin_valid = nullptr;
    int* d_pending_seq = nullptr;
    int* d_pending_seq_count = nullptr;
    const int num_power_levels = std::max(0, static_cast<int>(level_list_end_cpu.size()) - 1);
    power_cuda_call(cudaMalloc(&d_density, sizeof(float) * n), "activity malloc density");
    power_cuda_call(cudaMalloc(&d_duty, sizeof(float) * n), "activity malloc duty");
    power_cuda_call(cudaMalloc(&d_prev_density, sizeof(float) * n), "activity malloc prev_density");
    power_cuda_call(cudaMalloc(&d_prev_duty, sizeof(float) * n), "activity malloc prev_duty");
    power_cuda_call(cudaMalloc(&d_seq_pin_density, sizeof(float) * n), "activity malloc seq_pin_density");
    power_cuda_call(cudaMalloc(&d_seq_pin_duty, sizeof(float) * n), "activity malloc seq_pin_duty");
    power_cuda_call(cudaMalloc(&d_origin, sizeof(int) * n), "activity malloc origin");
    power_cuda_call(cudaMalloc(&d_prev_origin, sizeof(int) * n), "activity malloc prev_origin");
    power_cuda_call(cudaMalloc(&d_active, sizeof(int) * n), "activity malloc active");
    power_cuda_call(cudaMalloc(&d_active_level, sizeof(uint8_t) * std::max(1, num_power_levels)), "activity malloc active_level");
    power_cuda_call(cudaMalloc(&d_visit_active, sizeof(uint8_t) * n), "activity malloc visit_active");
    power_cuda_call(cudaMalloc(&d_seq_pin_valid, sizeof(uint8_t) * n), "activity malloc seq_pin_valid");
    power_cuda_call(cudaMalloc(&d_pending_seq, sizeof(int) * std::max(1, num_seqs)), "activity malloc pending_seq");
    power_cuda_call(cudaMalloc(&d_pending_seq_count, sizeof(int)), "activity malloc pending_seq_count");
    power_cuda_call(cudaMemset(d_density, 0, sizeof(float) * n), "activity memset density");
    power_cuda_call(cudaMemset(d_duty, 0, sizeof(float) * n), "activity memset duty");
    power_cuda_call(cudaMemset(d_prev_density, 0, sizeof(float) * n), "activity memset prev_density");
    power_cuda_call(cudaMemset(d_prev_duty, 0, sizeof(float) * n), "activity memset prev_duty");
    power_cuda_call(cudaMemset(d_seq_pin_density, 0, sizeof(float) * n), "activity memset seq_pin_density");
    power_cuda_call(cudaMemset(d_seq_pin_duty, 0, sizeof(float) * n), "activity memset seq_pin_duty");
    power_cuda_call(cudaMemset(d_origin, 0, sizeof(int) * n), "activity memset origin");
    power_cuda_call(cudaMemset(d_prev_origin, 0, sizeof(int) * n), "activity memset prev_origin");
    power_cuda_call(cudaMemset(d_active, 0, sizeof(int) * n), "activity memset active");
    power_cuda_call(cudaMemset(d_active_level, 0, sizeof(uint8_t) * std::max(1, num_power_levels)), "activity memset active_level");
    power_cuda_call(cudaMemset(d_visit_active, 0, sizeof(uint8_t) * n), "activity memset visit_active");
    power_cuda_call(cudaMemset(d_seq_pin_valid, 0, sizeof(uint8_t) * n), "activity memset seq_pin_valid");
    power_cuda_call(cudaMemset(d_pending_seq, 0, sizeof(int) * std::max(1, num_seqs)), "activity memset pending_seq");
    power_cuda_call(cudaMemset(d_pending_seq_count, 0, sizeof(int)), "activity memset pending_seq_count");
    size_t power_stack_size = 32768;
    if (const char* env_stack = std::getenv("XPLACE_POWER_CUDA_STACK_SIZE")) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(env_stack, &end, 10);
        if (end != env_stack && parsed > 0) power_stack_size = parsed;
    }
    power_cuda_call(cudaDeviceSetLimit(cudaLimitStackSize, power_stack_size), "activity set stack size");
    power_cuda_call(cudaMemcpyToSymbol(g_power_allow_clock_activity_override,
                                       &allow_clock_activity_override,
                                       sizeof(bool)),
                    "activity copy allow_clock_activity_override");
    power_cuda_call(cudaMemcpyToSymbol(g_power_min_activity_density,
                                       &min_activity_density,
                                       sizeof(float)),
                    "activity copy min_activity_density");
    float min_activity_duty = 0.0f;
    if (const char* env = std::getenv("XPLACE_POWER_MIN_ACTIVITY_DUTY")) {
        char* end = nullptr;
        const float value = std::strtof(env, &end);
        if (end != env && value >= 0.0f) min_activity_duty = value;
    }
    power_cuda_call(cudaMemcpyToSymbol(g_power_min_activity_duty,
                                       &min_activity_duty,
                                       sizeof(float)),
                    "activity copy min_activity_duty");
    const bool disable_activity_slew_cap =
        std::getenv("XPLACE_POWER_DISABLE_ACTIVITY_SLEW_CAP") != nullptr;
    power_cuda_call(cudaMemcpyToSymbol(g_power_disable_activity_slew_cap,
                                       &disable_activity_slew_cap,
                                       sizeof(bool)),
                    "activity copy disable_activity_slew_cap");
    float seq_clock_limit_rel_tol = 0.0f;
    if (const char* env = std::getenv("XPLACE_POWER_SEQ_CLOCK_LIMIT_REL_TOL")) {
        char* end = nullptr;
        const float value = std::strtof(env, &end);
        if (end != env && value >= 0.0f) seq_clock_limit_rel_tol = value;
    }
    power_cuda_call(cudaMemcpyToSymbol(g_power_seq_clock_limit_rel_tol,
                                       &seq_clock_limit_rel_tol,
                                       sizeof(float)),
                    "activity copy seq_clock_limit_rel_tol");
    float seq_pending_min_density = 0.0f;
    if (const char* env = std::getenv("XPLACE_POWER_SEQ_PENDING_MIN_DENSITY")) {
        char* end = nullptr;
        const float value = std::strtof(env, &end);
        if (end != env && value >= 0.0f) seq_pending_min_density = value;
    }
    power_cuda_call(cudaMemcpyToSymbol(g_power_seq_pending_min_density,
                                       &seq_pending_min_density,
                                       sizeof(float)),
                    "activity copy seq_pending_min_density");
    int seq_clock_limit_rel_tol_start_pass = 1;
    if (const char* env = std::getenv("XPLACE_POWER_SEQ_CLOCK_LIMIT_REL_TOL_START_PASS"))
        seq_clock_limit_rel_tol_start_pass = std::max(1, std::atoi(env));
    const bool clamp_activity_to_clock_density =
        std::getenv("XPLACE_POWER_CLAMP_ACTIVITY_TO_CLOCK_DENSITY") != nullptr;
    const float activity_clock_density_cap = clamp_activity_to_clock_density
        ? clock_density
        : 3.4028234663852886e38f;
    power_cuda_call(cudaMemcpyToSymbol(g_power_activity_clock_density_cap,
                                       &activity_clock_density_cap,
                                       sizeof(float)),
                    "activity copy activity_clock_density_cap");
    const int require_known_seq_data =
        read_power_bool_env_host("XPLACE_POWER_REQUIRE_KNOWN_SEQ_DATA", false) ? 1 : 0;
    power_cuda_call(cudaMemcpyToSymbol(g_power_require_known_seq_data,
                                       &require_known_seq_data,
                                       sizeof(int)),
                    "activity copy require_known_seq_data");
    std::vector<int> h_trace_pins(std::max(0, num_trace_pins));
    if (num_trace_pins > 0 && d_trace_pins) {
        cudaMemcpy(h_trace_pins.data(), d_trace_pins, sizeof(int) * num_trace_pins, cudaMemcpyDeviceToHost);
    }
    std::vector<uint8_t> h_trace_first_seen(std::max(1, num_trace_pins), 0);
    auto trace_cuda = [&](const char* tag, int pass, int pending_count) {
        for (int idx = 0; idx < num_trace_pins; ++idx) {
            const int pin = h_trace_pins[idx];
            if (pin < 0 || pin >= n) continue;
            float pin_density = 0.0f;
            float pin_duty = 0.0f;
            int pin_origin = 0;
            cudaMemcpy(&pin_density, d_density + pin, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pin_duty, d_duty + pin, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pin_origin, d_origin + pin, sizeof(int), cudaMemcpyDeviceToHost);
            int seq_count = 0;
            int seq_pending = 0;
            if (d_pin_seq_list_start && d_pin_seq_list && d_pending_seq) {
                int start = 0;
                int end = 0;
                cudaMemcpy(&start, d_pin_seq_list_start + pin, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&end, d_pin_seq_list_start + pin + 1, sizeof(int), cudaMemcpyDeviceToHost);
                seq_count = std::max(0, end - start);
                for (int pos = start; pos < end; ++pos) {
                    int seq_id = -1;
                    int pending = 0;
                    cudaMemcpy(&seq_id, d_pin_seq_list + pos, sizeof(int), cudaMemcpyDeviceToHost);
                    if (seq_id >= 0 && seq_id < num_seqs) {
                        cudaMemcpy(&pending, d_pending_seq + seq_id, sizeof(int), cudaMemcpyDeviceToHost);
                        if (pending) seq_pending++;
                    }
                }
            }
            const bool first_nonzero = pin_density > 0.0f && !h_trace_first_seen[idx];
            if (first_nonzero) h_trace_first_seen[idx] = 1;
            fprintf(stderr,
                    "[power_activity_trace_cuda] tag=%s pass=%d pending=%d pin_id=%d density=%.10e duty=%.10g origin=%d first_nonzero=%d seq_count=%d seq_pending=%d\n",
                    tag, pass, pending_count, pin, pin_density, pin_duty, pin_origin,
                    first_nonzero ? 1 : 0, seq_count, seq_pending);
        }
    };
    const char* pending_seq_dump_file = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_FILE");
    int pending_seq_dump_pass = -1;
    if (const char* env = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_PASS"))
        pending_seq_dump_pass = std::atoi(env);
    std::string pending_seq_dump_tag = "after_pass";
    if (const char* env = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_TAG"))
        pending_seq_dump_tag = env;
    auto dump_cuda_pending_seq = [&](const char* tag, int pass) {
        if (!pending_seq_dump_file || pending_seq_dump_file[0] == '\0') return;
        if (pending_seq_dump_pass >= 0 && pass != pending_seq_dump_pass) return;
        if (pending_seq_dump_tag != (tag ? tag : "")) return;
        if (!d_pending_seq || !d_seqs || num_seqs <= 0) return;
        std::vector<int> h_pending(std::max(1, num_seqs), 0);
        std::vector<GpuPowerSeqHost> h_seq_dump(std::max(1, num_seqs));
        cudaMemcpy(h_pending.data(), d_pending_seq, sizeof(int) * num_seqs, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_seq_dump.data(), d_seqs, sizeof(GpuPowerSeqHost) * num_seqs, cudaMemcpyDeviceToHost);
        std::ofstream out(pending_seq_dump_file, std::ios::app);
        if (!out) return;
        out << "engine,pass,tag,node_id,inst_name,seq_id,q_pin,qn_pin,pin_id,pin_name\n";
        for (int seq_id = 0; seq_id < num_seqs; ++seq_id) {
            if (!h_pending[seq_id]) continue;
            const auto seq = h_seq_dump[seq_id];
            const int pin_id = seq.q_pin >= 0 ? seq.q_pin : seq.qn_pin;
            out << "xplace_cuda," << pass << ',' << (tag ? tag : "")
                << ",-1,," << seq_id << ','
                << seq.q_pin << ',' << seq.qn_pin << ','
                << pin_id << ",\n";
        }
    };

    if (d_precomputed_activity) {
        power_unpack_precomputed_activity_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
            n, d_precomputed_activity, d_density, d_duty, d_origin);
        check_power_cuda_error("activity unpack_precomputed");
        trace_cuda("precomputed", 0, 0);
    } else {
    bool use_frontier = false;
    if (const char* env_frontier = std::getenv("XPLACE_POWER_ACTIVITY_FRONTIER"))
        use_frontier = std::atoi(env_frontier) != 0;
    bool use_ordered_frontier = false;
    if (const char* env_ordered = std::getenv("XPLACE_POWER_ACTIVITY_ORDERED_QUEUE"))
        use_ordered_frontier = std::atoi(env_ordered) != 0;
    if (use_ordered_frontier) use_frontier = true;
    cudaDeviceProp prop{};
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);
    if (use_frontier && !use_ordered_frontier && !prop.cooperativeLaunch) {
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
        check_power_cuda_error("activity seed_seq_feedback_state");
    }

    if (use_frontier) {
        int *d_level_offsets = nullptr, *d_level_queue = nullptr, *d_level_counts = nullptr;
        int *d_queued = nullptr, *d_overflow = nullptr;
        int *d_frontier_pending_seq_list = nullptr, *d_frontier_pending_seq_list_count = nullptr;
        std::vector<int> frontier_level_offsets = level_list_end_cpu;
        int frontier_queue_size = std::max(1, n);
        if (use_ordered_frontier) {
            int cap_mult = 4;
            if (const char* env_cap = std::getenv("XPLACE_POWER_ORDERED_QUEUE_CAP_MULT"))
                cap_mult = std::max(1, std::atoi(env_cap));
            frontier_level_offsets.assign(std::max(1, num_power_levels + 1), 0);
            for (int level = 0; level < num_power_levels; ++level) {
                const int level_count = level_list_end_cpu[level + 1] - level_list_end_cpu[level];
                frontier_level_offsets[level + 1] =
                    frontier_level_offsets[level] + std::max(1, level_count * cap_mult);
            }
            frontier_queue_size = std::max(1, frontier_level_offsets.back());
            cudaMalloc(&d_frontier_pending_seq_list, sizeof(int) * std::max(1, num_seqs));
            cudaMalloc(&d_frontier_pending_seq_list_count, sizeof(int));
            cudaMemset(d_frontier_pending_seq_list_count, 0, sizeof(int));
        }
        cudaMalloc(&d_level_offsets, sizeof(int) * std::max(1, num_power_levels + 1));
        cudaMalloc(&d_level_queue, sizeof(int) * frontier_queue_size);
        cudaMalloc(&d_level_counts, sizeof(int) * std::max(1, num_power_levels));
        cudaMalloc(&d_queued, sizeof(int) * std::max(1, n));
        cudaMalloc(&d_overflow, sizeof(int));
        if (num_power_levels + 1 > 0) {
            cudaMemcpy(d_level_offsets, frontier_level_offsets.data(), sizeof(int) * (num_power_levels + 1), cudaMemcpyHostToDevice);
        }
        cudaMemset(d_level_queue, 0, sizeof(int) * frontier_queue_size);
        cudaMemset(d_level_counts, 0, sizeof(int) * std::max(1, num_power_levels));
        cudaMemset(d_queued, 0, sizeof(int) * std::max(1, n));
        cudaMemset(d_overflow, 0, sizeof(int));

        const bool ordered_root_seed =
            std::getenv("XPLACE_POWER_ORDERED_ROOT_SEED") != nullptr;
        if (ordered_root_seed) {
            power_seed_roots_level_queue_ordered_kernel<<<1, 1>>>(
                n, d_case_values, d_primary_inputs, num_primary_inputs,
                default_density, d_clock_pins, num_clock_pins, clock_density,
                d_clock_pin_densities, d_clock_pin_duties, d_clock_pin_enqueue,
                d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_is_load_pin, d_pin2net_map, d_net_driver_pin,
                d_flat_net2pin_start_map, d_flat_net2pin_map,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_level_offsets, num_power_levels,
                d_level_queue, d_level_counts, d_queued, d_overflow);
        } else {
        if (d_case_values) {
            power_seed_case_level_queue_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                n, d_case_values, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_is_load_pin, d_pin2net_map, d_net_driver_pin,
                d_flat_net2pin_start_map, d_flat_net2pin_map,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_level_offsets, num_power_levels,
                d_level_queue, d_level_counts, d_queued, d_overflow);
        }
        if (num_primary_inputs > 0) {
            power_seed_pi_level_queue_kernel<<<BLOCK_NUMBER(num_primary_inputs), BLOCK_SIZE>>>(
                d_primary_inputs, num_primary_inputs, default_density, clock_density, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_is_load_pin, d_pin2net_map, d_net_driver_pin,
                d_flat_net2pin_start_map, d_flat_net2pin_map,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_level_offsets, num_power_levels,
                d_level_queue, d_level_counts, d_queued, d_overflow);
        }
        if (num_clock_pins > 0) {
            power_seed_clock_level_queue_kernel<<<BLOCK_NUMBER(num_clock_pins), BLOCK_SIZE>>>(
                d_clock_pins, num_clock_pins, clock_density,
                d_clock_pin_densities, d_clock_pin_duties, d_clock_pin_enqueue,
                d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_is_load_pin, d_pin2net_map, d_net_driver_pin,
                d_flat_net2pin_start_map, d_flat_net2pin_map,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_level_offsets, num_power_levels,
                d_level_queue, d_level_counts, d_queued, d_overflow);
        }
        }
        check_power_cuda_error("activity frontier seed");
        if (num_trace_pins > 0) {
            fprintf(stderr, "[power_activity_trace_cuda] frontier_trace=unsupported\n");
        }

        if (use_ordered_frontier) {
            power_activity_level_queue_ordered_kernel<<<1, 1>>>(
                n, num_power_levels, d_level_offsets, d_case_values,
                d_is_load_pin, d_is_driver_pin, d_pin2net_map, d_net_driver_pin,
                d_flat_net2pin_start_map, d_flat_net2pin_map,
                d_pin_func_expr_id, d_clock_gate_out_for_input,
                d_clock_gate_clock_for_out, d_clock_gate_enable_for_out,
                d_expr_ops, d_expr_start, d_expr_count, clock_density,
                d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_seq_pin_density, d_seq_pin_duty, d_seq_pin_valid,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list,
                d_timing_arc_to_pin_id, d_arc_types, d_arc_id2test_id,
                d_is_seq_output_pin, d_pin_seq_list_start, d_pin_seq_list,
                d_seqs, num_seqs, d_pin_power_level, d_pending_seq,
                d_pending_seq_count, d_frontier_pending_seq_list,
                d_frontier_pending_seq_list_count,
                d_level_queue, d_level_counts, d_queued,
                max_activity_passes, d_overflow);
        } else {
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
                &d_seq_pin_density,
                &d_seq_pin_duty,
                &d_seq_pin_valid,
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
        }
        check_power_cuda_error("activity frontier propagate");
        int overflow = 0;
        cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost);
        if (overflow) fprintf(stderr, "[power_frontier] level queue overflow detected; results may be incomplete\n");
        cudaFree(d_level_offsets);
        cudaFree(d_level_queue);
        cudaFree(d_level_counts);
        cudaFree(d_queued);
        cudaFree(d_overflow);
        if (d_frontier_pending_seq_list) cudaFree(d_frontier_pending_seq_list);
        if (d_frontier_pending_seq_list_count) cudaFree(d_frontier_pending_seq_list_count);
    } else {
        if (d_case_values) {
            power_seed_case_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                n, d_case_values, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_is_load_pin, d_pin2net_map, d_net_driver_pin,
                d_flat_net2pin_start_map, d_flat_net2pin_map,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_active_level, num_power_levels, d_active);
        }
        if (num_primary_inputs > 0) {
            power_seed_pi_kernel<<<BLOCK_NUMBER(num_primary_inputs), BLOCK_SIZE>>>(
                d_primary_inputs, num_primary_inputs, default_density, clock_density, d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_is_load_pin, d_pin2net_map, d_net_driver_pin,
                d_flat_net2pin_start_map, d_flat_net2pin_map,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_active_level, num_power_levels, d_active);
        }
        if (num_clock_pins > 0) {
            power_seed_clock_active_kernel<<<BLOCK_NUMBER(num_clock_pins), BLOCK_SIZE>>>(
                d_clock_pins, num_clock_pins, clock_density,
                d_clock_pin_densities, d_clock_pin_duties, d_clock_pin_enqueue,
                d_pinSlew, time_unit, d_density, d_duty, d_origin,
                d_is_load_pin, d_pin2net_map, d_net_driver_pin,
                d_flat_net2pin_start_map, d_flat_net2pin_map,
                d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                d_pin_power_level, d_active_level, num_power_levels, d_active);
        }
        check_power_cuda_error("activity seed roots");
        int pending_count = 0;
        cudaMemcpy(&pending_count, d_pending_seq_count, sizeof(int), cudaMemcpyDeviceToHost);
        trace_cuda("after_seed", 0, pending_count);
        dump_cuda_pending_seq("after_seed", 0);

        bool defer_pending_seq = false;
        if (const char* env_defer_pending = std::getenv("XPLACE_POWER_DEFER_PENDING_SEQ"))
            defer_pending_seq = std::atoi(env_defer_pending) != 0;
        const bool trace_level_progress =
            std::getenv("XPLACE_POWER_TRACE_LEVEL_PROGRESS") != nullptr;
        int trace_level_progress_start_pass = 0;
        if (const char* env_trace_start = std::getenv("XPLACE_POWER_TRACE_LEVEL_PROGRESS_START_PASS"))
            trace_level_progress_start_pass = std::max(0, std::atoi(env_trace_start));
        int trace_level_progress_end_pass = max_activity_passes;
        if (const char* env_trace_end = std::getenv("XPLACE_POWER_TRACE_LEVEL_PROGRESS_END_PASS"))
            trace_level_progress_end_pass = std::max(0, std::atoi(env_trace_end));
        int trace_level_progress_pass = 0;
        const bool ascending_level_scan =
            std::getenv("XPLACE_POWER_ACTIVITY_ASCENDING_SCAN") != nullptr;
        bool use_serial_level = false;
        if (const char* env_serial_level = std::getenv("XPLACE_POWER_ACTIVITY_SERIAL_LEVEL"))
            use_serial_level = std::atoi(env_serial_level) != 0;
        int serial_level_max_active = 0x3fffffff;
        if (const char* env_serial_max = std::getenv("XPLACE_POWER_ACTIVITY_SERIAL_LEVEL_MAX_ACTIVE"))
            serial_level_max_active = std::max(0, std::atoi(env_serial_max));
        int serial_level_max_count = -1;
        if (const char* env_serial_count = std::getenv("XPLACE_POWER_ACTIVITY_SERIAL_LEVEL_MAX_COUNT"))
            serial_level_max_count = std::max(0, std::atoi(env_serial_count));
        std::vector<uint8_t> serial_level_selected(std::max(1, num_power_levels), 0);
        if (const char* env_serial_levels = std::getenv("XPLACE_POWER_ACTIVITY_SERIAL_LEVELS")) {
            std::stringstream stream(env_serial_levels);
            std::string item;
            while (std::getline(stream, item, ',')) {
                const int level = std::atoi(item.c_str());
                if (level >= 0 && level < num_power_levels) serial_level_selected[level] = 1;
            }
        }
        int* d_serial_active_pins = nullptr;
        int* d_serial_active_count = nullptr;
        if (use_serial_level) {
            cudaMalloc(&d_serial_active_pins, sizeof(int) * std::max(1, n));
            cudaMalloc(&d_serial_active_count, sizeof(int));
        }
        std::vector<uint8_t> active_levels(std::max(1, num_power_levels));
        auto run_level = [&](int level) {
            if (level < 0 || level >= num_power_levels || level + 1 >= static_cast<int>(level_list_end_cpu.size()))
                return;
            const int start = level_list_end_cpu[level];
            const int count = level_list_end_cpu[level + 1] - start;
            if (d_active_level) cudaMemsetAsync(d_active_level + level, 0, sizeof(uint8_t));
            if (count <= 0) {
                check_power_cuda_error("activity empty level");
                return;
            }
            if ((level < static_cast<int>(serial_level_selected.size()) && serial_level_selected[level])
                || (serial_level_max_count >= 0 && count <= serial_level_max_count)) {
                power_visit_level_serial_kernel<<<1, 1>>>(
                    d_level_list, start, count, d_case_values, d_is_load_pin, d_is_driver_pin,
                    d_pin2net_map, d_net_driver_pin, d_flat_net2pin_start_map, d_flat_net2pin_map,
                    d_pin_func_expr_id, d_missing_func_out_start, d_missing_func_out_list,
                    d_seq_pin_density, d_seq_pin_duty, d_seq_pin_valid,
                    d_clock_gate_out_for_input,
                    d_clock_gate_clock_for_out, d_clock_gate_enable_for_out,
                    d_expr_ops, d_expr_start, d_expr_count, clock_density,
                    d_pinSlew, time_unit, d_density, d_duty, d_origin, d_active,
                    d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                    d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                    d_pin_seq_list_start, d_pin_seq_list,
                    d_pin_power_level, d_active_level, num_power_levels, level,
                    d_pending_seq, d_pending_seq_count, defer_pending_seq);
            } else if (use_serial_level) {
                cudaMemset(d_serial_active_count, 0, sizeof(int));
                power_snapshot_level_active_list_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                    d_level_list, start, count, d_active, d_visit_active,
                    d_serial_active_count, d_serial_active_pins);
                check_power_cuda_error("activity snapshot level active list");
                int serial_active_count = 0;
                cudaMemcpy(&serial_active_count, d_serial_active_count, sizeof(int), cudaMemcpyDeviceToHost);
                if (serial_active_count <= serial_level_max_active) {
                    power_visit_active_list_serial_kernel<<<1, 1>>>(
                        d_serial_active_pins, d_serial_active_count,
                        d_case_values, d_is_load_pin, d_is_driver_pin,
                        d_pin2net_map, d_net_driver_pin, d_flat_net2pin_start_map, d_flat_net2pin_map,
                        d_pin_func_expr_id, d_missing_func_out_start, d_missing_func_out_list,
                        d_seq_pin_density, d_seq_pin_duty, d_seq_pin_valid,
                        d_clock_gate_out_for_input,
                        d_clock_gate_clock_for_out, d_clock_gate_enable_for_out,
                        d_expr_ops, d_expr_start, d_expr_count, clock_density,
                        d_pinSlew, time_unit, d_density, d_duty, d_origin, d_active, d_visit_active,
                        d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                        d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                        d_pin_seq_list_start, d_pin_seq_list,
                        d_pin_power_level, d_active_level, num_power_levels, level,
                        d_pending_seq, d_pending_seq_count, defer_pending_seq);
                } else {
                    power_visit_level_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                        d_level_list, start, count, d_case_values, d_is_load_pin, d_is_driver_pin,
                        d_pin2net_map, d_net_driver_pin, d_flat_net2pin_start_map, d_flat_net2pin_map,
                        d_pin_func_expr_id, d_missing_func_out_start, d_missing_func_out_list,
                        d_seq_pin_density, d_seq_pin_duty, d_seq_pin_valid,
                        d_clock_gate_out_for_input,
                        d_clock_gate_clock_for_out, d_clock_gate_enable_for_out,
                        d_expr_ops, d_expr_start, d_expr_count, clock_density,
                        d_pinSlew, time_unit, d_density, d_duty, d_origin, d_active, d_visit_active,
                        d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                        d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                        d_pin_seq_list_start, d_pin_seq_list,
                        d_pin_power_level, d_active_level, num_power_levels, level,
                        d_pending_seq, d_pending_seq_count, defer_pending_seq);
                }
            } else {
                power_snapshot_level_active_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                    d_level_list, start, count, d_active, d_visit_active);
                check_power_cuda_error("activity snapshot level active");
                power_visit_level_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                    d_level_list, start, count, d_case_values, d_is_load_pin, d_is_driver_pin,
                    d_pin2net_map, d_net_driver_pin, d_flat_net2pin_start_map, d_flat_net2pin_map,
                    d_pin_func_expr_id, d_missing_func_out_start, d_missing_func_out_list,
                    d_seq_pin_density, d_seq_pin_duty, d_seq_pin_valid,
                    d_clock_gate_out_for_input,
                    d_clock_gate_clock_for_out, d_clock_gate_enable_for_out,
                    d_expr_ops, d_expr_start, d_expr_count, clock_density,
                    d_pinSlew, time_unit, d_density, d_duty, d_origin, d_active, d_visit_active,
                    d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                    d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                    d_pin_seq_list_start, d_pin_seq_list,
                    d_pin_power_level, d_active_level, num_power_levels, level,
                    d_pending_seq, d_pending_seq_count, defer_pending_seq);
            }
            check_power_cuda_error("activity visit level");
        };

        const bool print_pass_stats = std::getenv("XPLACE_POWER_PRINT_PASS_STATS") != nullptr;
        int total_comb_sweeps = 0;
        auto drain_bfs = [&]() {
            if (defer_pending_seq) {
                cudaMemcpy(d_prev_density, d_density, sizeof(float) * n, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_prev_duty, d_duty, sizeof(float) * n, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_prev_origin, d_origin, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            }
            int level_visits = 0;
            const int max_level_visits = max_comb_sweeps;
            if (ascending_level_scan) {
                for (int level = 0; level < num_power_levels && level_visits < max_level_visits; ++level) {
                    uint8_t level_active = 0;
                    cudaMemcpy(&level_active, d_active_level + level, sizeof(uint8_t),
                               cudaMemcpyDeviceToHost);
                    if (!level_active) continue;
                    run_level(level);
                    ++level_visits;
                    if (trace_level_progress
                        && trace_level_progress_pass >= trace_level_progress_start_pass
                        && trace_level_progress_pass <= trace_level_progress_end_pass) {
                        char tag[64];
                        snprintf(tag, sizeof(tag), "after_level:%d", level);
                        trace_cuda(tag, trace_level_progress_pass, -1);
                    }
                }
            } else {
            for (; level_visits < max_level_visits; ++level_visits) {
                if (num_power_levels <= 0) break;
                cudaMemcpy(active_levels.data(), d_active_level, sizeof(uint8_t) * num_power_levels,
                           cudaMemcpyDeviceToHost);
                int next_level = -1;
                for (int level = 0; level < num_power_levels; ++level) {
                    if (active_levels[level]) {
                        next_level = level;
                        break;
                    }
                }
                if (next_level < 0) break;
                run_level(next_level);
                if (trace_level_progress
                    && trace_level_progress_pass >= trace_level_progress_start_pass
                    && trace_level_progress_pass <= trace_level_progress_end_pass) {
                    char tag[64];
                    snprintf(tag, sizeof(tag), "after_level:%d", next_level);
                    trace_cuda(tag, trace_level_progress_pass, -1);
                }
            }
            }
            total_comb_sweeps += level_visits;
            if (defer_pending_seq) {
                power_mark_pending_seq_changes_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                    n, d_is_load_pin, d_prev_density, d_prev_duty, d_prev_origin,
                    d_density, d_duty, d_origin, d_pin_seq_list_start, d_pin_seq_list,
                    d_pending_seq, d_pending_seq_count);
                check_power_cuda_error("activity mark pending seq changes");
            }
            return level_visits;
        };

        trace_level_progress_pass = 0;
        drain_bfs();
        cudaMemcpy(&pending_count, d_pending_seq_count, sizeof(int), cudaMemcpyDeviceToHost);
        trace_cuda("after_comb", 0, pending_count);
        dump_cuda_pending_seq("after_comb", 0);
        int seq_passes = 0;
        const bool direct_ordered_seq_seed =
            std::getenv("XPLACE_POWER_DIRECT_ORDERED_SEQ_SEED") != nullptr;
        const bool ordered_seq_seed =
            direct_ordered_seq_seed || std::getenv("XPLACE_POWER_ORDERED_SEQ_SEED") != nullptr;
        const int direct_ordered_seq_seed_device = direct_ordered_seq_seed ? 1 : 0;
        cudaMemcpyToSymbol(g_power_direct_ordered_seq_seed,
                           &direct_ordered_seq_seed_device,
                           sizeof(int));
        int* d_ordered_pending_seq_ids = nullptr;
        std::vector<int> h_ordered_pending_flags;
        std::vector<int> h_ordered_pending_seq_ids;
        if (ordered_seq_seed && num_seqs > 0) {
            cudaMalloc(&d_ordered_pending_seq_ids, sizeof(int) * num_seqs);
            h_ordered_pending_flags.resize(num_seqs);
            h_ordered_pending_seq_ids.reserve(num_seqs);
        }
        for (int pass = 1; pending_count > 0 && pass < max_activity_passes; pass++) {
            seq_passes = pass;
            const int pending_before_seed = pending_count;
            if (num_seqs > 0) {
                if (seq_clock_limit_rel_tol > 0.0f && seq_clock_limit_rel_tol_start_pass > 1) {
                    const float active_tol =
                        pass >= seq_clock_limit_rel_tol_start_pass ? seq_clock_limit_rel_tol : 0.0f;
                    cudaMemcpyToSymbol(g_power_seq_clock_limit_rel_tol,
                                       &active_tol,
                                       sizeof(float));
                }
                if (ordered_seq_seed) {
                    h_ordered_pending_seq_ids.clear();
                    cudaMemcpy(h_ordered_pending_flags.data(), d_pending_seq,
                               sizeof(int) * num_seqs, cudaMemcpyDeviceToHost);
                    for (int seq_id = 0; seq_id < num_seqs; ++seq_id) {
                        if (h_ordered_pending_flags[seq_id])
                            h_ordered_pending_seq_ids.push_back(seq_id);
                    }
                    if (std::getenv("XPLACE_POWER_REVERSE_ORDERED_SEQ_SEED") != nullptr) {
                        std::reverse(h_ordered_pending_seq_ids.begin(),
                                     h_ordered_pending_seq_ids.end());
                    }
                    if (!h_ordered_pending_seq_ids.empty()) {
                        cudaMemcpy(d_ordered_pending_seq_ids,
                                   h_ordered_pending_seq_ids.data(),
                                   sizeof(int) * h_ordered_pending_seq_ids.size(),
                                   cudaMemcpyHostToDevice);
                        const int ordered_pending_count =
                            static_cast<int>(h_ordered_pending_seq_ids.size());
                        power_seed_seq_id_list_ordered_kernel<<<1, 1>>>(
                            d_seqs, d_ordered_pending_seq_ids, ordered_pending_count,
                            d_expr_ops, d_expr_start, d_expr_count, clock_density,
                            d_pinSlew, time_unit, d_density, d_duty, d_origin,
                            d_seq_pin_density, d_seq_pin_duty, d_seq_pin_valid,
                            d_pending_seq, d_pending_seq_count,
                            d_pin_power_level, d_active_level, num_power_levels, d_active);
                    } else {
                        cudaMemset(d_pending_seq_count, 0, sizeof(int));
                    }
                } else {
                    power_seed_seq_kernel<<<BLOCK_NUMBER(num_seqs), BLOCK_SIZE>>>(
                        d_seqs, num_seqs, d_expr_ops, d_expr_start, d_expr_count, clock_density,
                        d_pinSlew, time_unit, d_density, d_duty, d_origin,
                        d_seq_pin_density, d_seq_pin_duty, d_seq_pin_valid,
                        d_pending_seq, d_pending_seq_count,
                        d_is_load_pin, d_pin2net_map, d_net_driver_pin,
                        d_flat_net2pin_start_map, d_flat_net2pin_map,
                        d_pin_forward_arc_list_end, d_pin_forward_arc_list, d_timing_arc_to_pin_id,
                        d_arc_types, d_arc_id2test_id, d_is_seq_output_pin,
                        d_pin_power_level, d_active_level, num_power_levels, d_active);
                }
                check_power_cuda_error("activity seed seq");
            }
            trace_cuda("after_seq_seed", pass, pending_before_seed);
            dump_cuda_pending_seq("after_seq_seed", pass);
            trace_level_progress_pass = pass;
            const int comb_sweeps = drain_bfs();
            cudaMemcpy(&pending_count, d_pending_seq_count, sizeof(int), cudaMemcpyDeviceToHost);
            trace_cuda("after_pass", pass, pending_count);
            dump_cuda_pending_seq("after_pass", pass);
            if (print_pass_stats && std::getenv("XPLACE_POWER_PRINT_PASS_STATS_VERBOSE")) {
                fprintf(stderr,
                        "[power_activity_pass] pass=%d pending=%d comb_sweeps=%d\n",
                        pass, pending_count, comb_sweeps);
            }
        }
        if (d_ordered_pending_seq_ids) cudaFree(d_ordered_pending_seq_ids);
        if (print_pass_stats) {
            fprintf(stderr,
                    "[power_activity_passes] seq_passes=%d final_pending=%d total_comb_sweeps=%d max_seq_passes=%d max_comb_sweeps=%d\n",
                    seq_passes, pending_count, total_comb_sweeps, max_activity_passes, max_comb_sweeps);
        }
        if (d_serial_active_pins) cudaFree(d_serial_active_pins);
        if (d_serial_active_count) cudaFree(d_serial_active_count);
    }

    }

    if (d_out) {
        power_pack_output_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(n, d_density, d_duty, d_origin, d_out);
        check_power_cuda_error("activity pack output");
    }
    if ((d_inst_switching || d_pin_switching) && d_out && d_pin2node_map && d_pinLoad) {
        if (d_inst_switching) cudaMemset(d_inst_switching, 0, sizeof(float) * num_nodes);
        if (d_pin_switching) cudaMemset(d_pin_switching, 0, sizeof(float) * n);
        power_switching_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
            n, num_nodes, d_is_driver_pin, d_is_cell_pin, d_pin2node_map, d_pinLoad, d_dmp_C1, d_dmp_C2, cap_unit, voltage,
            d_out, d_inst_switching, d_pin_switching);
        check_power_cuda_error("activity switching");
    }
    if ((d_inst_internal || d_internal_row_power) && d_internal_rows && num_internal_rows > 0 && d_power_allocator) {
        if (d_inst_internal) cudaMemset(d_inst_internal, 0, sizeof(float) * num_nodes);
        if (d_internal_row_power) cudaMemset(d_internal_row_power, 0, sizeof(float) * num_internal_rows);
        float* d_denom = nullptr;
        cudaMalloc(&d_denom, sizeof(float) * std::max(1, num_internal_denom_groups));
        cudaMemset(d_denom, 0, sizeof(float) * std::max(1, num_internal_denom_groups));
        power_internal_denom_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_internal_rows, num_internal_rows, d_expr_ops, d_expr_start, d_expr_count,
            d_node_port_pin_start, d_node_port_pin_list, d_density, d_duty, d_denom);
        check_power_cuda_error("activity internal denom");
        power_internal_contrib_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_internal_rows, num_internal_rows, num_nodes, d_expr_ops, d_expr_start, d_expr_count,
            d_node_port_pin_start, d_node_port_pin_list, d_density, d_duty, d_pinSlew, d_power_clock_slews, d_dmp_C1, d_dmp_C2, d_denom, d_power_allocator, cap_unit, d_inst_internal, d_internal_row_power);
        check_power_cuda_error("activity internal contrib");
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
                d_node_port_pin_start, d_node_port_pin_list, d_density, d_duty, d_group_cond_leakage, d_group_cond_duty_sum, d_group_cond_count, d_leakage_row_power);
            check_power_cuda_error("activity leakage rows");
        }
        if (d_inst_leakage) {
            power_leakage_summary_kernel<<<BLOCK_NUMBER(num_leakage_groups), BLOCK_SIZE>>>(
                d_leakage_groups, num_leakage_groups, d_group_cond_leakage, d_group_cond_duty_sum, d_group_cond_count,
                num_nodes, d_inst_leakage);
            check_power_cuda_error("activity leakage summary");
        }
        cudaFree(d_group_cond_leakage);
        cudaFree(d_group_cond_duty_sum);
        cudaFree(d_group_cond_count);
    }

    cudaFree(d_density);
    cudaFree(d_duty);
    cudaFree(d_prev_density);
    cudaFree(d_prev_duty);
    cudaFree(d_seq_pin_density);
    cudaFree(d_seq_pin_duty);
    cudaFree(d_origin);
    cudaFree(d_prev_origin);
    cudaFree(d_active);
    cudaFree(d_active_level);
    cudaFree(d_visit_active);
    cudaFree(d_seq_pin_valid);
    cudaFree(d_pending_seq);
    cudaFree(d_pending_seq_count);
}

}  // namespace gt
