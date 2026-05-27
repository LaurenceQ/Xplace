#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "common/lib/Liberty.h"
#include "common/lib/Lut.h"
#include "common/lib/Timing.h"

using std::vector;

namespace gt {

struct DmpGateLutMeta;

template <typename T>
__device__ __forceinline__ int lower_bound(T* arr, int size, T val);

template <typename T>
__device__ __forceinline__ float interpolate(T x1, T x2, T y1, T y2, T x);

class GPULutAllocator {
public:
    int num_luts_in_timing = 6;
    int num_luts;
    int x_size = 0, y_size = 0, table_size = 0;

    int *num_x, *num_y, *num_table;
    float *x_array, *y_array, *table_array;
    size_t *x_offset, *y_offset, *table_offset;
    bool *allocated;

    int *d_num_x, *d_num_y, *d_num_table;
    float *d_x_array, *d_y_array, *d_table_array;
    size_t *d_x_offset, *d_y_offset, *d_table_offset;
    bool *d_allocated;

    int num_timings;
    int* timing_sense;
    int* lut_template_var;
    bool *is_rising_edge_triggered, *is_falling_edge_triggered, *is_constraint, *is_latch_clock_arc;

    int* d_timing_sense;
    int* d_lut_template_var;
    bool *d_is_rising_edge_triggered, *d_is_falling_edge_triggered, *d_is_constraint, *d_is_latch_clock_arc;

public:
    GPULutAllocator() = default;
    void AllocateBatch(vector<TimingArc*> timings);
    void CopyToGPU();
    void CopyToGPU(GPULutAllocator* d_gpuluts);
    __device__ __forceinline__ bool is_input_transition_defined(int timing_id, int irf);
    __device__ __forceinline__ bool is_transition_defined(int timing_id, int irf, int orf);
    __device__ __forceinline__ float lut(int in_timing_lut, float x, float y);
    __device__ DmpGateLutMeta makeGateLutMeta(int lut_id);
    __device__ __forceinline__ float gateLutWithMeta(const DmpGateLutMeta& meta,
                                                     float input_slew,
                                                     float load);
    __device__ __forceinline__ float queryGateLutNoTransitionCheck(int timing_id,
                                                                   int output_rf,
                                                                   float input_slew,
                                                                   float load,
                                                                   int type);
    __device__ __forceinline__ float query(int timing_id, int irf, int orf, float slew_or_related,
                                           float load_or_constraint, int type);
    void freeMem();
    ~GPULutAllocator();
};

class GPUPowerLutAllocator {
public:
    int num_luts_in_internal_power = 2;
    int num_internal_powers = 0;
    int num_luts = 0;
    int x_size = 0, y_size = 0, table_size = 0;
    int *num_x = nullptr, *num_y = nullptr, *num_table = nullptr;
    float *x_array = nullptr, *y_array = nullptr, *table_array = nullptr;
    size_t *x_offset = nullptr, *y_offset = nullptr, *table_offset = nullptr;
    bool* allocated = nullptr;
    int* lut_template_var = nullptr;
    int *d_num_x = nullptr, *d_num_y = nullptr, *d_num_table = nullptr;
    float *d_x_array = nullptr, *d_y_array = nullptr, *d_table_array = nullptr;
    size_t *d_x_offset = nullptr, *d_y_offset = nullptr, *d_table_offset = nullptr;
    bool* d_allocated = nullptr;
    int* d_lut_template_var = nullptr;

public:
    GPUPowerLutAllocator() = default;
    void AllocateBatch(const vector<InternalPower*>& internal_powers);
    void CopyToGPU();
    void CopyToGPU(GPUPowerLutAllocator* d_gpu_luts);
    __device__ __forceinline__ float lut(int lut_id, float x, float y);
    __device__ __forceinline__ float query_internal_power(int internal_power_id, int rf, float input_slew,
                                                          float output_load);
    void freeMem();
    ~GPUPowerLutAllocator();
};

template <typename T>
__device__ __forceinline__ int lower_bound(T* arr, int size, T val) {
    int l = 0, r = size - 1;
    while (l < r) {
        int m = (l + r) / 2;
        if (arr[m] < val)
            l = m + 1;
        else
            r = m;
    }
    return l;
}

template <typename T>
__device__ __forceinline__ float interpolate(T x1, T x2, T y1, T y2, T x) {
    if (x1 == x2) return y1;
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
}

__device__ __forceinline__ bool GPULutAllocator::is_input_transition_defined(int timing_id, int irf) {
    if (d_is_rising_edge_triggered[timing_id] && irf != 0) return false;
    if (d_is_falling_edge_triggered[timing_id] && irf != 1) return false;
    return true;
}

__device__ __forceinline__ bool GPULutAllocator::is_transition_defined(int timing_id, int irf, int orf) {
    if (!is_input_transition_defined(timing_id, irf)) return false;
    int sense = d_timing_sense[timing_id];
    if (sense != -1) {
        switch (sense) {
            case 1:
                if (irf != orf) return false;
                break;
            case 2:
                if (irf == orf) return false;
                break;
            default:
                break;
        }
    }
    return true;
}

__device__ __forceinline__ float GPULutAllocator::lut(int in_timing_lut, float x, float y) {
    if (d_num_x[in_timing_lut] < 1 || d_num_y[in_timing_lut] < 1) return nanf("");
    if (d_num_table[in_timing_lut] == 1) return d_table_array[d_table_offset[in_timing_lut]];
    int x_idx[2], y_idx[2];
    x_idx[1] = lower_bound<float>(d_x_array + d_x_offset[in_timing_lut], d_num_x[in_timing_lut], x);
    y_idx[1] = lower_bound<float>(d_y_array + d_y_offset[in_timing_lut], d_num_y[in_timing_lut], y);
    x_idx[1] = max(1, min(d_num_x[in_timing_lut] - 1, x_idx[1]));
    y_idx[1] = max(1, min(d_num_y[in_timing_lut] - 1, y_idx[1]));
    x_idx[0] = x_idx[1] - 1;
    y_idx[0] = y_idx[1] - 1;
    if (d_num_x[in_timing_lut] == 1) x_idx[1] = 0;
    if (d_num_y[in_timing_lut] == 1) y_idx[1] = 0;

    float numeric[2];
    numeric[0] = interpolate<float>(d_x_array[d_x_offset[in_timing_lut] + x_idx[0]],
                                    d_x_array[d_x_offset[in_timing_lut] + x_idx[1]],
                                    d_table_array[d_table_offset[in_timing_lut] + x_idx[0] * d_num_y[in_timing_lut] + y_idx[0]],
                                    d_table_array[d_table_offset[in_timing_lut] + x_idx[1] * d_num_y[in_timing_lut] + y_idx[0]],
                                    x);
    numeric[1] = interpolate<float>(d_x_array[d_x_offset[in_timing_lut] + x_idx[0]],
                                    d_x_array[d_x_offset[in_timing_lut] + x_idx[1]],
                                    d_table_array[d_table_offset[in_timing_lut] + x_idx[0] * d_num_y[in_timing_lut] + y_idx[1]],
                                    d_table_array[d_table_offset[in_timing_lut] + x_idx[1] * d_num_y[in_timing_lut] + y_idx[1]],
                                    x);
    return interpolate<float>(d_y_array[d_y_offset[in_timing_lut] + y_idx[0]],
                              d_y_array[d_y_offset[in_timing_lut] + y_idx[1]],
                              numeric[0], numeric[1], y);
}

__device__ __forceinline__ float GPULutAllocator::query(int timing_id,
                                                        int irf,
                                                        int orf,
                                                        float slew_or_related,
                                                        float load_or_constraint,
                                                        int type) {
    if (!is_transition_defined(timing_id, irf, orf)) return nanf("");
    int in_timing_lut = num_luts_in_timing * timing_id + orf + type * 2;
    in_timing_lut = d_allocated[in_timing_lut] ? in_timing_lut : -1;
    if (in_timing_lut == -1) return nanf("");

    float val1{0.0f}, val2{0.0f};
    if (type == 0 || type == 1) {
        switch (d_lut_template_var[in_timing_lut * 2]) {
            case 0:
                if (d_lut_template_var[in_timing_lut * 2 + 1] != -1)
                    assert(d_lut_template_var[in_timing_lut * 2 + 1] == 1);
                val1 = load_or_constraint;
                val2 = slew_or_related;
                break;
            case 1:
                if (d_lut_template_var[in_timing_lut * 2 + 1] != -1)
                    assert(d_lut_template_var[in_timing_lut * 2 + 1] == 0);
                val1 = slew_or_related;
                val2 = load_or_constraint;
                break;
            default:
                break;
        }
    } else if (type == 2) {
        switch (d_lut_template_var[in_timing_lut * 2]) {
            case 2:
                if (d_lut_template_var[in_timing_lut * 2 + 1] != -1)
                    assert(d_lut_template_var[in_timing_lut * 2 + 1] == 3);
                val1 = load_or_constraint;
                val2 = slew_or_related;
                break;
            case 3:
                if (d_lut_template_var[in_timing_lut * 2 + 1] != -1)
                    assert(d_lut_template_var[in_timing_lut * 2 + 1] == 2);
                val1 = slew_or_related;
                val2 = load_or_constraint;
                break;
            default:
                break;
        }
    }
    return lut(in_timing_lut, val1, val2);
}

__device__ __forceinline__ float GPUPowerLutAllocator::lut(int lut_id, float x, float y) {
    if (lut_id < 0 || lut_id >= num_luts || !d_allocated[lut_id]) return nanf("");
    if (d_num_x[lut_id] < 1 || d_num_y[lut_id] < 1) return nanf("");
    if (d_num_table[lut_id] == 1) return d_table_array[d_table_offset[lut_id]];
    int x_idx[2], y_idx[2];
    x_idx[1] = lower_bound<float>(d_x_array + d_x_offset[lut_id], d_num_x[lut_id], x);
    y_idx[1] = lower_bound<float>(d_y_array + d_y_offset[lut_id], d_num_y[lut_id], y);
    x_idx[1] = max(1, min(d_num_x[lut_id] - 1, x_idx[1]));
    y_idx[1] = max(1, min(d_num_y[lut_id] - 1, y_idx[1]));
    x_idx[0] = x_idx[1] - 1;
    y_idx[0] = y_idx[1] - 1;
    if (d_num_x[lut_id] == 1) x_idx[1] = 0;
    if (d_num_y[lut_id] == 1) y_idx[1] = 0;
    float numeric[2];
    numeric[0] = interpolate<float>(d_x_array[d_x_offset[lut_id] + x_idx[0]],
                                    d_x_array[d_x_offset[lut_id] + x_idx[1]],
                                    d_table_array[d_table_offset[lut_id] + x_idx[0] * d_num_y[lut_id] + y_idx[0]],
                                    d_table_array[d_table_offset[lut_id] + x_idx[1] * d_num_y[lut_id] + y_idx[0]], x);
    numeric[1] = interpolate<float>(d_x_array[d_x_offset[lut_id] + x_idx[0]],
                                    d_x_array[d_x_offset[lut_id] + x_idx[1]],
                                    d_table_array[d_table_offset[lut_id] + x_idx[0] * d_num_y[lut_id] + y_idx[1]],
                                    d_table_array[d_table_offset[lut_id] + x_idx[1] * d_num_y[lut_id] + y_idx[1]], x);
    return interpolate<float>(d_y_array[d_y_offset[lut_id] + y_idx[0]],
                              d_y_array[d_y_offset[lut_id] + y_idx[1]],
                              numeric[0], numeric[1], y);
}

__device__ __forceinline__ float GPUPowerLutAllocator::query_internal_power(int internal_power_id,
                                                                            int rf,
                                                                            float input_slew,
                                                                            float output_load) {
    if (internal_power_id < 0 || internal_power_id >= num_internal_powers || rf < 0 || rf > 1) return nanf("");
    int lut_id = num_luts_in_internal_power * internal_power_id + rf;
    if (!d_allocated[lut_id]) return nanf("");
    float val1 = input_slew;
    float val2 = output_load;
    switch (d_lut_template_var[lut_id * 2]) {
        case 0:
            val1 = output_load;
            val2 = input_slew;
            break;
        case 1:
        case 4:
            val1 = input_slew;
            val2 = output_load;
            break;
        default:
            break;
    }
    return lut(lut_id, val1, val2);
}

}  // namespace gt
