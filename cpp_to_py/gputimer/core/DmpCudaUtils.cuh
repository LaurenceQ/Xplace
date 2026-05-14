#pragma once

#include "DmpModel.h"
#include "gputiming.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gt {

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        std::fprintf(stderr, "GPUassert: %s,\nat %s, line %d\n",
                     cudaGetErrorString(code), file, line);
        if (abort) {
            std::exit(code);
        }
    }
}

static constexpr int DMP_TIMING_BLOCK_SIZE = 128;
static constexpr int DMP_DEBUG_BLOCK_SIZE = 128;
static constexpr int DMP_PIN_GROUP_SIZE = 2 * NUM_ATTR;

inline void dmp_clear_stale_cuda_error(const char* label)
{
    (void)label;
    cudaGetLastError();
}

inline void dmp_event_create(cudaEvent_t* start, cudaEvent_t* stop)
{
    gpuAssert(cudaEventCreate(start), __FILE__, __LINE__);
    gpuAssert(cudaEventCreate(stop), __FILE__, __LINE__);
}

inline bool dmp_kernel_profile_enabled()
{
    const char* value = std::getenv("DMP_PROFILE_KERNELS");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

inline bool dmp_timing_debug_enabled()
{
    const char* value = std::getenv("DMP_DEBUG_TIMING");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

__device__ __forceinline__ double exp2(double x) {
    if (x < -12.0) {
        return 0.0;
    }
    double y = 1.0 + x / 4096.0;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    return y;
}

__device__ __forceinline__ double DmpModel::y0(double t, double rd, double cl) {
    return t - rd * cl * (1.0 - exp2(-t / (rd * cl)));
}

__device__ __forceinline__ double DmpModel::y(double t,
                                              double t0,
                                              double dt,
                                              double rd,
                                              double cl) {
    double t1 = t - t0;
    if (t1 <= 0.0) {
        return 0.0;
    }
    if (t1 <= dt) {
        return y0(t1, rd, cl) / dt;
    }
    return (y0(t1, rd, cl) - y0(t1 - dt, rd, cl)) / dt;
}

__device__ __forceinline__ double DmpModel::y0dt(double t, double rd, double cl) {
    return 1.0 - exp2(-t / (rd * cl));
}

__device__ __forceinline__ double DmpModel::y0dcl(double t, double rd, double cl) {
    return rd * ((1.0 + t / (rd * cl)) * exp2(-t / (rd * cl)) - 1);
}

__device__ __forceinline__ void DmpModel::dy(double t,
                                             double t0,
                                             double dt,
                                             double rd,
                                             double cl,
                                             double &dydt0,
                                             double &dyddt,
                                             double &dydcl) {
    double t1 = t - t0;
    if (t1 <= 0.0) {
        dydt0 = dyddt = dydcl = 0.0;
    } else if (t1 <= dt) {
        dydt0 = -y0dt(t1, rd, cl) / dt;
        dyddt = -y0(t1, rd, cl) / (dt * dt);
        dydcl = y0dcl(t1, rd, cl) / dt;
    } else {
        dydt0 = -(y0dt(t1, rd, cl) - y0dt(t1 - dt, rd, cl)) / dt;
        dyddt = -(y0(t1, rd, cl) + y0(t1 - dt, rd, cl)) / (dt * dt) +
                 y0dt(t1 - dt, rd, cl) / dt;
        dydcl = (y0dcl(t1, rd, cl) - y0dcl(t1 - dt, rd, cl)) / dt;
    }
}

__global__ void dmpBackwardKernel(DmpModel* model,
                                  int level_start_offset,
                                  int num_pins_level);
__global__ void dmpTestKernel(DmpModel* model,
                              int level_start_offset,
                              int num_pins_level);
__global__ void dmpDirectNetKernel(DmpModel* model,
                                   const index_type* level_arc_list,
                                   int num_level_arcs);
__global__ void dmpGateKernel(DmpModel* model,
                              const index_type* level_arc_list,
                              int num_level_arcs,
                              unsigned long long* debug_counts);
__global__ void dmpPinWinnerKernel(DmpModel* model,
                                   int level_start_offset,
                                   int num_pins_level);
__global__ void dmpNetWinnerKernel(DmpModel* model,
                                   const index_type* level_arc_list,
                                   int num_level_arcs);

void reset_dmp_root_profile_cuda();
void print_dmp_root_profile_cuda();

void dmp_debug_print_counts(DmpModel* model, const char* label);
void dmp_debug_print_first_level_sample(DmpModel* model,
                                        int level_idx,
                                        index_type level_start_offset);
void dmp_debug_print_parallel_stats(DmpModel* model,
                                    const std::vector<int>& level_list_end_cpu,
                                    const char* label);

} // namespace gt

#define gpuErrchk(ans) \
    do { \
        ::gt::gpuAssert((ans), __FILE__, __LINE__); \
    } while (0)

#define DMP_TIMING_BLOCK_NUMBER(n) (((n) + ::gt::DMP_TIMING_BLOCK_SIZE - 1) / ::gt::DMP_TIMING_BLOCK_SIZE)
#define DMP_DEBUG_BLOCK_NUMBER(n) (((n) + ::gt::DMP_DEBUG_BLOCK_SIZE - 1) / ::gt::DMP_DEBUG_BLOCK_SIZE)
