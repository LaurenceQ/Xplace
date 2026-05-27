#pragma once

#include "DmpModel.h"
#include "gputiming.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gt {

void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true);

static constexpr int DMP_TIMING_BLOCK_SIZE = 128;
static constexpr int DMP_DEBUG_BLOCK_SIZE = 128;
static constexpr int DMP_PIN_GROUP_SIZE = 2 * NUM_ATTR;

void dmp_clear_stale_cuda_error(const char* label);
void dmp_event_create(cudaEvent_t* start, cudaEvent_t* stop);
bool dmp_kernel_profile_enabled();
bool dmp_timing_debug_enabled();

__device__ __forceinline__ float exp2(float x);
__device__ inline bool dmpSolve2x2(double a00,
                                      double a01,
                                      double a10,
                                      double a11,
                                      double b0,
                                      double b1,
                                      double& x0,
                                      double& x1);
__device__ inline bool dmpSolve3x3(double a00,
                                      double a01,
                                      double a02,
                                      double a10,
                                      double a11,
                                      double a12,
                                      double a20,
                                      double a21,
                                      double a22,
                                      double b0,
                                      double b1,
                                      double b2,
                                      double& x0,
                                      double& x1,
                                      double& x2);
__device__ inline bool dmpSolve3x3A00Zero(double a01,
                                             double a02,
                                             double a10,
                                             double a11,
                                             double a12,
                                             double a20,
                                             double a21,
                                             double a22,
                                             double b0,
                                             double b1,
                                             double b2,
                                             double& x0,
                                             double& x1,
                                             double& x2);
__global__ void dmpBackwardKernel(DmpModel* model,
                                  int level_start_offset,
                                  int num_pins_level);
__global__ void dmpTestKernel(DmpModel* model,
                              int level_start_offset,
                              int num_pins_level);
__global__ void dmpDirectNetKernel(DmpModel* model,
                                   const index_type* level_arc_list,
                                   int num_level_arcs);
__global__ void dmpResetForwardTargetsKernel(DmpModel* model);
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
void dmp_debug_print_driving_cell_counts(int num_sources,
                                         int total,
                                         const unsigned long long* counts);
void dmp_debug_print_driving_cell_kernel_profile(float elapsed_ms, int total);

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort)
{
    if (code != cudaSuccess) {
        std::fprintf(stderr, "GPUassert: %s,\nat %s, line %d\n",
                     cudaGetErrorString(code), file, line);
        if (abort) {
            std::exit(code);
        }
    }
}


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

__device__ __forceinline__ float exp2(float x) {
    if (x < -12.0f) {
        return 0.0f;
    }
    return __expf(x);
}

__device__ inline bool dmpSolve2x2(double a00,
                                      double a01,
                                      double a10,
                                      double a11,
                                      double b0,
                                      double b1,
                                      double& x0,
                                      double& x1) {
    const double det = a00 * a11 - a01 * a10;
    const double scale = fabs(a00 * a11) + fabs(a01 * a10) + 1e-300;
    if (!isfinite(det) || fabs(det) <= 1e-14 * scale) {
        return false;
    }
    x0 = (b0 * a11 - a01 * b1) / det;
    x1 = (a00 * b1 - b0 * a10) / det;
    return isfinite(x0) && isfinite(x1);
}

__device__ inline bool dmpSolve3x3(double a00,
                                      double a01,
                                      double a02,
                                      double a10,
                                      double a11,
                                      double a12,
                                      double a20,
                                      double a21,
                                      double a22,
                                      double b0,
                                      double b1,
                                      double b2,
                                      double& x0,
                                      double& x1,
                                      double& x2) {
    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = a10 * a22 - a12 * a20;
    const double c02 = a10 * a21 - a11 * a20;
    const double det = a00 * c00 - a01 * c01 + a02 * c02;
    const double scale = fabs(a00 * c00) + fabs(a01 * c01) + fabs(a02 * c02) + 1e-300;
    if (!isfinite(det) || fabs(det) <= 1e-14 * scale) {
        return false;
    }
    const double det0 = b0 * c00 - a01 * (b1 * a22 - a12 * b2) +
                        a02 * (b1 * a21 - a11 * b2);
    const double det1 = a00 * (b1 * a22 - a12 * b2) -
                        b0 * c01 +
                        a02 * (a10 * b2 - b1 * a20);
    const double det2 = a00 * (a11 * b2 - b1 * a21) -
                        a01 * (a10 * b2 - b1 * a20) +
                        b0 * c02;
    x0 = det0 / det;
    x1 = det1 / det;
    x2 = det2 / det;
    return isfinite(x0) && isfinite(x1) && isfinite(x2);
}

__device__ inline bool dmpSolve3x3A00Zero(double a01,
                                             double a02,
                                             double a10,
                                             double a11,
                                             double a12,
                                             double a20,
                                             double a21,
                                             double a22,
                                             double b0,
                                             double b1,
                                             double b2,
                                             double& x0,
                                             double& x1,
                                             double& x2) {
    const double e1 = a20 * a11 - a10 * a21;
    const double e2 = a20 * a12 - a10 * a22;
    const double eb = a20 * b1 - a10 * b2;
    const double det = a01 * e2 - a02 * e1;
    const double scale = fabs(a01 * e2) + fabs(a02 * e1) + 1e-300;
    if (!isfinite(det) || fabs(det) <= 1e-14 * scale) {
        return false;
    }

    x1 = (b0 * e2 - a02 * eb) / det;
    x2 = (a01 * eb - b0 * e1) / det;
    const bool use_row1 = fabs(a10) >= fabs(a20);
    const double x0_denom = use_row1 ? a10 : a20;
    if (!isfinite(x0_denom) || x0_denom == 0.0) {
        return false;
    }
    x0 = use_row1
             ? (b1 - a11 * x1 - a12 * x2) / a10
             : (b2 - a21 * x1 - a22 * x2) / a20;
    return isfinite(x0) && isfinite(x1) && isfinite(x2);
}
} // namespace gt

#define gpuErrchk(ans)     do {         ::gt::gpuAssert((ans), __FILE__, __LINE__);     } while (0)

#define DMP_TIMING_BLOCK_NUMBER(n) (((n) + ::gt::DMP_TIMING_BLOCK_SIZE - 1) / ::gt::DMP_TIMING_BLOCK_SIZE)
#define DMP_DEBUG_BLOCK_NUMBER(n) (((n) + ::gt::DMP_DEBUG_BLOCK_SIZE - 1) / ::gt::DMP_DEBUG_BLOCK_SIZE)
