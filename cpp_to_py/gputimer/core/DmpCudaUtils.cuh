#pragma once

#include "gputiming.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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

// Arc-level forward uses per-arc DMP scratch slots and locked destination
// winner writes. Keep the pin fallback compiled for validation.
static constexpr bool DMP_FORWARD_ARC_LEVEL = true;

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

} // namespace gt

#define gpuErrchk(ans) \
    do { \
        ::gt::gpuAssert((ans), __FILE__, __LINE__); \
    } while (0)

#define DMP_TIMING_BLOCK_NUMBER(n) (((n) + ::gt::DMP_TIMING_BLOCK_SIZE - 1) / ::gt::DMP_TIMING_BLOCK_SIZE)
#define DMP_DEBUG_BLOCK_NUMBER(n) (((n) + ::gt::DMP_DEBUG_BLOCK_SIZE - 1) / ::gt::DMP_DEBUG_BLOCK_SIZE)
