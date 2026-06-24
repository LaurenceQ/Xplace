#include "PowerCudaActivityKernels.cuh"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gt {

void clear_power_cuda_error() { cudaGetLastError(); }

void check_power_cuda_error(const char* label) {
    cudaError_t last_err = cudaGetLastError();
    if (last_err != cudaSuccess) {
        std::string where = label ? label : "unknown";
        throw std::runtime_error("[power] CUDA error at " + where + ": " +
                                 cudaGetErrorString(last_err));
    }
    const char* sync_checks = std::getenv("XPLACE_POWER_CUDA_SYNC_CHECKS");
    if (!sync_checks || sync_checks[0] == '\0' || sync_checks[0] == '0') return;
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::string where = label ? label : "unknown";
        cudaGetLastError();
        throw std::runtime_error("[power] CUDA error at " + where + ": " +
                                 cudaGetErrorString(sync_err));
    }
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
    check_power_cuda_error("chunk activity unpack");
}

void init_power_chunk_activity_storage(int n,
                                       const float* packed_activity,
                                       PowerChunkActivityStorage* storage) {
    if (!storage) return;
    float* density = nullptr;
    float* duty = nullptr;
    unpack_power_activity_density_duty(n, packed_activity, &density, &duty);
    *storage = PowerChunkActivityStorage(density, duty);
}

void free_power_chunk_activity_storage(PowerChunkActivityStorage* storage) {
    if (!storage) return;
    float* density = storage->density;
    float* duty = storage->duty;
    *storage = PowerChunkActivityStorage{};
    if (density) cudaFree(density);
    if (duty) cudaFree(duty);
}

static bool prepare_power_component_scratch(int n,
                                            const float* packed_activity,
                                            const float* activity_density,
                                            const float* activity_duty,
                                            PowerActivityPropDevice& scratch,
                                            float** owned_density,
                                            float** owned_duty) {
    *owned_density = nullptr;
    *owned_duty = nullptr;
    scratch = PowerActivityPropDevice{};
    if (n <= 0) return false;
    if (activity_density && activity_duty) {
        scratch = PowerActivityPropDevice(const_cast<float*>(activity_density),
                                           const_cast<float*>(activity_duty));
        return true;
    }
    if (!packed_activity) return false;
    unpack_power_activity_density_duty(n, packed_activity, owned_density, owned_duty);
    if (!*owned_density || !*owned_duty) return false;
    scratch = PowerActivityPropDevice(*owned_density, *owned_duty);
    return true;
}

static void release_owned_power_component_scratch(float* owned_density,
                                                  float* owned_duty) {
    if (owned_density) cudaFree(owned_density);
    if (owned_duty) cudaFree(owned_duty);
}

void run_power_internal_denom_chunk_cuda_launcher(const PowerInternalDenomDevice& model) {
    int n = model.n;
    const float* d_precomputed_activity = model.precomputed_activity;
    GpuPowerInternalHost* d_internal_rows = model.internal_rows;
    const int num_internal_rows = model.num_internal_rows;
    float* d_denom = model.denom;
    if (n <= 0 || (!d_precomputed_activity && (!model.activity_density || !model.activity_duty)) ||
        !d_internal_rows || num_internal_rows <= 0 || !d_denom)
        return;
    PowerActivityPropDevice scratch;
    float* owned_density = nullptr;
    float* owned_duty = nullptr;
    if (prepare_power_component_scratch(n, d_precomputed_activity,
                                        model.activity_density, model.activity_duty,
                                        scratch, &owned_density, &owned_duty)) {
        power_internal_denom_fast_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            model, scratch);
        check_power_cuda_error("internal denom fast chunk");
        constexpr int POWER_COMPONENT_EXPR_BLOCK_SIZE = 128;
        power_internal_denom_kernel<<<(num_internal_rows + POWER_COMPONENT_EXPR_BLOCK_SIZE - 1) /
                                      POWER_COMPONENT_EXPR_BLOCK_SIZE,
                                      POWER_COMPONENT_EXPR_BLOCK_SIZE>>>(
            model, scratch);
        check_power_cuda_error("internal denom chunk");
    }
    release_owned_power_component_scratch(owned_density, owned_duty);
}

void run_power_internal_contrib_chunk_cuda_launcher(const PowerInternalInstDevice& model) {
    int n = model.n;
    const float* d_precomputed_activity = model.precomputed_activity;
    GpuPowerInternalHost* d_internal_rows = model.internal_rows;
    const int num_internal_rows = model.num_internal_rows;
    const float* d_denom = model.denom;
    GPUPowerLutAllocator* d_power_allocator = model.power_allocator;
    if (n <= 0 || (!d_precomputed_activity && (!model.activity_density || !model.activity_duty)) ||
        !d_internal_rows || num_internal_rows <= 0 || !d_denom || !d_power_allocator)
        return;
    PowerActivityPropDevice scratch;
    float* owned_density = nullptr;
    float* owned_duty = nullptr;
    if (prepare_power_component_scratch(n, d_precomputed_activity,
                                        model.activity_density, model.activity_duty,
                                        scratch, &owned_density, &owned_duty)) {
        power_internal_contrib_fast_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            model, scratch);
        check_power_cuda_error("internal contrib fast chunk");
        constexpr int POWER_COMPONENT_EXPR_BLOCK_SIZE = 128;
        power_internal_contrib_kernel<<<(num_internal_rows + POWER_COMPONENT_EXPR_BLOCK_SIZE - 1) /
                                        POWER_COMPONENT_EXPR_BLOCK_SIZE,
                                        POWER_COMPONENT_EXPR_BLOCK_SIZE>>>(
            model, scratch);
        check_power_cuda_error("internal contrib chunk");
    }
    release_owned_power_component_scratch(owned_density, owned_duty);
}

void run_power_leakage_rows_chunk_cuda_launcher(const PowerLeakageCondDevice& model) {
    const int n = model.n;
    const float* d_precomputed_activity = model.precomputed_activity;
    GpuPowerLeakageRowHost* d_leakage_rows = model.leakage_rows;
    const int num_leakage_rows = model.num_leakage_rows;
    float* d_group_cond_leakage = model.group_cond_leakage;
    float* d_group_cond_duty_sum = model.group_cond_duty_sum;
    int* d_group_cond_count = model.group_cond_count;
    if (n <= 0 || (!d_precomputed_activity && (!model.activity_density || !model.activity_duty)) ||
        !d_leakage_rows || num_leakage_rows <= 0 || !d_group_cond_leakage ||
        !d_group_cond_duty_sum || !d_group_cond_count)
        return;
    PowerActivityPropDevice scratch;
    float* owned_density = nullptr;
    float* owned_duty = nullptr;
    if (prepare_power_component_scratch(n, d_precomputed_activity,
                                        model.activity_density, model.activity_duty,
                                        scratch, &owned_density, &owned_duty)) {
        power_leakage_row_fast_kernel<<<BLOCK_NUMBER(num_leakage_rows), BLOCK_SIZE>>>(
            model);
        check_power_cuda_error("leakage rows fast chunk");
        constexpr int POWER_COMPONENT_EXPR_BLOCK_SIZE = 128;
        power_leakage_row_kernel<<<(num_leakage_rows + POWER_COMPONENT_EXPR_BLOCK_SIZE - 1) /
                                   POWER_COMPONENT_EXPR_BLOCK_SIZE,
                                   POWER_COMPONENT_EXPR_BLOCK_SIZE>>>(
            model, scratch);
        check_power_cuda_error("leakage rows chunk");
    }
    release_owned_power_component_scratch(owned_density, owned_duty);
}

void run_power_leakage_summary_chunk_cuda_launcher(const PowerLeakageInstDevice& model) {
    GpuPowerLeakageGroupHost* d_leakage_groups = model.leakage_groups;
    const int num_leakage_groups = model.num_leakage_groups;
    float* d_group_cond_leakage = model.group_cond_leakage;
    float* d_group_cond_duty_sum = model.group_cond_duty_sum;
    int* d_group_cond_count = model.group_cond_count;
    float* d_inst_leakage = model.inst_leakage;
    if (!d_leakage_groups || num_leakage_groups <= 0 || !d_group_cond_leakage ||
        !d_group_cond_duty_sum || !d_group_cond_count || !d_inst_leakage)
        return;
    power_leakage_summary_kernel<<<BLOCK_NUMBER(num_leakage_groups), BLOCK_SIZE>>>(model);
    check_power_cuda_error("leakage summary chunk");
}

}  // namespace gt
