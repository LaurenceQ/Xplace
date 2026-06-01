

#include "GPUTimer.h"
#include "gputimer/db/GTDatabase.h"
#include "common/lib/Timing.h"
#include "gputiming.h"
#include "utils.cuh"

#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gt {
void release_dmp_forward_schedule_cuda(void* schedule_ptr);

bool gputimer_env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

static double gputimer_gib(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

void gputimer_log_cuda_mem_info(const char* label)
{
    if (!gputimer_env_enabled("GPUTIMER_MEM_PROFILE")) {
        return;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess) {
        std::fprintf(stderr,
                     "[GPUTIMER MEM] %s cuda_free=%.3f GiB cuda_used=%.3f GiB cuda_total=%.3f GiB\n",
                     label,
                     gputimer_gib(free_bytes),
                     gputimer_gib(total_bytes - free_bytes),
                     gputimer_gib(total_bytes));
    } else {
        std::fprintf(stderr,
                     "[GPUTIMER MEM] %s cudaMemGetInfo failed: %s\n",
                     label,
                     cudaGetErrorString(err));
        cudaGetLastError();
    }
}

void gputimer_empty_cuda_cache(const char* label)
{
    if (!gputimer_env_enabled("GPUTIMER_EMPTY_CACHE_AFTER_GTDB")) {
        return;
    }
    gputimer_log_cuda_mem_info(label);
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::string after_label(label);
    after_label += " after_empty_cache";
    gputimer_log_cuda_mem_info(after_label.c_str());
}

#define CUDA_CHECK(msg) do { \
    cudaDeviceSynchronize(); \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) \
        printf("[GPUTIMER INIT] CUDA error at %s (line %d): %s\n", msg, __LINE__, cudaGetErrorString(_e)); \
} while(0)

namespace {

bool dmp_profile_luts_enabled()
{
    const char* value = std::getenv("DMP_PROFILE_LUTS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool host_transition_defined_for_lut_profile(const GPULutAllocator* allocator,
                                             int timing_id,
                                             int input_rf,
                                             int output_rf)
{
    if (allocator == nullptr || timing_id < 0 || timing_id >= allocator->num_timings) {
        return false;
    }
    if (allocator->is_rising_edge_triggered[timing_id] && input_rf != 0) {
        return false;
    }
    if (allocator->is_falling_edge_triggered[timing_id] && input_rf != 1) {
        return false;
    }
    const int sense = allocator->timing_sense[timing_id];
    if (sense == 1 && input_rf != output_rf) {
        return false;
    }
    if (sense == 2 && input_rf == output_rf) {
        return false;
    }
    return true;
}

const char* dmp_lut_type_name(int type)
{
    switch (type) {
        case 0:
            return "delay";
        case 1:
            return "slew";
        case 2:
            return "constraint";
        default:
            return "unknown";
    }
}

void maybe_profile_dmp_gate_lut_usage(const GPUTimer* timer)
{
    if (!dmp_profile_luts_enabled() || timer == nullptr || timer->allocator == nullptr) {
        return;
    }

    const GPULutAllocator* allocator = timer->allocator;
    std::vector<unsigned long long> lut_counts(std::max(allocator->num_luts, 0), 0ULL);
    std::vector<unsigned long long> timing_counts(std::max(allocator->num_timings, 0), 0ULL);
    std::map<std::tuple<int, int, int, int>, unsigned long long> dim_counts;
    std::map<std::tuple<int, int, int, int>, unsigned long long> dim_unique_luts;
    std::vector<unsigned char> lut_seen(std::max(allocator->num_luts, 0), 0);

    unsigned long long gate_arcs = 0;
    unsigned long long total_lanes = 0;
    unsigned long long valid_lanes = 0;
    unsigned long long invalid_timing_lanes = 0;
    unsigned long long invalid_transition_lanes = 0;
    unsigned long long missing_lut_refs = 0;

    for (int arc_id = 0; arc_id < timer->num_arcs; ++arc_id) {
        if (timer->gtdb.arc_types[arc_id] != 1) {
            continue;
        }
        ++gate_arcs;
        for (int lane = 0; lane < 8; ++lane) {
            ++total_lanes;
            const int el = lane >> 2;
            const int from_attr = lane >> 1;
            const int to_attr = ((lane & 0b100) >> 1) + (lane & 1);
            const int input_rf = from_attr & 1;
            const int output_rf = to_attr & 1;
            const int timing_id = timer->gtdb.timing_arc_id_map[arc_id * 2 + el];
            if (timing_id < 0 || timing_id >= allocator->num_timings) {
                ++invalid_timing_lanes;
                continue;
            }
            if (!host_transition_defined_for_lut_profile(allocator, timing_id, input_rf, output_rf)) {
                ++invalid_transition_lanes;
                continue;
            }
            ++valid_lanes;
            timing_counts[timing_id] += 2ULL;
            for (int type = 0; type <= 1; ++type) {
                const int lut_id = allocator->num_luts_in_timing * timing_id + output_rf + type * 2;
                if (lut_id < 0 || lut_id >= allocator->num_luts || !allocator->allocated[lut_id]) {
                    ++missing_lut_refs;
                    continue;
                }
                ++lut_counts[lut_id];
                const auto dim_key = std::make_tuple(allocator->num_x[lut_id],
                                                     allocator->num_y[lut_id],
                                                     allocator->num_table[lut_id],
                                                     allocator->lut_template_var[lut_id * 2]);
                ++dim_counts[dim_key];
                if (!lut_seen[lut_id]) {
                    lut_seen[lut_id] = 1;
                    ++dim_unique_luts[dim_key];
                }
            }
        }
    }

    const unsigned long long total_lut_refs =
        std::accumulate(lut_counts.begin(), lut_counts.end(), 0ULL);

    std::vector<int> lut_order;
    lut_order.reserve(lut_counts.size());
    for (int lut_id = 0; lut_id < static_cast<int>(lut_counts.size()); ++lut_id) {
        if (lut_counts[lut_id] > 0) {
            lut_order.push_back(lut_id);
        }
    }
    std::sort(lut_order.begin(), lut_order.end(), [&](int lhs, int rhs) {
        if (lut_counts[lhs] != lut_counts[rhs]) {
            return lut_counts[lhs] > lut_counts[rhs];
        }
        return lhs < rhs;
    });

    std::vector<int> timing_order;
    timing_order.reserve(timing_counts.size());
    for (int timing_id = 0; timing_id < static_cast<int>(timing_counts.size()); ++timing_id) {
        if (timing_counts[timing_id] > 0) {
            timing_order.push_back(timing_id);
        }
    }
    std::sort(timing_order.begin(), timing_order.end(), [&](int lhs, int rhs) {
        if (timing_counts[lhs] != timing_counts[rhs]) {
            return timing_counts[lhs] > timing_counts[rhs];
        }
        return lhs < rhs;
    });

    auto cumulative_fraction = [&](const std::vector<int>& order, int top_n) {
        unsigned long long sum = 0;
        const int n = std::min(top_n, static_cast<int>(order.size()));
        for (int i = 0; i < n; ++i) {
            sum += lut_counts[order[i]];
        }
        return total_lut_refs == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(total_lut_refs);
    };

    printf("[DMP LUT PROFILE] gate_arcs=%llu total_lanes=%llu valid_lanes=%llu invalid_timing=%llu invalid_transition=%llu total_lut_refs=%llu unique_luts=%zu missing_lut_refs=%llu top1=%.6f top5=%.6f top10=%.6f top32=%.6f\n",
           gate_arcs,
           total_lanes,
           valid_lanes,
           invalid_timing_lanes,
           invalid_transition_lanes,
           total_lut_refs,
           lut_order.size(),
           missing_lut_refs,
           cumulative_fraction(lut_order, 1),
           cumulative_fraction(lut_order, 5),
           cumulative_fraction(lut_order, 10),
           cumulative_fraction(lut_order, 32));

    const char* out_prefix_env = std::getenv("DMP_LUT_PROFILE_OUT");
    if (out_prefix_env == nullptr || out_prefix_env[0] == '\0') {
        return;
    }

    const std::filesystem::path prefix(out_prefix_env);
    if (prefix.has_parent_path()) {
        std::filesystem::create_directories(prefix.parent_path());
    }

    {
        std::ofstream out(prefix.string() + ".summary.txt");
        out << "gate_arcs," << gate_arcs << "\n";
        out << "total_lanes," << total_lanes << "\n";
        out << "valid_lanes," << valid_lanes << "\n";
        out << "invalid_timing_lanes," << invalid_timing_lanes << "\n";
        out << "invalid_transition_lanes," << invalid_transition_lanes << "\n";
        out << "total_lut_refs," << total_lut_refs << "\n";
        out << "unique_luts," << lut_order.size() << "\n";
        out << "missing_lut_refs," << missing_lut_refs << "\n";
        out << "top1_frac," << cumulative_fraction(lut_order, 1) << "\n";
        out << "top5_frac," << cumulative_fraction(lut_order, 5) << "\n";
        out << "top10_frac," << cumulative_fraction(lut_order, 10) << "\n";
        out << "top32_frac," << cumulative_fraction(lut_order, 32) << "\n";
    }

    {
        std::ofstream out(prefix.string() + ".top_luts.csv");
        out << "rank,lut_id,timing_id,type,output_rf,num_x,num_y,num_table,var0,count,frac\n";
        for (int rank = 0; rank < static_cast<int>(lut_order.size()); ++rank) {
            const int lut_id = lut_order[rank];
            const int timing_id = lut_id / allocator->num_luts_in_timing;
            const int in_timing_lut = lut_id % allocator->num_luts_in_timing;
            const int type = in_timing_lut / 2;
            const int output_rf = in_timing_lut & 1;
            const double frac = total_lut_refs == 0 ? 0.0 :
                static_cast<double>(lut_counts[lut_id]) / static_cast<double>(total_lut_refs);
            out << (rank + 1) << ','
                << lut_id << ','
                << timing_id << ','
                << dmp_lut_type_name(type) << ','
                << output_rf << ','
                << allocator->num_x[lut_id] << ','
                << allocator->num_y[lut_id] << ','
                << allocator->num_table[lut_id] << ','
                << allocator->lut_template_var[lut_id * 2] << ','
                << lut_counts[lut_id] << ','
                << frac << "\n";
        }
    }

    {
        std::vector<std::pair<std::tuple<int, int, int, int>, unsigned long long>> dims(dim_counts.begin(), dim_counts.end());
        std::sort(dims.begin(), dims.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
            }
            return lhs.first < rhs.first;
        });
        std::ofstream out(prefix.string() + ".dims.csv");
        out << "num_x,num_y,num_table,var0,refs,unique_luts,frac\n";
        for (const auto& item : dims) {
            const auto& key = item.first;
            const double frac = total_lut_refs == 0 ? 0.0 :
                static_cast<double>(item.second) / static_cast<double>(total_lut_refs);
            out << std::get<0>(key) << ','
                << std::get<1>(key) << ','
                << std::get<2>(key) << ','
                << std::get<3>(key) << ','
                << item.second << ','
                << dim_unique_luts[key] << ','
                << frac << "\n";
        }
    }

    {
        std::ofstream out(prefix.string() + ".top_timings.csv");
        out << "rank,timing_id,lut_refs,frac\n";
        for (int rank = 0; rank < static_cast<int>(timing_order.size()); ++rank) {
            const int timing_id = timing_order[rank];
            const double frac = total_lut_refs == 0 ? 0.0 :
                static_cast<double>(timing_counts[timing_id]) / static_cast<double>(total_lut_refs);
            out << (rank + 1) << ','
                << timing_id << ','
                << timing_counts[timing_id] << ','
                << frac << "\n";
        }
    }
}

}  // namespace

void GPUTimer::initialize() {
    CUDA_CHECK("entry");
    gputimer_log_cuda_mem_info("GPUTimer::initialize before_cuda_mallocs");
    __pinSlew__ = nullptr;
    __pinLoad__ = nullptr;
    __pinRAT__ = nullptr;
    __pinAT__ = nullptr;
    pinWireCap = nullptr;
    pinRootRes = nullptr;
    cudaMalloc(&pinCap, num_pins * (NUM_ATTR + 2) * sizeof(float));
    if (!gtdb.skip_legacy_rc_tensors) {
        cudaMalloc(&pinWireCap, num_pins * NUM_ATTR * sizeof(float));
    }
    cudaMalloc(&testRelatedAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testRAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testConstraint, num_tests * NUM_ATTR * sizeof(float));
    if (!gtdb.skip_legacy_rc_tensors) {
        cudaMalloc(&pinRootRes, num_pins * NUM_ATTR * sizeof(float));
    }

    cudaMalloc(&net_is_clock, num_nets * sizeof(int));
    cudaMalloc(&pin_is_clk, num_pins * sizeof(int));
    cudaMalloc(&pin_is_ideal_clk, num_pins * sizeof(int));
    cudaMalloc(&level_list, num_pins * sizeof(int));
    cudaMalloc(&primary_outputs, num_POs * sizeof(index_type));
    gputimer_log_cuda_mem_info("GPUTimer::initialize after_core_cuda_mallocs");

    cudaMemcpy(pinCap, gtdb.pin_capacitance.data(), num_pins * (NUM_ATTR + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net_is_clock, gtdb.net_is_clock.data(), num_nets * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_is_clk, gtdb.pin_is_clk.data(), num_pins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_is_ideal_clk, gtdb.pin_is_ideal_clk.data(), num_pins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(primary_outputs, gtdb.primary_outputs.data(), gtdb.primary_outputs.size() * sizeof(index_type), cudaMemcpyHostToDevice);


    allocator = new GPULutAllocator();
    allocator->AllocateBatch(gtdb.liberty_timing_arcs);
    allocator->CopyToGPU();
    cudaMalloc((void **)&d_allocator, sizeof(GPULutAllocator));
    cudaMemcpy(d_allocator, allocator, sizeof(GPULutAllocator), cudaMemcpyHostToDevice);
    allocator->CopyToGPU(d_allocator);
    maybe_profile_dmp_gate_lut_usage(this);
    gputimer_log_cuda_mem_info("GPUTimer::initialize after_lut_gpu_copy");

    power_allocator = new GPUPowerLutAllocator();
    power_allocator->AllocateBatch(gtdb.liberty_internal_powers);
    power_allocator->CopyToGPU();
    cudaMalloc((void **)&d_power_allocator, sizeof(GPUPowerLutAllocator));
    cudaMemcpy(d_power_allocator, power_allocator, sizeof(GPUPowerLutAllocator), cudaMemcpyHostToDevice);
    power_allocator->CopyToGPU(d_power_allocator);
    gputimer_log_cuda_mem_info("GPUTimer::initialize after_power_lut_gpu_copy");

    logger.info("GPUTimer initialized");

    if (!gputimer_env_enabled("GPUTIMER_DISABLE_STATE_BACKUP_TENSORS")) {
        cudaMalloc(&__pinSlew__, num_pins * NUM_ATTR * sizeof(float));
        cudaMalloc(&__pinLoad__, num_pins * NUM_ATTR * sizeof(float));
        cudaMalloc(&__pinRAT__, num_pins * NUM_ATTR * sizeof(float));
        cudaMalloc(&__pinAT__, num_pins * NUM_ATTR * sizeof(float));

        device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinSlew, __pinSlew__, num_pins * NUM_ATTR);
        device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinLoad, __pinLoad__, num_pins * NUM_ATTR);
        device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinRAT, __pinRAT__, num_pins * NUM_ATTR);
        device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinAT, __pinAT__, num_pins * NUM_ATTR);
    }
    gputimer_log_cuda_mem_info("GPUTimer::initialize after_state_backups");
    CUDA_CHECK("exit");
}

GPUTimer::~GPUTimer() {
    logger.info("destruct GPUTimer");

    if (dmp_forward_schedule != nullptr) {
        release_dmp_forward_schedule_cuda(dmp_forward_schedule);
        dmp_forward_schedule = nullptr;
    }

    cudaFree(pinCap);
    cudaFree(pinWireCap);
    cudaFree(testRelatedAT);
    cudaFree(testRAT);
    cudaFree(testConstraint);
    cudaFree(pinRootRes);

    cudaFree(net_is_clock);
    cudaFree(pin_is_clk);
    cudaFree(pin_is_ideal_clk);
    cudaFree(level_list);
    cudaFree(level_list_end);
    cudaFree(power_level_list);
    cudaFree(power_level_list_end);
    cudaFree(primary_outputs);

    cudaFree(__pinSlew__);
    cudaFree(__pinLoad__);
    cudaFree(__pinRAT__);
    cudaFree(__pinAT__);

    if (allocator) {
        delete allocator;
        allocator = nullptr;
    }
    if (d_allocator) {
        cudaFree(d_allocator);
        d_allocator = nullptr;
    }
    if (power_allocator) {
        delete power_allocator;
        power_allocator = nullptr;
    }
    if (d_power_allocator) {
        cudaFree(d_power_allocator);
        d_power_allocator = nullptr;
    }
}

__global__ void power_internal_lut_probe_kernel(GPUPowerLutAllocator* power_allocator, int n, float* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx * 2 + 0] = power_allocator ? power_allocator->query_internal_power(idx, 0, 0.0f, 0.0f) : nanf("");
    out[idx * 2 + 1] = power_allocator ? power_allocator->query_internal_power(idx, 1, 0.0f, 0.0f) : nanf("");
}

torch::Tensor GPUTimer::report_power_internal_lut_cuda_probe() {
    if (!d_power_allocator) {
        throw std::runtime_error("report_power_internal_lut_cuda_probe requires timer.init() first");
    }
    const int n = static_cast<int>(gtdb.liberty_internal_powers.size());
    auto out = torch::empty({n, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    if (n > 0) {
        power_internal_lut_probe_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(d_power_allocator, n, out.data_ptr<float>());
        CUDA_CHECK("power_internal_lut_probe_kernel");
    }
    return out.to(torch::kCPU);
}

void GPUTimer::update_states() {
    if (pinImpulse != nullptr) cudaMemset(pinImpulse, 0, num_pins * NUM_ATTR * sizeof(float));
    if (pinRootRes != nullptr) cudaMemset(pinRootRes, 0, num_pins * NUM_ATTR * sizeof(float));
    if (pinRootDelay != nullptr) cudaMemset(pinRootDelay, 0, num_pins * NUM_ATTR * sizeof(float));
    if (pinWireCap != nullptr) cudaMemset(pinWireCap, 0, num_pins * NUM_ATTR * sizeof(float));

    reset_val<float><<<BLOCK_NUMBER(2 * num_arcs * NUM_ATTR), BLOCK_SIZE>>>(arcDelay, 2 * num_arcs * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testRelatedAT, num_tests * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testRAT, num_tests * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testConstraint, num_tests * NUM_ATTR);

    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_pin, num_pins * NUM_ATTR);
    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_arc, num_pins * NUM_ATTR);
    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_attr, num_pins * NUM_ATTR);

    if (__pinSlew__ != nullptr) {
        device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinSlew__, pinSlew, num_pins * NUM_ATTR);
        device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinLoad__, pinLoad, num_pins * NUM_ATTR);
        device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinRAT__, pinRAT, num_pins * NUM_ATTR);
        device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinAT__, pinAT, num_pins * NUM_ATTR);
    }
    cudaDeviceSynchronize();
}


}  // namespace gt
