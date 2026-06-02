#include "gputimer/core/GPUTimer.h"
#include "gputimer/core/DmpModel.h"
#include "gputimer/core/power/common/PowerCudaModel.h"
#include "gputimer/core/power/common/PowerHostCommon.h"
#include "gputimer/core/power/common/PowerActivityHostUtils.h"
#include "gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "common/lib/Liberty.h"
#include "common/lib/Lut.h"
#include "common/lib/Timing.h"
#include "gputimer/db/GTDatabase.h"
#include "io_parser/gp/GPDatabase.h"

#include <torch/cuda.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gt {


PowerActivityLevelSelection::PowerActivityLevelSelection() = default;

PowerActivityLevelSelection::PowerActivityLevelSelection(torch::Tensor level_list_gpu_,
                                                         std::vector<int> owned_level_list_end_,
                                                         std::vector<int> pin_power_level_,
                                                         const std::vector<int>* external_level_list_end_,
                                                         index_type* level_list_,
                                                         bool use_owned_level_list_end_)
    : level_list_gpu(std::move(level_list_gpu_)),
      owned_level_list_end(std::move(owned_level_list_end_)),
      pin_power_level(std::move(pin_power_level_)),
      external_level_list_end(external_level_list_end_),
      level_list(level_list_),
      use_owned_level_list_end(use_owned_level_list_end_) {}

const std::vector<int>* PowerActivityLevelSelection::levelListEnd() const {
    return use_owned_level_list_end ? &owned_level_list_end : external_level_list_end;
}

PowerCudaRunBuffers::PowerCudaRunBuffers() = default;

PowerCudaRunBuffers::PowerCudaRunBuffers(torch::Tensor out_gpu_,
                                         torch::Tensor inst_switching_gpu_,
                                         torch::Tensor pin_switching_gpu_,
                                         torch::Tensor inst_internal_gpu_,
                                         torch::Tensor internal_row_power_gpu_,
                                         torch::Tensor inst_leakage_gpu_,
                                         torch::Tensor leakage_row_power_gpu_,
                                         torch::Tensor precomputed_activity_cpu_,
                                         torch::Tensor precomputed_activity_gpu_,
                                         float* out_gpu_ptr_,
                                         float* activity_density_ptr_,
                                         float* activity_duty_ptr_,
                                         float* inst_switching_ptr_,
                                         float* pin_switching_ptr_,
                                         float* inst_internal_ptr_,
                                         float* internal_row_power_ptr_,
                                         float* inst_leakage_ptr_,
                                         float* leakage_row_power_ptr_,
                                         const float* precomputed_activity_ptr_,
                                         int out_activity_fields_)
    : out_gpu(std::move(out_gpu_)),
      inst_switching_gpu(std::move(inst_switching_gpu_)),
      pin_switching_gpu(std::move(pin_switching_gpu_)),
      inst_internal_gpu(std::move(inst_internal_gpu_)),
      internal_row_power_gpu(std::move(internal_row_power_gpu_)),
      inst_leakage_gpu(std::move(inst_leakage_gpu_)),
      leakage_row_power_gpu(std::move(leakage_row_power_gpu_)),
      precomputed_activity_cpu(std::move(precomputed_activity_cpu_)),
      precomputed_activity_gpu(std::move(precomputed_activity_gpu_)),
      out_gpu_ptr(out_gpu_ptr_),
      activity_density_ptr(activity_density_ptr_),
      activity_duty_ptr(activity_duty_ptr_),
      inst_switching_ptr(inst_switching_ptr_),
      pin_switching_ptr(pin_switching_ptr_),
      inst_internal_ptr(inst_internal_ptr_),
      internal_row_power_ptr(internal_row_power_ptr_),
      inst_leakage_ptr(inst_leakage_ptr_),
      leakage_row_power_ptr(leakage_row_power_ptr_),
      precomputed_activity_ptr(precomputed_activity_ptr_),
      out_activity_fields(out_activity_fields_) {}

PowerDmpLoadPointers::PowerDmpLoadPointers() = default;

PowerDmpLoadPointers::PowerDmpLoadPointers(const float* C1_, const float* C2_)
    : C1(C1_), C2(C2_) {}

namespace {

void writePowerRowMetaOutputs(torch::Tensor* internal_row_meta_cpu,
                              torch::Tensor* leakage_row_meta_cpu,
                              const std::vector<GpuPowerInternalHost>& internal_rows,
                              const std::vector<GpuPowerLeakageRowHost>& leakage_rows) {
    auto i64opt_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (internal_row_meta_cpu) {
        std::vector<int64_t> meta;
        meta.reserve(internal_rows.size() * 6);
        for (const auto& row : internal_rows) {
            meta.push_back(row.node_id);
            meta.push_back(row.to_pin);
            meta.push_back(row.from_pin);
            meta.push_back(row.kind);
            meta.push_back(row.internal_power_id);
            meta.push_back(row.duty_mode);
        }
        *internal_row_meta_cpu = internal_rows.empty()
            ? torch::empty({0, 6}, i64opt_cpu)
            : torch::from_blob(meta.data(), {(long)internal_rows.size(), 6}, i64opt_cpu).clone();
    }
    if (leakage_row_meta_cpu) {
        std::vector<int64_t> meta;
        meta.reserve(leakage_rows.size() * 4);
        for (const auto& row : leakage_rows) {
            meta.push_back(row.node_id);
            meta.push_back(row.group_id);
            meta.push_back(row.leakage_power_id);
            meta.push_back(row.when_expr_id);
        }
        *leakage_row_meta_cpu = leakage_rows.empty()
            ? torch::empty({0, 4}, i64opt_cpu)
            : torch::from_blob(meta.data(), {(long)leakage_rows.size(), 4}, i64opt_cpu).clone();
    }
}

PowerActivityLevelSelection choosePowerActivityLevels(GTDatabase& gtdb,
                                                      int n,
                                                      const std::vector<int>& power_level_list_end_cpu,
                                                      index_type* power_level_list,
                                                      const std::vector<int>& power_pin_level_cpu,
                                                      const std::vector<int>& level_list_end_cpu,
                                                      index_type* level_list,
                                                      const std::vector<int>& pin_level_cpu) {
    const bool use_cpu_activity_levels_for_power =
        std::getenv("XPLACE_POWER_USE_CPU_ACTIVITY_LEVELS") != nullptr;
    const bool use_timing_levels_for_power =
        std::getenv("XPLACE_POWER_USE_TIMING_LEVELS") != nullptr;
    if (use_cpu_activity_levels_for_power) {
        PowerCpuActivityLevels cpu_activity_levels = buildPowerCpuActivityLevels(gtdb, n);
        torch::Tensor level_list_gpu = powerCudaIntTensor(cpu_activity_levels.level_list);
        index_type* activity_level_list = level_list_gpu.data_ptr<int>();
        return PowerActivityLevelSelection(std::move(level_list_gpu),
                                           std::move(cpu_activity_levels.level_list_end),
                                           std::move(cpu_activity_levels.pin_level),
                                           nullptr,
                                           activity_level_list,
                                           true);
    }
    if (use_timing_levels_for_power) {
        if (!level_list || level_list_end_cpu.empty()) {
            throw std::runtime_error("timing level list is unavailable for power activity");
        }
        std::vector<int> pin_power_level = pin_level_cpu;
        if (static_cast<int>(pin_power_level.size()) != n) pin_power_level.assign(n, -1);
        return PowerActivityLevelSelection(torch::Tensor{},
                                           std::vector<int>{},
                                           std::move(pin_power_level),
                                           &level_list_end_cpu,
                                           level_list,
                                           false);
    }
    std::vector<int> pin_power_level = power_pin_level_cpu;
    if (static_cast<int>(pin_power_level.size()) != n) pin_power_level.assign(n, -1);
    return PowerActivityLevelSelection(torch::Tensor{},
                                       std::vector<int>{},
                                       std::move(pin_power_level),
                                       &power_level_list_end_cpu,
                                       power_level_list,
                                       false);
}

PowerCudaRunBuffers preparePowerCudaRunBuffers(GPUTimer& timer,
                                               int n,
                                               int num_nodes,
                                               bool need_switching_power,
                                               bool want_activity_cpu,
                                               bool chunk_internal_rows,
                                               bool chunk_leakage_rows,
                                               bool output_inst_switching,
                                               bool output_pin_switching,
                                               bool output_inst_internal,
                                               bool output_internal_row_power,
                                               bool output_inst_leakage,
                                               bool output_leakage_row_power,
                                               size_t internal_row_count,
                                               size_t leakage_row_count,
                                               const torch::TensorOptions& fopt_cuda,
                                               PowerStageProfiler& profile) {
    torch::Tensor out_gpu;
    float* out_gpu_ptr = nullptr;
    float* activity_density_ptr = nullptr;
    float* activity_duty_ptr = nullptr;
    int out_activity_fields = 0;
    if (need_switching_power || want_activity_cpu || chunk_internal_rows || chunk_leakage_rows) {
        out_activity_fields = want_activity_cpu ? 3 : 2;
        out_gpu = torch::empty({out_activity_fields, n}, fopt_cuda);
        out_gpu_ptr = out_gpu.data_ptr<float>();
        activity_density_ptr = out_gpu_ptr;
        activity_duty_ptr = out_gpu_ptr + n;
    }

    constexpr int64_t default_cpu_activity_pin_limit = 0;
    const int64_t auto_cpu_activity_pin_limit =
        readPowerEnvInt64("XPLACE_POWER_AUTO_CPU_ACTIVITY_PIN_LIMIT", default_cpu_activity_pin_limit);
    const bool force_cpu_activity_for_power =
        readPowerEnvFlag("XPLACE_POWER_USE_CPU_ACTIVITY_FOR_POWER", false);
    const bool use_cpu_activity_for_power =
        force_cpu_activity_for_power || (auto_cpu_activity_pin_limit > 0 && n <= auto_cpu_activity_pin_limit);
    torch::Tensor precomputed_activity_cpu;
    torch::Tensor precomputed_activity_gpu;
    const float* precomputed_activity_ptr = nullptr;
    if (use_cpu_activity_for_power) {
        precomputed_activity_cpu = timer.report_power_activity_cpu();
        profile.mark(force_cpu_activity_for_power ? "cpu_activity_for_power"
                                                  : "auto_cpu_activity_for_power");
        if (precomputed_activity_cpu.dim() != 2 || precomputed_activity_cpu.size(0) != n ||
            precomputed_activity_cpu.size(1) != 3) {
            throw std::runtime_error("report_power_activity_cpu returned an unexpected activity tensor shape");
        }
        precomputed_activity_gpu = precomputed_activity_cpu.to(torch::kCUDA);
        precomputed_activity_ptr = precomputed_activity_gpu.data_ptr<float>();
    }

    torch::Tensor inst_switching_gpu;
    torch::Tensor pin_switching_gpu;
    torch::Tensor inst_internal_gpu;
    torch::Tensor internal_row_power_gpu;
    torch::Tensor inst_leakage_gpu;
    torch::Tensor leakage_row_power_gpu;
    float* inst_switching_ptr = nullptr;
    float* pin_switching_ptr = nullptr;
    float* inst_internal_ptr = nullptr;
    float* internal_row_power_ptr = nullptr;
    float* inst_leakage_ptr = nullptr;
    float* leakage_row_power_ptr = nullptr;
    if (output_inst_switching) {
        inst_switching_gpu = torch::zeros({num_nodes}, fopt_cuda);
        inst_switching_ptr = inst_switching_gpu.data_ptr<float>();
    }
    if (output_pin_switching) {
        pin_switching_gpu = torch::zeros({n}, fopt_cuda);
        pin_switching_ptr = pin_switching_gpu.data_ptr<float>();
    }
    if (output_inst_internal || output_internal_row_power) {
        inst_internal_gpu = torch::zeros({num_nodes}, fopt_cuda);
        inst_internal_ptr = inst_internal_gpu.data_ptr<float>();
    }
    if (output_internal_row_power) {
        internal_row_power_gpu = torch::zeros({static_cast<long>(internal_row_count)}, fopt_cuda);
        internal_row_power_ptr = internal_row_power_gpu.data_ptr<float>();
    }
    if (output_inst_leakage) {
        inst_leakage_gpu = torch::zeros({num_nodes}, fopt_cuda);
        inst_leakage_ptr = inst_leakage_gpu.data_ptr<float>();
    }
    if (output_leakage_row_power) {
        leakage_row_power_gpu = torch::zeros({static_cast<long>(leakage_row_count)}, fopt_cuda);
        leakage_row_power_ptr = leakage_row_power_gpu.data_ptr<float>();
    }
    return PowerCudaRunBuffers(std::move(out_gpu),
                               std::move(inst_switching_gpu),
                               std::move(pin_switching_gpu),
                               std::move(inst_internal_gpu),
                               std::move(internal_row_power_gpu),
                               std::move(inst_leakage_gpu),
                               std::move(leakage_row_power_gpu),
                               std::move(precomputed_activity_cpu),
                               std::move(precomputed_activity_gpu),
                               out_gpu_ptr,
                               activity_density_ptr,
                               activity_duty_ptr,
                               inst_switching_ptr,
                               pin_switching_ptr,
                               inst_internal_ptr,
                               internal_row_power_ptr,
                               inst_leakage_ptr,
                               leakage_row_power_ptr,
                               precomputed_activity_ptr,
                               out_activity_fields);
}

PowerDmpLoadPointers choosePowerDmpLoadPointers(const DmpModel* h_dmp_db) {
    bool use_dmp_power_load = true;
    if (const char* env = std::getenv("XPLACE_POWER_USE_DMP_LOAD")) {
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        use_dmp_power_load = !(value.empty() || value == "0" || value == "false" || value == "no");
    }
    if (use_dmp_power_load && h_dmp_db && h_dmp_db->C1 && h_dmp_db->C2) {
        return PowerDmpLoadPointers(h_dmp_db->C1, h_dmp_db->C2);
    }
    return PowerDmpLoadPointers{};
}

void runPowerChunkedComponents(int n,
                               int num_nodes,
                               bool chunk_internal_rows,
                               bool chunk_leakage_rows,
                               const PowerCudaRunBuffers& buffers,
                               const std::vector<GpuPowerInternalHost>& internal_rows,
                               const std::unordered_map<uint64_t, int>& internal_denom_group,
                               size_t internal_chunk_bytes,
                               const std::vector<GpuPowerLeakageRowHost>& leakage_rows,
                               const std::vector<GpuPowerLeakageGroupHost>& leakage_groups,
                               size_t leakage_chunk_bytes,
                               GpuPowerLeakageGroupHost* d_leakage_groups_ptr,
                               const torch::Tensor& d_expr_ops,
                               const torch::Tensor& d_expr_start,
                               const torch::Tensor& d_expr_count,
                               const torch::Tensor& d_node_port_pin_start,
                               const torch::Tensor& d_node_port_pin_list,
                               const float* pinSlew,
                               const float* pin_clock_slews,
                               const int* d_power_clock_slew_pins_ptr,
                               int num_power_clock_slew_pins,
                               const float* power_clock_slew_fallback,
                               const PowerDmpLoadPointers& dmp_load,
                               GPUPowerLutAllocator* d_power_allocator,
                               float cap_unit,
                               const torch::TensorOptions& fopt_cuda,
                               const torch::TensorOptions& iopt_cuda) {
    const bool needs_chunk_activity = chunk_internal_rows || chunk_leakage_rows;
    const float* chunk_activity_ptr = buffers.precomputed_activity_ptr;
    const float* chunk_activity_density = buffers.activity_density_ptr;
    const float* chunk_activity_duty = buffers.activity_duty_ptr;
    if (needs_chunk_activity && (!chunk_activity_density || !chunk_activity_duty) && !chunk_activity_ptr) {
        throw std::runtime_error("chunked CUDA power requires activity density/duty or a precomputed activity tensor");
    }

    PowerChunkActivityStorage chunk_activity_storage;
    if (needs_chunk_activity && (!chunk_activity_density || !chunk_activity_duty)) {
        init_power_chunk_activity_storage(n, chunk_activity_ptr, &chunk_activity_storage);
        if (!chunk_activity_storage.density || !chunk_activity_storage.duty) {
            free_power_chunk_activity_storage(&chunk_activity_storage);
            throw std::runtime_error("chunked CUDA power failed to prepare activity density/duty");
        }
        chunk_activity_density = chunk_activity_storage.density;
        chunk_activity_duty = chunk_activity_storage.duty;
    }

    if (chunk_internal_rows && buffers.inst_internal_ptr) {
        const size_t denom_count = std::max<size_t>(1, internal_denom_group.size());
        torch::Tensor internal_denom_gpu = torch::zeros({static_cast<long>(denom_count)}, fopt_cuda);
        const size_t chunk_rows = powerRowsPerChunk(internal_chunk_bytes, sizeof(GpuPowerInternalHost));
        std::fprintf(stderr,
                     "[power_row_chunk] component=internal phase=denom rows=%zu chunk_rows=%zu chunks=%zu\n",
                     internal_rows.size(), chunk_rows,
                     (internal_rows.size() + chunk_rows - 1) / chunk_rows);
        for (size_t begin = 0; begin < internal_rows.size(); begin += chunk_rows) {
            const size_t count = std::min(chunk_rows, internal_rows.size() - begin);
            auto d_rows_chunk = powerCudaBytesTensorRange(internal_rows, begin, count);
            PowerInternalDenomModel denom_model(
                n,
                chunk_activity_ptr,
                chunk_activity_density,
                chunk_activity_duty,
                reinterpret_cast<GpuPowerInternalHost*>(d_rows_chunk.data_ptr<uint8_t>()),
                static_cast<int>(count),
                reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>()),
                d_expr_start.data_ptr<int>(),
                d_expr_count.data_ptr<int>(),
                d_node_port_pin_start.data_ptr<int>(),
                d_node_port_pin_list.data_ptr<int>(),
                internal_denom_gpu.data_ptr<float>());
            run_power_internal_denom_chunk_cuda_launcher(denom_model);
        }
        std::fprintf(stderr,
                     "[power_row_chunk] component=internal phase=contrib rows=%zu chunk_rows=%zu chunks=%zu\n",
                     internal_rows.size(), chunk_rows,
                     (internal_rows.size() + chunk_rows - 1) / chunk_rows);
        for (size_t begin = 0; begin < internal_rows.size(); begin += chunk_rows) {
            const size_t count = std::min(chunk_rows, internal_rows.size() - begin);
            auto d_rows_chunk = powerCudaBytesTensorRange(internal_rows, begin, count);
            float* row_power_ptr = buffers.internal_row_power_ptr
                ? buffers.internal_row_power_ptr + begin
                : nullptr;
            PowerInternalContribModel contrib_model(
                n,
                num_nodes,
                chunk_activity_ptr,
                chunk_activity_density,
                chunk_activity_duty,
                reinterpret_cast<GpuPowerInternalHost*>(d_rows_chunk.data_ptr<uint8_t>()),
                static_cast<int>(count),
                reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>()),
                d_expr_start.data_ptr<int>(),
                d_expr_count.data_ptr<int>(),
                d_node_port_pin_start.data_ptr<int>(),
                d_node_port_pin_list.data_ptr<int>(),
                pinSlew,
                pin_clock_slews,
                d_power_clock_slew_pins_ptr,
                num_power_clock_slew_pins,
                power_clock_slew_fallback,
                dmp_load.C1,
                dmp_load.C2,
                internal_denom_gpu.data_ptr<float>(),
                d_power_allocator,
                cap_unit,
                buffers.inst_internal_ptr,
                row_power_ptr);
            run_power_internal_contrib_chunk_cuda_launcher(contrib_model);
        }
    }

    if (chunk_leakage_rows && buffers.inst_leakage_ptr && d_leakage_groups_ptr) {
        const size_t group_count = std::max<size_t>(1, leakage_groups.size());
        torch::Tensor group_cond_leakage_gpu =
            torch::zeros({static_cast<long>(group_count)}, fopt_cuda);
        torch::Tensor group_cond_duty_sum_gpu =
            torch::zeros({static_cast<long>(group_count)}, fopt_cuda);
        torch::Tensor group_cond_count_gpu =
            torch::zeros({static_cast<long>(group_count)}, iopt_cuda);
        const size_t chunk_rows = powerRowsPerChunk(leakage_chunk_bytes, sizeof(GpuPowerLeakageRowHost));
        std::fprintf(stderr,
                     "[power_row_chunk] component=leakage phase=rows rows=%zu chunk_rows=%zu chunks=%zu\n",
                     leakage_rows.size(), chunk_rows,
                     (leakage_rows.size() + chunk_rows - 1) / chunk_rows);
        for (size_t begin = 0; begin < leakage_rows.size(); begin += chunk_rows) {
            const size_t count = std::min(chunk_rows, leakage_rows.size() - begin);
            auto d_rows_chunk = powerCudaBytesTensorRange(leakage_rows, begin, count);
            float* row_power_ptr = buffers.leakage_row_power_ptr
                ? buffers.leakage_row_power_ptr + begin
                : nullptr;
            PowerLeakageRowsModel rows_model(
                n,
                chunk_activity_ptr,
                chunk_activity_density,
                chunk_activity_duty,
                reinterpret_cast<GpuPowerLeakageRowHost*>(d_rows_chunk.data_ptr<uint8_t>()),
                static_cast<int>(count),
                reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>()),
                d_expr_start.data_ptr<int>(),
                d_expr_count.data_ptr<int>(),
                d_node_port_pin_start.data_ptr<int>(),
                d_node_port_pin_list.data_ptr<int>(),
                group_cond_leakage_gpu.data_ptr<float>(),
                group_cond_duty_sum_gpu.data_ptr<float>(),
                group_cond_count_gpu.data_ptr<int>(),
                row_power_ptr);
            run_power_leakage_rows_chunk_cuda_launcher(rows_model);
        }
        PowerLeakageSummaryModel summary_model(
            d_leakage_groups_ptr,
            static_cast<int>(leakage_groups.size()),
            group_cond_leakage_gpu.data_ptr<float>(),
            group_cond_duty_sum_gpu.data_ptr<float>(),
            group_cond_count_gpu.data_ptr<int>(),
            num_nodes,
            buffers.inst_leakage_ptr);
        run_power_leakage_summary_chunk_cuda_launcher(summary_model);
    }
    free_power_chunk_activity_storage(&chunk_activity_storage);
}

torch::Tensor finishPowerActivityOutputs(torch::Tensor* inst_switching_cpu,
                                         torch::Tensor* pin_switching_cpu,
                                         torch::Tensor* inst_internal_cpu,
                                         torch::Tensor* internal_row_power_cpu,
                                         torch::Tensor* inst_leakage_cpu,
                                         torch::Tensor* leakage_row_power_cpu,
                                         bool output_power_tensors_cuda,
                                         bool want_activity_cpu,
                                         const PowerCudaRunBuffers& buffers,
                                         PowerStageProfiler& profile) {
    if (inst_switching_cpu)
        *inst_switching_cpu = outputPowerTensorForRequest(buffers.inst_switching_gpu, output_power_tensors_cuda);
    if (pin_switching_cpu)
        *pin_switching_cpu = outputPowerTensorForRequest(buffers.pin_switching_gpu, output_power_tensors_cuda);
    if (inst_internal_cpu)
        *inst_internal_cpu = outputPowerTensorForRequest(buffers.inst_internal_gpu, output_power_tensors_cuda);
    if (internal_row_power_cpu)
        *internal_row_power_cpu = outputPowerTensorForRequest(buffers.internal_row_power_gpu, output_power_tensors_cuda);
    if (inst_leakage_cpu)
        *inst_leakage_cpu = outputPowerTensorForRequest(buffers.inst_leakage_gpu, output_power_tensors_cuda);
    if (leakage_row_power_cpu)
        *leakage_row_power_cpu = outputPowerTensorForRequest(buffers.leakage_row_power_gpu, output_power_tensors_cuda);
    profile.mark("downloads");
    if (want_activity_cpu) {
        auto activity_cpu = buffers.out_gpu.to(torch::kCPU);
        return activity_cpu.transpose(0, 1).contiguous();
    }
    return torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
}

}  // namespace

torch::Tensor GPUTimer::compute_power_activity_cuda(torch::Tensor* inst_switching_cpu,
                                                    torch::Tensor* pin_switching_cpu,
                                                    torch::Tensor* inst_internal_cpu,
                                                    torch::Tensor* internal_row_power_cpu,
                                                    torch::Tensor* internal_row_meta_cpu,
                                                    torch::Tensor* inst_leakage_cpu,
                                                    torch::Tensor* leakage_row_power_cpu,
                                                    torch::Tensor* leakage_row_meta_cpu,
                                                    bool output_power_tensors_cuda) {
    const int n = static_cast<int>(gtdb.pin_names.size());
    if (n <= 0) return torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("report_power_activity_cuda requires CUDA");
    }
    // Some existing init kernels leave a stale CUDA error status that CPU reports ignore.
    // Clear it before allocating/uploading the Plan-A power activity data structures.
    clear_power_cuda_error();
    const bool profile_power_stages = readPowerBoolEnv("XPLACE_POWER_PROFILE_STAGES", false);
    if (profile_power_stages) resetPowerStageProfileElapsed();
    PowerStageProfiler profile(profile_power_stages);

    const double sdc_time_scale =
        canonicalPowerTimeScale(gtdb.sdc_time_unit.has_value() ? *gtdb.sdc_time_unit : gtdb.time_unit);
    double min_period_sec = std::numeric_limits<double>::infinity();
    for (auto& kv : gtdb.clocks) {
        const double period_sec = static_cast<double>(kv.second.period()) * sdc_time_scale;
        if (period_sec > 0.0) min_period_sec = std::min(min_period_sec, period_sec);
    }
    if (!std::isfinite(min_period_sec) || min_period_sec <= 0.0) {
        const double fallback_scale = canonicalPowerTimeScale(gtdb.time_unit);
        min_period_sec = fallback_scale > 0.0 ? fallback_scale : 1.0e-9;
    }
    const float default_density = static_cast<float>(0.1 / min_period_sec);
    const float clock_density = static_cast<float>(2.0 / min_period_sec);
    const bool need_switching_power = inst_switching_cpu || pin_switching_cpu;
    const bool need_internal_power =
        inst_internal_cpu || internal_row_power_cpu || internal_row_meta_cpu;
    const bool need_leakage_power =
        inst_leakage_cpu || leakage_row_power_cpu || leakage_row_meta_cpu;
    const bool want_activity_cpu = !inst_switching_cpu && !pin_switching_cpu &&
        !inst_internal_cpu && !internal_row_power_cpu && !internal_row_meta_cpu &&
        !inst_leakage_cpu && !leakage_row_power_cpu && !leakage_row_meta_cpu;

    dumpPowerPinNamesIfRequested(gtdb, n);

    std::vector<int> h_pin_to_node;
    std::vector<int> h_pin_to_net;
    buildPowerPinNodeNetMaps(gtdb, n, h_pin_to_node, h_pin_to_net);

    std::vector<uint8_t> h_is_load_pin;
    std::vector<uint8_t> h_is_driver_pin;
    std::vector<uint8_t> h_is_cell_pin;
    classifyPowerPins(gtdb, n, h_pin_to_node, h_is_load_pin, h_is_driver_pin, &h_is_cell_pin);

    std::vector<int> h_net_driver_pin;
    buildPowerNetDriverPins(gtdb, n, h_is_driver_pin, h_net_driver_pin);

    std::vector<int> h_clock_gate_out_for_input;
    std::vector<int> h_clock_gate_clock_for_out;
    std::vector<int> h_clock_gate_enable_for_out;
    std::vector<uint8_t> h_is_clock_gate_clock_pin;
    buildPowerClockGateMaps(gtdb, n, h_pin_to_node, h_clock_gate_out_for_input,
                             h_clock_gate_clock_for_out, h_clock_gate_enable_for_out,
                             h_is_clock_gate_clock_pin);
    profile.mark("pin_maps");

    PowerClockPinActivity clock_pin_activity =
        buildPowerClockPinActivity(gtdb, n, h_pin_to_node, h_pin_to_net,
                                   h_is_load_pin, h_is_driver_pin,
                                   h_is_clock_gate_clock_pin, sdc_time_scale, clock_density);
    std::vector<int> h_clock_pins = std::move(clock_pin_activity.pins);
    std::vector<float> h_clock_pin_densities = std::move(clock_pin_activity.densities);
    std::vector<float> h_clock_pin_duties = std::move(clock_pin_activity.duties);
    std::vector<uint8_t> h_clock_pin_enqueue = std::move(clock_pin_activity.enqueue);
    profile.mark("clock_pins");

    PowerCudaExprInputs expr_inputs =
        buildPowerCudaExprInputs(gtdb, n, h_is_load_pin, h_is_driver_pin);
    profile.mark("function_exprs");

    PowerCudaSeqInputs seq_inputs =
        buildPowerCudaSeqInputs(gtdb, n, h_pin_to_node, h_is_load_pin, expr_inputs);
    profile.mark("sequentials");

    PowerClockSlewSparse h_power_clock_slews =
        buildPowerClockSlews(gtdb, n, h_clock_pins, seq_inputs.is_seq_clock_input_pin,
                             h_pin_to_net, need_internal_power);

    PowerCudaRootInputs roots =
        buildPowerCudaRootInputs(gtdb, n, h_clock_pins, h_pin_to_node, h_pin_to_net,
                                 h_net_driver_pin, h_is_load_pin, h_is_driver_pin,
                                 seq_inputs, expr_inputs);
    profile.mark("roots");
    std::vector<GpuPowerInternalHost> h_internal_rows;
    std::unordered_map<uint64_t, int> internal_denom_group;
    buildPowerCudaInternalRows(gtdb, n, need_internal_power, h_is_load_pin,
                               h_is_driver_pin, expr_inputs, h_internal_rows,
                               internal_denom_group);

    std::vector<GpuPowerLeakageRowHost> h_leakage_rows;
    std::vector<GpuPowerLeakageGroupHost> h_leakage_groups;
    buildPowerCudaLeakageRows(gtdb, need_leakage_power, expr_inputs,
                              h_leakage_rows, h_leakage_groups);
    profile.mark("component_rows");

    writePowerRowMetaOutputs(internal_row_meta_cpu, leakage_row_meta_cpu,
                             h_internal_rows, h_leakage_rows);
    auto fopt_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto iopt_cuda = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    PowerCudaUploader uploader(readPowerBoolEnv("XPLACE_POWER_UPLOAD_DEBUG", false),
                               readPowerBoolEnv("XPLACE_POWER_UPLOAD_SYNC_DEBUG", false));
    constexpr size_t default_power_row_chunk_bytes = 8ull * 1024ull * 1024ull * 1024ull;
    const size_t internal_row_bytes = h_internal_rows.size() * sizeof(GpuPowerInternalHost);
    const size_t leakage_row_bytes = h_leakage_rows.size() * sizeof(GpuPowerLeakageRowHost);
    const size_t internal_chunk_bytes = readPowerChunkBytes("XPLACE_POWER_INTERNAL_ROW_CHUNK_BYTES",
                                                         readPowerChunkBytes("XPLACE_POWER_ROW_CHUNK_BYTES",
                                                                          default_power_row_chunk_bytes));
    const size_t leakage_chunk_bytes = readPowerChunkBytes("XPLACE_POWER_LEAKAGE_ROW_CHUNK_BYTES",
                                                        readPowerChunkBytes("XPLACE_POWER_ROW_CHUNK_BYTES",
                                                                         default_power_row_chunk_bytes));
    const bool chunk_internal_rows =
        need_internal_power && !h_internal_rows.empty() && internal_row_bytes > internal_chunk_bytes;
    const bool chunk_leakage_rows =
        need_leakage_power && !h_leakage_rows.empty() && leakage_row_bytes > leakage_chunk_bytes;
    printPowerRowStatsIfRequested(h_internal_rows, h_leakage_rows,
                                  internal_row_bytes, leakage_row_bytes,
                                  internal_chunk_bytes, leakage_chunk_bytes,
                                  chunk_internal_rows, chunk_leakage_rows,
                                  internal_denom_group.size(), h_leakage_groups.size(),
                                  expr_inputs.ops.size(), expr_inputs.ops.size() * sizeof(GpuPowerExprOpHost),
                                  expr_inputs.template_expr_cache.size());
    std::vector<int> h_node_port_pin_start;
    std::vector<int> h_node_port_pin_list;
    if (need_internal_power || need_leakage_power) {
        buildPowerNodePortPinMap(gtdb, h_node_port_pin_start, h_node_port_pin_list);
    }
    profile.mark("node_port_pin_map");

    const bool has_power_seq_output_arc_keep =
        std::any_of(roots.seq_output_arc_keep.begin(),
                    roots.seq_output_arc_keep.end(),
                    [](uint32_t word) { return word != 0; });
    torch::Tensor d_power_seq_output_arc_keep;
    const uint32_t* d_power_seq_output_arc_keep_ptr = nullptr;
    if (has_power_seq_output_arc_keep) {
        d_power_seq_output_arc_keep =
            uploader.uploadBytes("power_seq_output_arc_keep", roots.seq_output_arc_keep);
        d_power_seq_output_arc_keep_ptr =
            reinterpret_cast<const uint32_t*>(d_power_seq_output_arc_keep.data_ptr<uint8_t>());
    }
    PowerCudaArcSkipInputs arc_skip_inputs =
        buildPowerCudaArcSkipInputs(gtdb, roots, flat_net2pin_start_map, flat_net2pin_map);
    auto d_arc_skip = uploader.uploadU8("power_arc_skip", arc_skip_inputs.arc_skip);
    auto d_net_driver_pin = uploader.uploadInt("net_driver_pin", h_net_driver_pin);
    auto d_is_load_pin = uploader.uploadU8("is_load_pin", h_is_load_pin);
    auto d_is_driver_pin = uploader.uploadU8("is_driver_pin", h_is_driver_pin);
    auto d_is_cell_pin = uploader.uploadU8("is_cell_pin", h_is_cell_pin);
    auto d_is_seq_output_pin = uploader.uploadU8("is_seq_output_pin", seq_inputs.is_seq_output_pin);
    auto d_is_seq_clock_input_pin = uploader.uploadU8("is_seq_clock_input_pin", seq_inputs.is_seq_clock_input_pin);
    auto d_clock_gate_out_for_input = uploader.uploadInt("clock_gate_out_for_input", h_clock_gate_out_for_input);
    auto d_clock_gate_clock_for_out = uploader.uploadInt("clock_gate_clock_for_out", h_clock_gate_clock_for_out);
    auto d_clock_gate_enable_for_out = uploader.uploadInt("clock_gate_enable_for_out", h_clock_gate_enable_for_out);
    auto d_clock_pins = uploader.uploadInt("clock_pins", h_clock_pins);
    auto d_clock_pin_densities = uploader.uploadFloat("clock_pin_densities", h_clock_pin_densities);
    auto d_clock_pin_duties = uploader.uploadFloat("clock_pin_duties", h_clock_pin_duties);
    auto d_clock_pin_enqueue = uploader.uploadU8("clock_pin_enqueue", h_clock_pin_enqueue);
    torch::Tensor d_power_clock_slew_pins;
    const int* d_power_clock_slew_pins_ptr = nullptr;
    if (!h_power_clock_slews.pins.empty()) {
        d_power_clock_slew_pins = uploader.uploadInt("power_clock_slew_pins", h_power_clock_slews.pins);
        d_power_clock_slew_pins_ptr = d_power_clock_slew_pins.data_ptr<int>();
    }
    auto d_expr_ops = uploader.uploadBytes("expr_ops", expr_inputs.ops);
    auto d_expr_start = uploader.uploadInt("expr_start", expr_inputs.start);
    auto d_expr_count = uploader.uploadInt("expr_count", expr_inputs.count);
    auto d_node_port_pin_start = uploader.uploadInt("node_port_pin_start", h_node_port_pin_start);
    auto d_node_port_pin_list = uploader.uploadInt("node_port_pin_list", h_node_port_pin_list);
    auto d_pin_func_expr_id = uploader.uploadInt("pin_func_expr_id", expr_inputs.pin_func_expr_id);
    auto d_missing_func_out_start = uploader.uploadInt("missing_func_out_start", expr_inputs.missing_func_out_start);
    auto d_missing_func_out_list = uploader.uploadInt("missing_func_out_list", expr_inputs.missing_func_out_list);
    auto d_seqs = uploader.uploadBytes("seqs", seq_inputs.seqs);
    auto d_pin_seq_list_start = uploader.uploadInt("pin_seq_list_start", seq_inputs.pin_seq_list_start);
    auto d_pin_seq_list = uploader.uploadInt("pin_seq_list", seq_inputs.pin_seq_list);
    auto d_feedback_seed_pins = uploader.uploadInt("feedback_seed_pins", roots.feedback_seed_pins);
    auto d_feedback_seed_seqs = uploader.uploadInt("feedback_seed_seqs", roots.feedback_seed_seqs);
    auto h_trace_pins = resolvePowerTracePins(readPowerTracePinQueries(), gtdb.pin_names);
    auto d_trace_pins = uploader.uploadInt("trace_pins", h_trace_pins);
    torch::Tensor d_internal_rows;
    torch::Tensor d_leakage_rows;
    torch::Tensor d_leakage_groups;
    GpuPowerInternalHost* d_internal_rows_ptr = nullptr;
    GpuPowerLeakageRowHost* d_leakage_rows_ptr = nullptr;
    GpuPowerLeakageGroupHost* d_leakage_groups_ptr = nullptr;
    if (need_internal_power && !chunk_internal_rows && !h_internal_rows.empty()) {
        d_internal_rows = uploader.uploadBytes("internal_rows", h_internal_rows);
        d_internal_rows_ptr = reinterpret_cast<GpuPowerInternalHost*>(d_internal_rows.data_ptr<uint8_t>());
    }
    if (need_leakage_power && !chunk_leakage_rows && !h_leakage_rows.empty()) {
        d_leakage_rows = uploader.uploadBytes("leakage_rows", h_leakage_rows);
        d_leakage_rows_ptr = reinterpret_cast<GpuPowerLeakageRowHost*>(d_leakage_rows.data_ptr<uint8_t>());
    }
    if (need_leakage_power && !h_leakage_groups.empty()) {
        d_leakage_groups = uploader.uploadBytes("leakage_groups", h_leakage_groups);
        d_leakage_groups_ptr = reinterpret_cast<GpuPowerLeakageGroupHost*>(d_leakage_groups.data_ptr<uint8_t>());
    }
    profile.mark("uploads");

    // Power-specific CUDA levelization: use the same propagation edge predicate
    // as power_enqueue_adjacent() (skip constraints/tests and sequential Q/Q_N arcs).
    levelize_power(d_is_seq_output_pin.data_ptr<uint8_t>(),
                   arc_types,
                   d_power_seq_output_arc_keep_ptr,
                   d_arc_skip.data_ptr<uint8_t>(),
                   d_is_load_pin.data_ptr<uint8_t>(),
                   pin2net_map,
                   d_net_driver_pin.data_ptr<int>(),
                   arc_skip_inputs.flat_net2pin_start_map,
                   arc_skip_inputs.flat_net2pin_map);
    if (!power_level_list || power_level_list_end_cpu.empty()) {
        throw std::runtime_error("levelize_power failed to build power level list");
    }
    profile.mark("levelize_power");
    finalizePowerCudaRootInputs(gtdb, n, roots, seq_inputs, h_is_load_pin,
                                 h_is_driver_pin, h_pin_to_node, h_pin_to_net,
                                 power_level_root_pins_cpu, power_pin_level_cpu,
                                 arc_skip_inputs.arc_skip);
    auto d_primary_inputs = powerCudaIntTensor(roots.primary_inputs);
    profile.mark("root_upload");


    PowerActivityLevelSelection activity_levels =
        choosePowerActivityLevels(gtdb, n, power_level_list_end_cpu, power_level_list,
                                  power_pin_level_cpu, level_list_end_cpu, level_list,
                                  pin_level_cpu);
    auto d_pin_power_level = powerCudaIntTensor(activity_levels.pin_power_level);
    profile.mark("activity_levels");

    int max_activity_passes = 50;
    if (const char* env = std::getenv("XPLACE_POWER_ACTIVITY_MAX_PASSES"))
        max_activity_passes = std::max(1, std::atoi(env));
    const float min_activity_density =
        std::max(0.0f, readPowerFloatEnv("XPLACE_POWER_MIN_ACTIVITY_DENSITY", 1.0e-10f));

    PowerCudaRunBuffers run_buffers =
        preparePowerCudaRunBuffers(*this, n, num_nodes, need_switching_power,
                                   want_activity_cpu, chunk_internal_rows, chunk_leakage_rows,
                                   inst_switching_cpu != nullptr, pin_switching_cpu != nullptr,
                                   inst_internal_cpu != nullptr, internal_row_power_cpu != nullptr,
                                   inst_leakage_cpu != nullptr, leakage_row_power_cpu != nullptr,
                                   h_internal_rows.size(), h_leakage_rows.size(),
                                   fopt_cuda, profile);
    const float power_voltage = powerVoltageForReport(gtdb);
    const PowerDmpLoadPointers dmp_load = choosePowerDmpLoadPointers(h_dmp_db);

    GpuPowerInternalHost* launcher_internal_rows_ptr =
        chunk_internal_rows ? nullptr : d_internal_rows_ptr;
    const int launcher_internal_row_count =
        chunk_internal_rows ? 0 : static_cast<int>(h_internal_rows.size());
    float* launcher_inst_internal_ptr =
        chunk_internal_rows ? nullptr : run_buffers.inst_internal_ptr;
    float* launcher_internal_row_power_ptr =
        chunk_internal_rows ? nullptr : run_buffers.internal_row_power_ptr;
    GpuPowerLeakageRowHost* launcher_leakage_rows_ptr =
        chunk_leakage_rows ? nullptr : d_leakage_rows_ptr;
    const int launcher_leakage_row_count =
        chunk_leakage_rows ? 0 : static_cast<int>(h_leakage_rows.size());
    GpuPowerLeakageGroupHost* launcher_leakage_groups_ptr =
        chunk_leakage_rows ? nullptr : d_leakage_groups_ptr;
    const int launcher_leakage_group_count =
        chunk_leakage_rows ? 0 : static_cast<int>(h_leakage_groups.size());
    float* launcher_inst_leakage_ptr =
        chunk_leakage_rows ? nullptr : run_buffers.inst_leakage_ptr;
    float* launcher_leakage_row_power_ptr =
        chunk_leakage_rows ? nullptr : run_buffers.leakage_row_power_ptr;

    PowerGraphDeviceView graph_view(
        activity_levels.level_list,
        d_pin_power_level.data_ptr<int>(),
        pin_forward_arc_list_end,
        pin_forward_arc_list,
        timing_arc_to_pin_id,
        arc_types,
        d_power_seq_output_arc_keep_ptr,
        d_arc_skip.data_ptr<uint8_t>(),
        pin2net_map,
        d_net_driver_pin.data_ptr<int>(),
        arc_skip_inputs.flat_net2pin_start_map,
        arc_skip_inputs.flat_net2pin_map,
        d_is_load_pin.data_ptr<uint8_t>(),
        d_is_driver_pin.data_ptr<uint8_t>(),
        d_is_cell_pin.data_ptr<uint8_t>(),
        d_is_seq_output_pin.data_ptr<uint8_t>(),
        d_is_seq_clock_input_pin.data_ptr<uint8_t>(),
        d_clock_gate_out_for_input.data_ptr<int>(),
        d_clock_gate_clock_for_out.data_ptr<int>(),
        d_clock_gate_enable_for_out.data_ptr<int>(),
        pin2node_map,
        pinLoad,
        dmp_load.C1,
        dmp_load.C2,
        pinSlew,
        pin_clock_slews,
        d_power_clock_slew_pins_ptr,
        static_cast<int>(h_power_clock_slews.pins.size()),
        h_power_clock_slews.fallback.data(),
        num_nodes);
    PowerExprDeviceView expr_view(
        reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>()),
        d_expr_start.data_ptr<int>(),
        d_expr_count.data_ptr<int>(),
        d_node_port_pin_start.data_ptr<int>(),
        d_node_port_pin_list.data_ptr<int>(),
        d_pin_func_expr_id.data_ptr<int>(),
        d_missing_func_out_start.data_ptr<int>(),
        d_missing_func_out_list.data_ptr<int>());
    PowerActivityState activity_state(
        d_primary_inputs.data_ptr<int>(),
        static_cast<int>(roots.primary_inputs.size()),
        nullptr,
        d_clock_pins.data_ptr<int>(),
        static_cast<int>(h_clock_pins.size()),
        d_clock_pin_densities.data_ptr<float>(),
        d_clock_pin_duties.data_ptr<float>(),
        d_clock_pin_enqueue.data_ptr<uint8_t>(),
        reinterpret_cast<GpuPowerSeqHost*>(d_seqs.data_ptr<uint8_t>()),
        static_cast<int>(seq_inputs.seqs.size()),
        d_pin_seq_list_start.data_ptr<int>(),
        d_pin_seq_list.data_ptr<int>(),
        d_feedback_seed_pins.data_ptr<int>(),
        static_cast<int>(roots.feedback_seed_pins.size()),
        d_feedback_seed_seqs.data_ptr<int>(),
        static_cast<int>(roots.feedback_seed_seqs.size()));
    PowerActivityConfig activity_config(
        default_density,
        clock_density,
        gtdb.time_unit,
        max_activity_passes,
        d_trace_pins.data_ptr<int>(),
        static_cast<int>(h_trace_pins.size()),
        run_buffers.precomputed_activity_ptr,
        readPowerBoolEnv("XPLACE_POWER_ALLOW_CLOCK_ACTIVITY_OVERRIDE", false),
        min_activity_density);
    PowerComponentDeviceView component_view(
        launcher_internal_rows_ptr,
        launcher_internal_row_count,
        static_cast<int>(internal_denom_group.size()),
        d_power_allocator,
        cap_unit,
        power_voltage,
        run_buffers.inst_switching_ptr,
        run_buffers.pin_switching_ptr,
        launcher_inst_internal_ptr,
        launcher_internal_row_power_ptr,
        launcher_leakage_rows_ptr,
        launcher_leakage_row_count,
        launcher_leakage_groups_ptr,
        launcher_leakage_group_count,
        launcher_inst_leakage_ptr,
        launcher_leakage_row_power_ptr);
    PowerActivityCudaModel activity_model(n,
                                          activity_levels.levelListEnd(),
                                          graph_view,
                                          expr_view,
                                          activity_state,
                                          activity_config,
                                          component_view,
                                          run_buffers.out_gpu_ptr,
                                          run_buffers.out_activity_fields);
    profile.mark("launcher_prepare");
    run_power_activity_cuda_launcher(activity_model);
    profile.mark("launcher");

    runPowerChunkedComponents(n,
                               num_nodes,
                               chunk_internal_rows,
                               chunk_leakage_rows,
                               run_buffers,
                               h_internal_rows,
                               internal_denom_group,
                               internal_chunk_bytes,
                               h_leakage_rows,
                               h_leakage_groups,
                               leakage_chunk_bytes,
                               d_leakage_groups_ptr,
                               d_expr_ops,
                               d_expr_start,
                               d_expr_count,
                               d_node_port_pin_start,
                               d_node_port_pin_list,
                               pinSlew,
                               pin_clock_slews,
                               d_power_clock_slew_pins_ptr,
                               static_cast<int>(h_power_clock_slews.pins.size()),
                               h_power_clock_slews.fallback.data(),
                               dmp_load,
                               d_power_allocator,
                               cap_unit,
                               fopt_cuda,
                               iopt_cuda);
    profile.mark("chunk_components");

    return finishPowerActivityOutputs(inst_switching_cpu,
                                      pin_switching_cpu,
                                      inst_internal_cpu,
                                      internal_row_power_cpu,
                                      inst_leakage_cpu,
                                      leakage_row_power_cpu,
                                      output_power_tensors_cuda,
                                      want_activity_cpu,
                                      run_buffers,
                                      profile);
}


}  // namespace gt
