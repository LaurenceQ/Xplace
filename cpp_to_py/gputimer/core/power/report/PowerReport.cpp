#include "gputimer/core/GPUTimer.h"
#include "gputimer/core/DmpModel.h"
#include "gputimer/core/power/common/PowerCudaModel.h"
#include "gputimer/core/power/common/PowerHostCommon.h"
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
#include <vector>

namespace gt {

torch::Tensor GPUTimer::report_power_activity_cuda() {
    return compute_power_activity_cuda(nullptr, nullptr);
}

tuple<torch::Tensor, torch::Tensor> GPUTimer::report_power_switching_cuda() {
    torch::Tensor inst_switching_cpu;
    torch::Tensor pin_switching_cpu;
    compute_power_activity_cuda(&inst_switching_cpu, &pin_switching_cpu);
    return {inst_switching_cpu, pin_switching_cpu};
}

torch::Tensor GPUTimer::report_power_internal_cuda() {
    torch::Tensor inst_internal_cpu;
    compute_power_activity_cuda(nullptr, nullptr, &inst_internal_cpu);
    return inst_internal_cpu;
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> GPUTimer::report_power_internal_arcs_cuda() {
    torch::Tensor inst_internal_cpu;
    torch::Tensor internal_row_power_cpu;
    torch::Tensor internal_row_meta_cpu;
    compute_power_activity_cuda(nullptr, nullptr, &inst_internal_cpu, &internal_row_power_cpu, &internal_row_meta_cpu);
    return {inst_internal_cpu, internal_row_power_cpu, internal_row_meta_cpu};
}

torch::Tensor GPUTimer::report_power_leakage_cuda() {
    torch::Tensor inst_leakage_cpu;
    compute_power_activity_cuda(nullptr, nullptr, nullptr, nullptr, nullptr, &inst_leakage_cpu);
    return inst_leakage_cpu;
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> GPUTimer::report_power_leakage_rows_cuda() {
    torch::Tensor inst_leakage_cpu;
    torch::Tensor leakage_row_power_cpu;
    torch::Tensor leakage_row_meta_cpu;
    compute_power_activity_cuda(nullptr, nullptr, nullptr, nullptr, nullptr, &inst_leakage_cpu, &leakage_row_power_cpu, &leakage_row_meta_cpu);
    return {inst_leakage_cpu, leakage_row_power_cpu, leakage_row_meta_cpu};
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GPUTimer::report_power_total_cuda() {
    const bool profile_power_stages = readPowerBoolEnv("XPLACE_POWER_PROFILE_STAGES", false);
    const auto report_start = std::chrono::steady_clock::now();
    torch::Tensor inst_switching_gpu;
    torch::Tensor inst_internal_gpu;
    torch::Tensor inst_leakage_gpu;
    compute_power_activity_cuda(&inst_switching_gpu, nullptr, &inst_internal_gpu,
                                nullptr, nullptr, &inst_leakage_gpu,
                                nullptr, nullptr, true);
    const auto total_sum_start = std::chrono::steady_clock::now();
    torch::Tensor inst_total_gpu = inst_internal_gpu + inst_switching_gpu + inst_leakage_gpu;
    if (inst_total_gpu.is_cuda()) {
        torch::cuda::synchronize();
    }
    if (profile_power_stages) {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - total_sum_start).count();
        addPowerStageProfileElapsed(elapsed);
        std::fprintf(stderr, "[power_stage_profile] total_gpu_sum %.6f\n", elapsed);
        const double report_elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - report_start).count();
        const double unprofiled = report_elapsed - powerStageProfileElapsed();
        if (unprofiled > 1.0e-6) {
            addPowerStageProfileElapsed(unprofiled);
            std::fprintf(stderr, "[power_stage_profile] report_total_unprofiled %.6f\n", unprofiled);
        }
    }
    return {inst_internal_gpu, inst_switching_gpu, inst_leakage_gpu, inst_total_gpu};
}

}  // namespace gt
