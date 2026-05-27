#include "gputimer/core/GPUTimer.h"
#include "gputimer/core/DmpModel.h"
#include "gputimer/core/power/PowerCudaModel.h"
#include "gputimer/core/power/PowerHostCommon.h"
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
    torch::Tensor inst_switching_cpu;
    torch::Tensor inst_internal_cpu;
    torch::Tensor inst_leakage_cpu;
    compute_power_activity_cuda(&inst_switching_cpu, nullptr, &inst_internal_cpu,
                                nullptr, nullptr, &inst_leakage_cpu);
    torch::Tensor inst_total_cpu = inst_internal_cpu + inst_switching_cpu + inst_leakage_cpu;
    return {inst_internal_cpu, inst_switching_cpu, inst_leakage_cpu, inst_total_cpu};
}

}  // namespace gt
