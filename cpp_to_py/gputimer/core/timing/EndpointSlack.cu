#include "gputimer/core/GPUTimer.h"
#include "gputimer/db/GTDatabase.h"
#include "gputimer/db/sdc/SdcUtils.h"
#include "common/lib/Timing.h"
#include "gputimer/core/gputiming.h"
#include "gputimer/core/utils.cuh"

#include <cfloat>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace gt {

bool gputimer_env_enabled(const char* name);

struct EndpointSlackModel {
    float* pinAT = nullptr;
    float* pinRAT = nullptr;
    float* testRAT = nullptr;
    int* test_id2_arc_id = nullptr;
    index_type* timing_arc_from_pin_id = nullptr;
    index_type* timing_arc_to_pin_id = nullptr;
    index_type* primary_outputs = nullptr;
    int* test_id2_endpoint_id = nullptr;
    int* primary_output2_endpoint_id = nullptr;
    float* endpoints0 = nullptr;
    float* endpoints1 = nullptr;
    float* endpoint_pin_slacks = nullptr;
    unsigned long long* debug_counts = nullptr;
    int num_pins = 0;
    int num_arcs = 0;
    int num_tests = 0;
    int num_POs = 0;
    int num_endpoint_pins = 0;
};

static void endpoint_clear_stale_cuda_error(const char* label)
{
    cudaError_t stale_error = cudaGetLastError();
    if (stale_error != cudaSuccess) {
        fprintf(stderr, "[GPUTIMER CUDA] cleared stale CUDA error before %s: %s\n",
                label, cudaGetErrorString(stale_error));
    }
}

static void endpoint_check_cuda_and_clear(const char* label)
{
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        fprintf(stderr, "[GPUTIMER CUDA] %s launch error: %s\n",
                label, cudaGetErrorString(launch_error));
    }
    cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        fprintf(stderr, "[GPUTIMER CUDA] %s sync error: %s\n",
                label, cudaGetErrorString(sync_error));
        cudaGetLastError();
    }
}

static void endpoint_abort_on_cuda_error(cudaError_t error, const char* label)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "[GPUTIMER CUDA] %s failed: %s\n",
                label, cudaGetErrorString(error));
        std::exit(error);
    }
}

__global__ void update_endpoints_kernel0(EndpointSlackModel* model) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int test_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    if (test_idx < model->num_tests) {
        const int arc_id = model->test_id2_arc_id[test_idx];
        if (arc_id < 0 || arc_id >= model->num_arcs) return;
        const int to_pin_id = model->timing_arc_to_pin_id[arc_id];
        if (to_pin_id < 0 || to_pin_id >= model->num_pins) return;
        if (isnan(model->pinAT[to_pin_id * NUM_ATTR + i]) || isnan(model->testRAT[test_idx * NUM_ATTR + i])) return;
        if (el == 0) {
            model->endpoints0[test_idx * NUM_ATTR + i] = model->pinAT[to_pin_id * NUM_ATTR + i] - model->testRAT[test_idx * NUM_ATTR + i];
        } else {
            model->endpoints0[test_idx * NUM_ATTR + i] = model->testRAT[test_idx * NUM_ATTR + i] - model->pinAT[to_pin_id * NUM_ATTR + i];
        }
    }
}

__global__ void update_endpoints_kernel1(EndpointSlackModel* model) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int po_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    if (po_idx < model->num_POs) {
        const int pin_idx = model->primary_outputs[po_idx];
        if (pin_idx < 0 || pin_idx >= model->num_pins) return;
        if (isnan(model->pinAT[pin_idx * NUM_ATTR + i]) || isnan(model->pinRAT[pin_idx * NUM_ATTR + i])) return;
        if (el == 0) {
            model->endpoints1[po_idx * NUM_ATTR + i] = model->pinAT[pin_idx * NUM_ATTR + i] - model->pinRAT[pin_idx * NUM_ATTR + i];
        } else {
            model->endpoints1[po_idx * NUM_ATTR + i] = model->pinRAT[pin_idx * NUM_ATTR + i] - model->pinAT[pin_idx * NUM_ATTR + i];
        }
    }
}

__device__ void atomicMinFloatValue(float *addr, float value) {
    if (!isfinite(value)) {
        return;
    }
    int *addr_as_i = reinterpret_cast<int *>(addr);
    int old = *addr_as_i;
    while (value < __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
}

__global__ void update_endpoint_pin_slacks_kernel0(EndpointSlackModel* model) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int test_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    if (test_idx < model->num_tests) {
        const int arc_id = model->test_id2_arc_id[test_idx];
        if (arc_id < 0 || arc_id >= model->num_arcs) return;
        const int to_pin_id = model->timing_arc_to_pin_id[arc_id];
        if (to_pin_id < 0 || to_pin_id >= model->num_pins) return;
        const int endpoint_id = model->test_id2_endpoint_id[test_idx];
        if (endpoint_id < 0 || endpoint_id >= model->num_endpoint_pins) {
            return;
        }
        const float at = model->pinAT[to_pin_id * NUM_ATTR + i];
        const float rat = model->testRAT[test_idx * NUM_ATTR + i];
        if (!isfinite(at) || !isfinite(rat)) {
            return;
        }
        const float slack = (el == 0) ? (at - rat) : (rat - at);
        atomicMinFloatValue(&model->endpoint_pin_slacks[endpoint_id * NUM_ATTR + i], slack);
    }
}

__global__ void update_endpoint_pin_slacks_kernel1(EndpointSlackModel* model) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int po_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    if (po_idx < model->num_POs) {
        const int pin_idx = model->primary_outputs[po_idx];
        if (pin_idx < 0 || pin_idx >= model->num_pins) return;
        const int endpoint_id = model->primary_output2_endpoint_id[po_idx];
        if (endpoint_id < 0 || endpoint_id >= model->num_endpoint_pins) {
            return;
        }
        const float at = model->pinAT[pin_idx * NUM_ATTR + i];
        const float rat = model->pinRAT[pin_idx * NUM_ATTR + i];
        if (!isfinite(at) || !isfinite(rat)) {
            return;
        }
        const float slack = (el == 0) ? (at - rat) : (rat - at);
        atomicMinFloatValue(&model->endpoint_pin_slacks[endpoint_id * NUM_ATTR + i], slack);
    }
}

__global__ void endpoint_debug_count_kernel(EndpointSlackModel* model) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_total = model->num_pins * NUM_ATTR;
    const int test_total = model->num_tests * NUM_ATTR;
    const int po_total = model->num_POs * NUM_ATTR;
    if (idx < pin_total) {
        if (isfinite(model->pinAT[idx])) atomicAdd(&model->debug_counts[0], 1ULL);
        if (isfinite(model->pinRAT[idx])) atomicAdd(&model->debug_counts[1], 1ULL);
    }
    if (idx < test_total) {
        if (isfinite(model->testRAT[idx])) atomicAdd(&model->debug_counts[2], 1ULL);
        if (isfinite(model->endpoints0[idx])) atomicAdd(&model->debug_counts[3], 1ULL);
    }
    if (idx < po_total && isfinite(model->endpoints1[idx])) {
        atomicAdd(&model->debug_counts[4], 1ULL);
    }
}

void print_endpoint_debug_summary(EndpointSlackModel& model) {
    const int pin_total = model.num_pins * NUM_ATTR;
    const int test_total = model.num_tests * NUM_ATTR;
    const int po_total = model.num_POs * NUM_ATTR;
    const int total = std::max(1, std::max(pin_total, std::max(test_total, po_total)));
    unsigned long long *d_counts = nullptr;
    unsigned long long h_counts[5] = {0, 0, 0, 0, 0};

    cudaMalloc(&d_counts, sizeof(h_counts));
    cudaMemset(d_counts, 0, sizeof(h_counts));
    model.debug_counts = d_counts;
    EndpointSlackModel* d_model = nullptr;
    cudaMalloc(&d_model, sizeof(EndpointSlackModel));
    cudaMemcpy(d_model, &model, sizeof(EndpointSlackModel), cudaMemcpyHostToDevice);
    endpoint_debug_count_kernel<<<BLOCK_NUMBER(total), BLOCK_SIZE>>>(d_model);
    endpoint_check_cuda_and_clear("endpoint_debug_count_kernel");
    cudaMemcpy(h_counts, d_counts, sizeof(h_counts), cudaMemcpyDeviceToHost);
    cudaFree(d_model);
    cudaFree(d_counts);

    printf("[GPUTIMER ENDPOINT DEBUG] pinAT=%llu/%d pinRAT=%llu/%d testRAT=%llu/%d endpoints0=%llu/%d endpoints1=%llu/%d\n",
           h_counts[0], pin_total,
           h_counts[1], pin_total,
           h_counts[2], test_total,
           h_counts[3], test_total,
           h_counts[4], po_total);
}

static const char* debug_timing_type_name(TimingType type) {
    switch (type) {
        case TimingType::clear: return "clear";
        case TimingType::combinational: return "combinational";
        case TimingType::combinational_fall: return "combinational_fall";
        case TimingType::combinational_rise: return "combinational_rise";
        case TimingType::falling_edge: return "falling_edge";
        case TimingType::hold_falling: return "hold_falling";
        case TimingType::hold_rising: return "hold_rising";
        case TimingType::min_pulse_width: return "min_pulse_width";
        case TimingType::minimum_period: return "minimum_period";
        case TimingType::nochange_high_high: return "nochange_high_high";
        case TimingType::nochange_high_low: return "nochange_high_low";
        case TimingType::nochange_low_high: return "nochange_low_high";
        case TimingType::nochange_low_low: return "nochange_low_low";
        case TimingType::non_seq_hold_falling: return "non_seq_hold_falling";
        case TimingType::non_seq_hold_rising: return "non_seq_hold_rising";
        case TimingType::non_seq_setup_falling: return "non_seq_setup_falling";
        case TimingType::non_seq_setup_rising: return "non_seq_setup_rising";
        case TimingType::preset: return "preset";
        case TimingType::recovery_falling: return "recovery_falling";
        case TimingType::recovery_rising: return "recovery_rising";
        case TimingType::removal_falling: return "removal_falling";
        case TimingType::removal_rising: return "removal_rising";
        case TimingType::retaining_time: return "retaining_time";
        case TimingType::rising_edge: return "rising_edge";
        case TimingType::setup_falling: return "setup_falling";
        case TimingType::setup_rising: return "setup_rising";
        case TimingType::skew_falling: return "skew_falling";
        case TimingType::skew_rising: return "skew_rising";
        case TimingType::three_state_disable: return "three_state_disable";
        case TimingType::three_state_disable_fall: return "three_state_disable_fall";
        case TimingType::three_state_disable_rise: return "three_state_disable_rise";
        case TimingType::three_state_enable: return "three_state_enable";
        case TimingType::three_state_enable_fall: return "three_state_enable_fall";
        case TimingType::three_state_enable_rise: return "three_state_enable_rise";
        case TimingType::min_clock_tree_path: return "min_clock_tree_path";
        case TimingType::max_clock_tree_path: return "max_clock_tree_path";
        case TimingType::unknown: return "unknown";
    }
    return "unknown";
}

void GPUTimer::debug_dump_endpoint_tests(const std::string& outfile,
                                         const vector<std::string>& endpoint_pin_names) {
    std::unordered_set<int> endpoint_filter;
    for (const auto& pin_name : endpoint_pin_names) {
        const std::string lookup_key = pin_name_colon_to_slash(pin_name);
        auto iter = gtdb.pin_name2pin_id.find(lookup_key);
        if (iter != gtdb.pin_name2pin_id.end()) {
            endpoint_filter.insert(iter->second);
        }
    }

    std::filesystem::path outpath(outfile);
    if (!outpath.parent_path().empty()) {
        std::filesystem::create_directories(outpath.parent_path());
    }
    std::ofstream out(outfile);
    if (!out.is_open()) {
        logger.error("debug_dump_endpoint_tests: cannot open %s\n", outfile.c_str());
        return;
    }

    const float time_to_ns = time_unit() * 1e9f;
    vector<float> h_pinAT(num_pins * NUM_ATTR, nanf(""));
    vector<float> h_pinRAT(num_pins * NUM_ATTR, nanf(""));
    vector<float> h_pinSlew(num_pins * NUM_ATTR, nanf(""));
    vector<float> h_testRelatedAT(num_tests * NUM_ATTR, nanf(""));
    vector<float> h_testConstraint(num_tests * NUM_ATTR, nanf(""));
    vector<float> h_testRAT(num_tests * NUM_ATTR, nanf(""));

    if (num_pins > 0) {
        cudaMemcpy(h_pinAT.data(), pinAT, sizeof(float) * h_pinAT.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pinRAT.data(), pinRAT, sizeof(float) * h_pinRAT.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pinSlew.data(), pinSlew, sizeof(float) * h_pinSlew.size(), cudaMemcpyDeviceToHost);
    }
    if (num_tests > 0) {
        cudaMemcpy(h_testRelatedAT.data(), testRelatedAT, sizeof(float) * h_testRelatedAT.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_testConstraint.data(), testConstraint, sizeof(float) * h_testConstraint.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_testRAT.data(), testRAT, sizeof(float) * h_testRAT.size(), cudaMemcpyDeviceToHost);
    }
    endpoint_check_cuda_and_clear("debug_dump_endpoint_tests memcpy");

    out << "test_id,arc_id,attr,corner,to_pin_id,to_pin,from_pin_id,from_pin,"
        << "timing_id,timing_type,from_port,to_port,related_port,is_min_constraint,is_max_constraint,"
        << "from_at_ns,from_slew_ns,to_at_ns,to_slew_ns,related_at_ns,constraint_ns,test_rat_ns,"
        << "pin_rat_ns,slack_ns,pin_clock_slew_ns,pin_clock_rise_ns,pin_clock_fall_ns,pin_clock_period_ns,"
        << "test_clock_period_ns,test_setup_uncertainty_ns,test_hold_uncertainty_ns\n";

    const char* attr_names[NUM_ATTR] = {"early-rise", "early-fall", "late-rise", "late-fall"};
    int emitted = 0;
    for (int test_id = 0; test_id < num_tests; ++test_id) {
        if (test_id < 0 || test_id >= static_cast<int>(gtdb.test_id2_arc_id.size())) {
            continue;
        }
        const int arc_id = gtdb.test_id2_arc_id[test_id];
        if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) {
            continue;
        }
        const int to_pin_id = gtdb.timing_arc_to_pin_id[arc_id];
        const int from_pin_id = gtdb.timing_arc_from_pin_id[arc_id];
        if (!endpoint_filter.empty() && !endpoint_filter.count(to_pin_id)) {
            continue;
        }
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            const float rat = h_testRAT[test_id * NUM_ATTR + attr];
            const float related = h_testRelatedAT[test_id * NUM_ATTR + attr];
            const float constraint = h_testConstraint[test_id * NUM_ATTR + attr];
            if (!std::isfinite(rat) && !std::isfinite(related) && !std::isfinite(constraint)) {
                continue;
            }
            const int el = attr >> 1;
            const int timing_id = gtdb.timing_arc_id_map[arc_id * 2 + el];
            TimingArc* timing_arc =
                timing_id >= 0 && timing_id < static_cast<int>(gtdb.liberty_timing_arcs.size())
                    ? gtdb.liberty_timing_arcs[timing_id]
                    : nullptr;
            const float to_at = h_pinAT[to_pin_id * NUM_ATTR + attr];
            const float pin_rat = h_pinRAT[to_pin_id * NUM_ATTR + attr];
            const float slack = el == 0 ? (to_at - rat) : (rat - to_at);
            out << test_id << ',' << arc_id << ',' << attr << ',' << attr_names[attr] << ','
                << to_pin_id << ",\"" << gtdb.pin_names[to_pin_id] << "\","
                << from_pin_id << ",\"" << gtdb.pin_names[from_pin_id] << "\","
                << timing_id << ','
                << (timing_arc ? debug_timing_type_name(timing_arc->timing_type_) : "none") << ','
                << '"' << (timing_arc && timing_arc->from_port_ ? timing_arc->from_port_->name : "") << "\","
                << '"' << (timing_arc && timing_arc->to_port_ ? timing_arc->to_port_->name : "") << "\","
                << '"' << (timing_arc ? timing_arc->related_port_name_ : "") << "\","
                << (timing_arc && timing_arc->is_min_constraint() ? 1 : 0) << ','
                << (timing_arc && timing_arc->is_max_constraint() ? 1 : 0) << ','
                << h_pinAT[from_pin_id * NUM_ATTR + attr] * time_to_ns << ','
                << h_pinSlew[from_pin_id * NUM_ATTR + attr] * time_to_ns << ','
                << to_at * time_to_ns << ','
                << h_pinSlew[to_pin_id * NUM_ATTR + attr] * time_to_ns << ','
                << related * time_to_ns << ','
                << constraint * time_to_ns << ','
                << rat * time_to_ns << ','
                << pin_rat * time_to_ns << ','
                << slack * time_to_ns << ','
                << gtdb.ClockSlewForPin(from_pin_id, attr) << ','
                << gtdb.ClockRiseEdgeForPin(from_pin_id) << ','
                << gtdb.ClockFallEdgeForPin(from_pin_id) << ','
                << gtdb.ClockPeriodForPin(from_pin_id) << ','
                << (test_id >= 0 && test_id < static_cast<int>(gtdb.test_clock_ids.size()) &&
                    gtdb.ClockIdValid(gtdb.test_clock_ids[test_id])
                        ? gtdb.clock_periods[gtdb.test_clock_ids[test_id]]
                        : nanf("")) << ','
                << gtdb.ClockSetupUncertaintyForTest(test_id) << ','
                << gtdb.ClockHoldUncertaintyForTest(test_id) << '\n';
            ++emitted;
        }
    }
    logger.info("debug_dump_endpoint_tests wrote %d rows to %s", emitted, outfile.c_str());
}

void GPUTimer::update_endpoints() {
    endpoint_clear_stale_cuda_error("update_endpoints");
    const auto device = timing_raw_db.pinAT.device();
    int target_device = -1;
    if (device.is_cuda()) {
        target_device = static_cast<int>(timing_raw_db.pinAT.get_device());
        endpoint_abort_on_cuda_error(cudaSetDevice(target_device),
                                     "update_endpoints cudaSetDevice");
    }
    auto float_options = torch::dtype(torch::kFloat32).device(device);

    torch::Tensor endpoints0 = torch::zeros({num_tests, NUM_ATTR}, float_options).contiguous();
    torch::Tensor endpoints1 = torch::zeros({num_POs, NUM_ATTR}, float_options).contiguous();
    torch::fill_(endpoints0, nanf(""));
    torch::fill_(endpoints1, nanf(""));
    endpoint_pin_slacks = torch::full({num_endpoint_pins, NUM_ATTR},
                                      FLT_MAX,
                                      float_options).contiguous();

    EndpointSlackModel model;
    model.pinAT = pinAT;
    model.pinRAT = pinRAT;
    model.testRAT = testRAT;
    model.test_id2_arc_id = test_id2_arc_id;
    model.timing_arc_from_pin_id = timing_arc_from_pin_id;
    model.timing_arc_to_pin_id = timing_arc_to_pin_id;
    model.primary_outputs = primary_outputs;
    model.test_id2_endpoint_id = test_id2_endpoint_id;
    model.primary_output2_endpoint_id = primary_output2_endpoint_id;
    model.endpoints0 = endpoints0.data_ptr<float>();
    model.endpoints1 = endpoints1.data_ptr<float>();
    model.endpoint_pin_slacks = endpoint_pin_slacks.data_ptr<float>();
    model.num_pins = num_pins;
    model.num_arcs = num_arcs;
    model.num_tests = num_tests;
    model.num_POs = num_POs;
    model.num_endpoint_pins = num_endpoint_pins;
    EndpointSlackModel* d_model = nullptr;
    endpoint_abort_on_cuda_error(cudaMalloc(&d_model, sizeof(EndpointSlackModel)),
                                 "update_endpoints cudaMalloc model");
    endpoint_abort_on_cuda_error(cudaMemcpy(d_model, &model, sizeof(EndpointSlackModel), cudaMemcpyHostToDevice),
                                 "update_endpoints cudaMemcpy model");

    if (gputimer_env_enabled("GPUTIMER_ENDPOINT_DEBUG")) {
        fprintf(stderr,
                "[GPUTIMER ENDPOINT DEVICE] target=%d pinAT=%d pinRAT=%d endpoints0=%d endpoints1=%d endpoint_pin_slacks=%d\n",
                target_device,
                timing_raw_db.pinAT.is_cuda() ? static_cast<int>(timing_raw_db.pinAT.get_device()) : -1,
                timing_raw_db.pinRAT.is_cuda() ? static_cast<int>(timing_raw_db.pinRAT.get_device()) : -1,
                endpoints0.is_cuda() ? static_cast<int>(endpoints0.get_device()) : -1,
                endpoints1.is_cuda() ? static_cast<int>(endpoints1.get_device()) : -1,
                endpoint_pin_slacks.is_cuda() ? static_cast<int>(endpoint_pin_slacks.get_device()) : -1);
    }

    update_endpoints_kernel0<<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(d_model);
    endpoint_check_cuda_and_clear("update_endpoints_kernel0");
    update_endpoints_kernel1<<<BLOCK_NUMBER(num_POs * NUM_ATTR), BLOCK_SIZE>>>(d_model);
    endpoint_check_cuda_and_clear("update_endpoints_kernel1");
    update_endpoint_pin_slacks_kernel0<<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(d_model);
    endpoint_check_cuda_and_clear("update_endpoint_pin_slacks_kernel0");
    update_endpoint_pin_slacks_kernel1<<<BLOCK_NUMBER(num_POs * NUM_ATTR), BLOCK_SIZE>>>(d_model);
    endpoint_check_cuda_and_clear("update_endpoint_pin_slacks_kernel1");
    endpoint_abort_on_cuda_error(cudaFree(d_model),
                                 "update_endpoints cudaFree model");
    if (gputimer_env_enabled("GPUTIMER_ENDPOINT_DEBUG")) {
        print_endpoint_debug_summary(model);
    }

    endpoint_slacks = torch::cat({endpoints0, endpoints1}, 0).contiguous();
}


}  // namespace gt
