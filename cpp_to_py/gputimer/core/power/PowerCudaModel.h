#pragma once

#include "gputimer/core/GPUTimer.h"

#include <vector>

namespace gt {

struct PowerGraphDeviceView {
    index_type* level_list = nullptr;
    const int* pin_power_level = nullptr;
    index_type* pin_forward_arc_list_end = nullptr;
    index_type* pin_forward_arc_list = nullptr;
    index_type* timing_arc_to_pin_id = nullptr;
    int* arc_types = nullptr;
    int* arc_id2test_id = nullptr;
    const int* pin2net_map = nullptr;
    const int* net_driver_pin = nullptr;
    const int* flat_net2pin_start_map = nullptr;
    const int* flat_net2pin_map = nullptr;
    uint8_t* is_load_pin = nullptr;
    uint8_t* is_driver_pin = nullptr;
    uint8_t* is_cell_pin = nullptr;
    uint8_t* is_seq_output_pin = nullptr;
    uint8_t* is_seq_clock_input_pin = nullptr;
    int* clock_gate_out_for_input = nullptr;
    int* clock_gate_clock_for_out = nullptr;
    int* clock_gate_enable_for_out = nullptr;
    const int* pin2node_map = nullptr;
    const float* pinLoad = nullptr;
    const float* dmp_C1 = nullptr;
    const float* dmp_C2 = nullptr;
    const float* pinSlew = nullptr;
    const float* power_clock_slews = nullptr;
    int num_nodes = 0;
};

struct PowerExprDeviceView {
    GpuPowerExprOpHost* expr_ops = nullptr;
    int* expr_start = nullptr;
    int* expr_count = nullptr;
    int* node_port_pin_start = nullptr;
    int* node_port_pin_list = nullptr;
    int* pin_func_expr_id = nullptr;
    int* missing_func_out_start = nullptr;
    int* missing_func_out_list = nullptr;
};

struct PowerActivityState {
    int* primary_inputs = nullptr;
    int num_primary_inputs = 0;
    int* case_values = nullptr;
    int* clock_pins = nullptr;
    int num_clock_pins = 0;
    const float* clock_pin_densities = nullptr;
    const float* clock_pin_duties = nullptr;
    const uint8_t* clock_pin_enqueue = nullptr;
    GpuPowerSeqHost* seqs = nullptr;
    int num_seqs = 0;
    int* pin_seq_list_start = nullptr;
    int* pin_seq_list = nullptr;
    int* feedback_seed_pins = nullptr;
    int num_feedback_seed_pins = 0;
    int* feedback_seed_seqs = nullptr;
    int num_feedback_seed_seqs = 0;
};

struct PowerActivityConfig {
    float default_density = 0.0f;
    float clock_density = 0.0f;
    float time_unit = 0.0f;
    int max_activity_passes = 0;
    int* trace_pins = nullptr;
    int num_trace_pins = 0;
    const float* precomputed_activity = nullptr;
    bool allow_clock_activity_override = false;
    float min_activity_density = 0.0f;
};

struct PowerComponentDeviceView {
    GpuPowerInternalHost* internal_rows = nullptr;
    int num_internal_rows = 0;
    int num_internal_denom_groups = 0;
    GPUPowerLutAllocator* power_allocator = nullptr;
    float cap_unit = 1.0f;
    float voltage = 0.0f;
    float* inst_switching = nullptr;
    float* pin_switching = nullptr;
    float* inst_internal = nullptr;
    float* internal_row_power = nullptr;
    GpuPowerLeakageRowHost* leakage_rows = nullptr;
    int num_leakage_rows = 0;
    GpuPowerLeakageGroupHost* leakage_groups = nullptr;
    int num_leakage_groups = 0;
    float* inst_leakage = nullptr;
    float* leakage_row_power = nullptr;
};

struct PowerActivityCudaModel {
    int n = 0;
    const std::vector<int>* level_list_end_cpu = nullptr;
    PowerGraphDeviceView graph;
    PowerExprDeviceView expr;
    PowerActivityState state;
    PowerActivityConfig config;
    PowerComponentDeviceView components;
    float* out = nullptr;
};

struct PowerActivityScratchView {
    float* density = nullptr;
    float* duty = nullptr;
    float* prev_density = nullptr;
    float* prev_duty = nullptr;
    float* seq_pin_density = nullptr;
    float* seq_pin_duty = nullptr;
    int* origin = nullptr;
    int* prev_origin = nullptr;
    int* active = nullptr;
    uint8_t* active_level = nullptr;
    uint8_t* visit_active = nullptr;
    uint8_t* seq_pin_valid = nullptr;
    int* pending_seq = nullptr;
    int* pending_seq_count = nullptr;
    int num_power_levels = 0;
};

struct PowerActivityQueueView {
    const int* level_offsets = nullptr;
    int* level_queue = nullptr;
    int* level_counts = nullptr;
    int* queued = nullptr;
    int* overflow = nullptr;
    int* pending_seq_list = nullptr;
    int* pending_seq_list_count = nullptr;
};

struct PowerInternalDenomModel {
    int n = 0;
    const float* precomputed_activity = nullptr;
    GpuPowerInternalHost* internal_rows = nullptr;
    int num_internal_rows = 0;
    GpuPowerExprOpHost* expr_ops = nullptr;
    int* expr_start = nullptr;
    int* expr_count = nullptr;
    int* node_port_pin_start = nullptr;
    int* node_port_pin_list = nullptr;
    float* denom = nullptr;
};

struct PowerInternalContribModel {
    int n = 0;
    int num_nodes = 0;
    const float* precomputed_activity = nullptr;
    GpuPowerInternalHost* internal_rows = nullptr;
    int num_internal_rows = 0;
    GpuPowerExprOpHost* expr_ops = nullptr;
    int* expr_start = nullptr;
    int* expr_count = nullptr;
    int* node_port_pin_start = nullptr;
    int* node_port_pin_list = nullptr;
    const float* pinSlew = nullptr;
    const float* power_clock_slews = nullptr;
    const float* dmp_C1 = nullptr;
    const float* dmp_C2 = nullptr;
    const float* denom = nullptr;
    GPUPowerLutAllocator* power_allocator = nullptr;
    float cap_unit = 1.0f;
    float* inst_internal = nullptr;
    float* internal_row_power = nullptr;
};

struct PowerLeakageRowsModel {
    int n = 0;
    const float* precomputed_activity = nullptr;
    GpuPowerLeakageRowHost* leakage_rows = nullptr;
    int num_leakage_rows = 0;
    GpuPowerExprOpHost* expr_ops = nullptr;
    int* expr_start = nullptr;
    int* expr_count = nullptr;
    int* node_port_pin_start = nullptr;
    int* node_port_pin_list = nullptr;
    float* group_cond_leakage = nullptr;
    float* group_cond_duty_sum = nullptr;
    int* group_cond_count = nullptr;
    float* leakage_row_power = nullptr;
};

struct PowerLeakageSummaryModel {
    GpuPowerLeakageGroupHost* leakage_groups = nullptr;
    int num_leakage_groups = 0;
    float* group_cond_leakage = nullptr;
    float* group_cond_duty_sum = nullptr;
    int* group_cond_count = nullptr;
    int num_nodes = 0;
    float* inst_leakage = nullptr;
};

void clear_power_cuda_error();
void check_power_cuda_error(const char* label);
void run_power_activity_cuda_launcher(const PowerActivityCudaModel& model);
void run_power_internal_denom_chunk_cuda_launcher(const PowerInternalDenomModel& model);
void run_power_internal_contrib_chunk_cuda_launcher(const PowerInternalContribModel& model);
void run_power_leakage_rows_chunk_cuda_launcher(const PowerLeakageRowsModel& model);
void run_power_leakage_summary_chunk_cuda_launcher(const PowerLeakageSummaryModel& model);

}  // namespace gt
