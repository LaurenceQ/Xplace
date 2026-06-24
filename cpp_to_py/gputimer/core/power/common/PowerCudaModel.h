#pragma once

#include "gputimer/core/GPUTimer.h"

#include <cstdint>
#include <vector>

namespace gt {

struct PowerGraphDevice {
    index_type* level_list = nullptr;
    const int* pin_power_level = nullptr;
    index_type* pin_forward_arc_list_end = nullptr;
    index_type* pin_forward_arc_list = nullptr;
    index_type* timing_arc_to_pin_id = nullptr;
    const uint8_t* arc_types = nullptr;
    const uint32_t* seq_output_arc_keep = nullptr;
    const uint8_t* arc_skip = nullptr;
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
    const uint16_t* pin_clock_ids = nullptr;
    const float* clock_slews = nullptr;
    int clock_count = 0;
    const int* power_clock_slew_pins = nullptr;
    int num_power_clock_slew_pins = 0;
    float power_clock_slew_fallback[NUM_ATTR] = {0.0f, 0.0f, 0.0f, 0.0f};
    int num_nodes = 0;

    PowerGraphDevice() = default;
    PowerGraphDevice(index_type* level_list_,
                         const int* pin_power_level_,
                         index_type* pin_forward_arc_list_end_,
                         index_type* pin_forward_arc_list_,
                         index_type* timing_arc_to_pin_id_,
                         const uint8_t* arc_types_,
                         const uint32_t* seq_output_arc_keep_,
                         const uint8_t* arc_skip_,
                         const int* pin2net_map_,
                         const int* net_driver_pin_,
                         const int* flat_net2pin_start_map_,
                         const int* flat_net2pin_map_,
                         uint8_t* is_load_pin_,
                         uint8_t* is_driver_pin_,
                         uint8_t* is_cell_pin_,
                         uint8_t* is_seq_output_pin_,
                         uint8_t* is_seq_clock_input_pin_,
                         int* clock_gate_out_for_input_,
                         int* clock_gate_clock_for_out_,
                         int* clock_gate_enable_for_out_,
                         const int* pin2node_map_,
                         const float* pinLoad_,
                         const float* dmp_C1_,
                         const float* dmp_C2_,
                         const float* pinSlew_,
                         const uint16_t* pin_clock_ids_,
                         const float* clock_slews_,
                         int clock_count_,
                         const int* power_clock_slew_pins_,
                         int num_power_clock_slew_pins_,
                         const float* power_clock_slew_fallback_,
                         int num_nodes_)
        : level_list(level_list_),
          pin_power_level(pin_power_level_),
          pin_forward_arc_list_end(pin_forward_arc_list_end_),
          pin_forward_arc_list(pin_forward_arc_list_),
          timing_arc_to_pin_id(timing_arc_to_pin_id_),
          arc_types(arc_types_),
          seq_output_arc_keep(seq_output_arc_keep_),
          arc_skip(arc_skip_),
          pin2net_map(pin2net_map_),
          net_driver_pin(net_driver_pin_),
          flat_net2pin_start_map(flat_net2pin_start_map_),
          flat_net2pin_map(flat_net2pin_map_),
          is_load_pin(is_load_pin_),
          is_driver_pin(is_driver_pin_),
          is_cell_pin(is_cell_pin_),
          is_seq_output_pin(is_seq_output_pin_),
          is_seq_clock_input_pin(is_seq_clock_input_pin_),
          clock_gate_out_for_input(clock_gate_out_for_input_),
          clock_gate_clock_for_out(clock_gate_clock_for_out_),
          clock_gate_enable_for_out(clock_gate_enable_for_out_),
          pin2node_map(pin2node_map_),
          pinLoad(pinLoad_),
          dmp_C1(dmp_C1_),
          dmp_C2(dmp_C2_),
          pinSlew(pinSlew_),
          pin_clock_ids(pin_clock_ids_),
          clock_slews(clock_slews_),
          clock_count(clock_count_),
          power_clock_slew_pins(power_clock_slew_pins_),
          num_power_clock_slew_pins(num_power_clock_slew_pins_),
          num_nodes(num_nodes_) {
        for (int attr = 0; attr < NUM_ATTR; ++attr)
            power_clock_slew_fallback[attr] =
                power_clock_slew_fallback_ ? power_clock_slew_fallback_[attr] : 0.0f;
    }
};

struct PowerExprDevice {
    GpuPowerExprOpHost* expr_ops = nullptr;
    int* expr_start = nullptr;
    int* expr_count = nullptr;
    int* node_port_pin_start = nullptr;
    int* node_port_pin_list = nullptr;
    int* pin_expr_id = nullptr;
    int* missing_func_out_start = nullptr;
    int* missing_func_out_list = nullptr;

    PowerExprDevice() = default;
    PowerExprDevice(GpuPowerExprOpHost* expr_ops_,
                        int* expr_start_,
                        int* expr_count_,
                        int* node_port_pin_start_,
                        int* node_port_pin_list_,
                        int* pin_expr_id_,
                        int* missing_func_out_start_,
                        int* missing_func_out_list_)
        : expr_ops(expr_ops_),
          expr_start(expr_start_),
          expr_count(expr_count_),
          node_port_pin_start(node_port_pin_start_),
          node_port_pin_list(node_port_pin_list_),
          pin_expr_id(pin_expr_id_),
          missing_func_out_start(missing_func_out_start_),
          missing_func_out_list(missing_func_out_list_) {}
};

struct PowerActivitySeedDevice {
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

    PowerActivitySeedDevice() = default;
    PowerActivitySeedDevice(int* primary_inputs_,
                       int num_primary_inputs_,
                       int* case_values_,
                       int* clock_pins_,
                       int num_clock_pins_,
                       const float* clock_pin_densities_,
                       const float* clock_pin_duties_,
                       const uint8_t* clock_pin_enqueue_,
                       GpuPowerSeqHost* seqs_,
                       int num_seqs_,
                       int* pin_seq_list_start_,
                       int* pin_seq_list_,
                       int* feedback_seed_pins_,
                       int num_feedback_seed_pins_,
                       int* feedback_seed_seqs_,
                       int num_feedback_seed_seqs_)
        : primary_inputs(primary_inputs_),
          num_primary_inputs(num_primary_inputs_),
          case_values(case_values_),
          clock_pins(clock_pins_),
          num_clock_pins(num_clock_pins_),
          clock_pin_densities(clock_pin_densities_),
          clock_pin_duties(clock_pin_duties_),
          clock_pin_enqueue(clock_pin_enqueue_),
          seqs(seqs_),
          num_seqs(num_seqs_),
          pin_seq_list_start(pin_seq_list_start_),
          pin_seq_list(pin_seq_list_),
          feedback_seed_pins(feedback_seed_pins_),
          num_feedback_seed_pins(num_feedback_seed_pins_),
          feedback_seed_seqs(feedback_seed_seqs_),
          num_feedback_seed_seqs(num_feedback_seed_seqs_) {}
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

    PowerActivityConfig() = default;
    PowerActivityConfig(float default_density_,
                        float clock_density_,
                        float time_unit_,
                        int max_activity_passes_,
                        int* trace_pins_,
                        int num_trace_pins_,
                        const float* precomputed_activity_,
                        bool allow_clock_activity_override_,
                        float min_activity_density_)
        : default_density(default_density_),
          clock_density(clock_density_),
          time_unit(time_unit_),
          max_activity_passes(max_activity_passes_),
          trace_pins(trace_pins_),
          num_trace_pins(num_trace_pins_),
          precomputed_activity(precomputed_activity_),
          allow_clock_activity_override(allow_clock_activity_override_),
          min_activity_density(min_activity_density_) {}
};

struct PowerComponentDevice {
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

    PowerComponentDevice() = default;
    PowerComponentDevice(GpuPowerInternalHost* internal_rows_,
                             int num_internal_rows_,
                             int num_internal_denom_groups_,
                             GPUPowerLutAllocator* power_allocator_,
                             float cap_unit_,
                             float voltage_,
                             float* inst_switching_,
                             float* pin_switching_,
                             float* inst_internal_,
                             float* internal_row_power_,
                             GpuPowerLeakageRowHost* leakage_rows_,
                             int num_leakage_rows_,
                             GpuPowerLeakageGroupHost* leakage_groups_,
                             int num_leakage_groups_,
                             float* inst_leakage_,
                             float* leakage_row_power_)
        : internal_rows(internal_rows_),
          num_internal_rows(num_internal_rows_),
          num_internal_denom_groups(num_internal_denom_groups_),
          power_allocator(power_allocator_),
          cap_unit(cap_unit_),
          voltage(voltage_),
          inst_switching(inst_switching_),
          pin_switching(pin_switching_),
          inst_internal(inst_internal_),
          internal_row_power(internal_row_power_),
          leakage_rows(leakage_rows_),
          num_leakage_rows(num_leakage_rows_),
          leakage_groups(leakage_groups_),
          num_leakage_groups(num_leakage_groups_),
          inst_leakage(inst_leakage_),
          leakage_row_power(leakage_row_power_) {}
};

struct PowerActivityDevice {
    int n = 0;
    const std::vector<int>* level_list_end_cpu = nullptr;
    PowerGraphDevice graph;
    PowerExprDevice expr;
    PowerActivitySeedDevice seed;
    PowerActivityConfig config;
    PowerComponentDevice components;
    float* out = nullptr;
    int out_activity_fields = 0;

    PowerActivityDevice() = default;
    PowerActivityDevice(int n_,
                           const std::vector<int>* level_list_end_cpu_,
                           const PowerGraphDevice& graph_,
                           const PowerExprDevice& expr_,
                           const PowerActivitySeedDevice& seed_,
                           const PowerActivityConfig& config_,
                           const PowerComponentDevice& components_,
                           float* out_,
                           int out_activity_fields_)
        : n(n_),
          level_list_end_cpu(level_list_end_cpu_),
          graph(graph_),
          expr(expr_),
          seed(seed_),
          config(config_),
          components(components_),
          out(out_),
          out_activity_fields(out_activity_fields_) {}
};

struct PowerActivityPropDevice {
    float* density = nullptr;
    float* duty = nullptr;
    float* prev_density = nullptr;
    float* prev_duty = nullptr;
    float* seq_pin_density = nullptr;
    float* seq_pin_duty = nullptr;
    uint8_t* origin = nullptr;
    uint8_t* prev_origin = nullptr;
    uint32_t* active = nullptr;
    uint8_t* active_level = nullptr;
    uint8_t* visit_active = nullptr;
    uint8_t* seq_pin_valid = nullptr;
    int* pending_seq = nullptr;
    int* pending_seq_count = nullptr;
    int num_power_levels = 0;

    PowerActivityPropDevice() = default;
    PowerActivityPropDevice(float* density_, float* duty_)
        : density(density_), duty(duty_) {}
    PowerActivityPropDevice(float* density_,
                             float* duty_,
                             float* prev_density_,
                             float* prev_duty_,
                             float* seq_pin_density_,
                             float* seq_pin_duty_,
                             uint8_t* origin_,
                             uint8_t* prev_origin_,
                             uint32_t* active_,
                             uint8_t* active_level_,
                             uint8_t* visit_active_,
                             uint8_t* seq_pin_valid_,
                             int* pending_seq_,
                             int* pending_seq_count_,
                             int num_power_levels_)
        : density(density_),
          duty(duty_),
          prev_density(prev_density_),
          prev_duty(prev_duty_),
          seq_pin_density(seq_pin_density_),
          seq_pin_duty(seq_pin_duty_),
          origin(origin_),
          prev_origin(prev_origin_),
          active(active_),
          active_level(active_level_),
          visit_active(visit_active_),
          seq_pin_valid(seq_pin_valid_),
          pending_seq(pending_seq_),
          pending_seq_count(pending_seq_count_),
          num_power_levels(num_power_levels_) {}
};

struct PowerActivityLevelQueueDevice {
    const int* level_offsets = nullptr;
    int* level_queue = nullptr;
    int* level_counts = nullptr;
    uint32_t* queued = nullptr;
    int* overflow = nullptr;
    int* pending_seq_list = nullptr;
    int* pending_seq_list_count = nullptr;

    PowerActivityLevelQueueDevice() = default;
    PowerActivityLevelQueueDevice(const int* level_offsets_,
                           int* level_queue_,
                           int* level_counts_,
                           uint32_t* queued_,
                           int* overflow_,
                           int* pending_seq_list_,
                           int* pending_seq_list_count_)
        : level_offsets(level_offsets_),
          level_queue(level_queue_),
          level_counts(level_counts_),
          queued(queued_),
          overflow(overflow_),
          pending_seq_list(pending_seq_list_),
          pending_seq_list_count(pending_seq_list_count_) {}
};

struct PowerInternalDenomDevice {
    int n = 0;
    const float* precomputed_activity = nullptr;
    const float* activity_density = nullptr;
    const float* activity_duty = nullptr;
    GpuPowerInternalHost* internal_rows = nullptr;
    int num_internal_rows = 0;
    GpuPowerExprOpHost* expr_ops = nullptr;
    int* expr_start = nullptr;
    int* expr_count = nullptr;
    int* node_port_pin_start = nullptr;
    int* node_port_pin_list = nullptr;
    float* denom = nullptr;

    PowerInternalDenomDevice() = default;
    PowerInternalDenomDevice(int n_,
                            const float* precomputed_activity_,
                            const float* activity_density_,
                            const float* activity_duty_,
                            GpuPowerInternalHost* internal_rows_,
                            int num_internal_rows_,
                            GpuPowerExprOpHost* expr_ops_,
                            int* expr_start_,
                            int* expr_count_,
                            int* node_port_pin_start_,
                            int* node_port_pin_list_,
                            float* denom_)
        : n(n_),
          precomputed_activity(precomputed_activity_),
          activity_density(activity_density_),
          activity_duty(activity_duty_),
          internal_rows(internal_rows_),
          num_internal_rows(num_internal_rows_),
          expr_ops(expr_ops_),
          expr_start(expr_start_),
          expr_count(expr_count_),
          node_port_pin_start(node_port_pin_start_),
          node_port_pin_list(node_port_pin_list_),
          denom(denom_) {}
};

struct PowerInternalInstDevice {
    int n = 0;
    int num_nodes = 0;
    const float* precomputed_activity = nullptr;
    const float* activity_density = nullptr;
    const float* activity_duty = nullptr;
    GpuPowerInternalHost* internal_rows = nullptr;
    int num_internal_rows = 0;
    GpuPowerExprOpHost* expr_ops = nullptr;
    int* expr_start = nullptr;
    int* expr_count = nullptr;
    int* node_port_pin_start = nullptr;
    int* node_port_pin_list = nullptr;
    const float* pinSlew = nullptr;
    const uint16_t* pin_clock_ids = nullptr;
    const float* clock_slews = nullptr;
    int clock_count = 0;
    const int* power_clock_slew_pins = nullptr;
    int num_power_clock_slew_pins = 0;
    float power_clock_slew_fallback[NUM_ATTR] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float* dmp_C1 = nullptr;
    const float* dmp_C2 = nullptr;
    const float* denom = nullptr;
    GPUPowerLutAllocator* power_allocator = nullptr;
    float cap_unit = 1.0f;
    float* inst_internal = nullptr;
    float* internal_row_power = nullptr;

    PowerInternalInstDevice() = default;
    PowerInternalInstDevice(int n_,
                              int num_nodes_,
                              const float* precomputed_activity_,
                              const float* activity_density_,
                              const float* activity_duty_,
                              GpuPowerInternalHost* internal_rows_,
                              int num_internal_rows_,
                              GpuPowerExprOpHost* expr_ops_,
                              int* expr_start_,
                              int* expr_count_,
                              int* node_port_pin_start_,
                              int* node_port_pin_list_,
                              const float* pinSlew_,
                              const uint16_t* pin_clock_ids_,
                              const float* clock_slews_,
                              int clock_count_,
                              const int* power_clock_slew_pins_,
                              int num_power_clock_slew_pins_,
                              const float* power_clock_slew_fallback_,
                              const float* dmp_C1_,
                              const float* dmp_C2_,
                              const float* denom_,
                              GPUPowerLutAllocator* power_allocator_,
                              float cap_unit_,
                              float* inst_internal_,
                              float* internal_row_power_)
        : n(n_),
          num_nodes(num_nodes_),
          precomputed_activity(precomputed_activity_),
          activity_density(activity_density_),
          activity_duty(activity_duty_),
          internal_rows(internal_rows_),
          num_internal_rows(num_internal_rows_),
          expr_ops(expr_ops_),
          expr_start(expr_start_),
          expr_count(expr_count_),
          node_port_pin_start(node_port_pin_start_),
          node_port_pin_list(node_port_pin_list_),
          pinSlew(pinSlew_),
          pin_clock_ids(pin_clock_ids_),
          clock_slews(clock_slews_),
          clock_count(clock_count_),
          power_clock_slew_pins(power_clock_slew_pins_),
          num_power_clock_slew_pins(num_power_clock_slew_pins_),
          dmp_C1(dmp_C1_),
          dmp_C2(dmp_C2_),
          denom(denom_),
          power_allocator(power_allocator_),
          cap_unit(cap_unit_),
          inst_internal(inst_internal_),
          internal_row_power(internal_row_power_) {
        for (int attr = 0; attr < NUM_ATTR; ++attr)
            power_clock_slew_fallback[attr] =
                power_clock_slew_fallback_ ? power_clock_slew_fallback_[attr] : 0.0f;
    }
};

struct PowerLeakageCondDevice {
    int n = 0;
    const float* precomputed_activity = nullptr;
    const float* activity_density = nullptr;
    const float* activity_duty = nullptr;
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

    PowerLeakageCondDevice() = default;
    PowerLeakageCondDevice(int n_,
                          const float* precomputed_activity_,
                          const float* activity_density_,
                          const float* activity_duty_,
                          GpuPowerLeakageRowHost* leakage_rows_,
                          int num_leakage_rows_,
                          GpuPowerExprOpHost* expr_ops_,
                          int* expr_start_,
                          int* expr_count_,
                          int* node_port_pin_start_,
                          int* node_port_pin_list_,
                          float* group_cond_leakage_,
                          float* group_cond_duty_sum_,
                          int* group_cond_count_,
                          float* leakage_row_power_)
        : n(n_),
          precomputed_activity(precomputed_activity_),
          activity_density(activity_density_),
          activity_duty(activity_duty_),
          leakage_rows(leakage_rows_),
          num_leakage_rows(num_leakage_rows_),
          expr_ops(expr_ops_),
          expr_start(expr_start_),
          expr_count(expr_count_),
          node_port_pin_start(node_port_pin_start_),
          node_port_pin_list(node_port_pin_list_),
          group_cond_leakage(group_cond_leakage_),
          group_cond_duty_sum(group_cond_duty_sum_),
          group_cond_count(group_cond_count_),
          leakage_row_power(leakage_row_power_) {}
};

struct PowerLeakageInstDevice {
    GpuPowerLeakageGroupHost* leakage_groups = nullptr;
    int num_leakage_groups = 0;
    float* group_cond_leakage = nullptr;
    float* group_cond_duty_sum = nullptr;
    int* group_cond_count = nullptr;
    int num_nodes = 0;
    float* inst_leakage = nullptr;

    PowerLeakageInstDevice() = default;
    PowerLeakageInstDevice(GpuPowerLeakageGroupHost* leakage_groups_,
                             int num_leakage_groups_,
                             float* group_cond_leakage_,
                             float* group_cond_duty_sum_,
                             int* group_cond_count_,
                             int num_nodes_,
                             float* inst_leakage_)
        : leakage_groups(leakage_groups_),
          num_leakage_groups(num_leakage_groups_),
          group_cond_leakage(group_cond_leakage_),
          group_cond_duty_sum(group_cond_duty_sum_),
          group_cond_count(group_cond_count_),
          num_nodes(num_nodes_),
          inst_leakage(inst_leakage_) {}
};

struct PowerChunkActivityStorage {
    float* density = nullptr;
    float* duty = nullptr;

    PowerChunkActivityStorage() = default;
    PowerChunkActivityStorage(float* density_, float* duty_)
        : density(density_), duty(duty_) {}
};

void clear_power_cuda_error();
void check_power_cuda_error(const char* label);
void run_power_activity_cuda_launcher(const PowerActivityDevice& model);
void init_power_chunk_activity_storage(int n,
                                       const float* packed_activity,
                                       PowerChunkActivityStorage* storage);
void free_power_chunk_activity_storage(PowerChunkActivityStorage* storage);
void run_power_internal_denom_chunk_cuda_launcher(const PowerInternalDenomDevice& model);
void run_power_internal_contrib_chunk_cuda_launcher(const PowerInternalInstDevice& model);
void run_power_leakage_rows_chunk_cuda_launcher(const PowerLeakageCondDevice& model);
void run_power_leakage_summary_chunk_cuda_launcher(const PowerLeakageInstDevice& model);

}  // namespace gt
