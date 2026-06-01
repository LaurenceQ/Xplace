#include "PowerCudaActivityKernels.cuh"

namespace gt {

__global__ void power_snapshot_level_active_kernel(const PowerActivityCudaModel* model,
                                                   PowerActivityScratchView* scratch,
                                                   int level_start,
                                                   int num_level_pins) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = model->graph.level_list[level_start + pos];
    if (pin < 0) return;
    scratch->visit_active[pin] = static_cast<uint8_t>(atomicExch(&scratch->active[pin], 0) != 0);
}

__global__ void power_snapshot_level_active_list_kernel(const PowerActivityCudaModel* model,
                                                        PowerActivityScratchView* scratch,
                                                        int level_start,
                                                        int num_level_pins,
                                                        int* active_count,
                                                        int* active_pins) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = model->graph.level_list[level_start + pos];
    if (pin < 0) return;
    const bool is_active = atomicExch(&scratch->active[pin], 0) != 0;
    if (scratch->visit_active) scratch->visit_active[pin] = static_cast<uint8_t>(is_active);
    if (!is_active || !active_count || !active_pins) return;
    const int out_pos = atomicAdd(active_count, 1);
    active_pins[out_pos] = pin;
}



__global__ void power_seed_pi_kernel(PowerActivityCudaModel* model,
                                     PowerActivityScratchView* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->state.num_primary_inputs) return;
    const int pin = model->state.primary_inputs[idx];
    if (pin < 0) return;
    if (power_set_activity(pin, model->config.default_density, 0.5f, 1, false, model, scratch)) {
        power_enqueue_adjacent(pin, model, scratch);
    }
}

__global__ void power_seed_clock_active_kernel(PowerActivityCudaModel* model,
                                               PowerActivityScratchView* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->state.num_clock_pins) return;
    const int pin = model->state.clock_pins[idx];
    if (pin < 0) return;
    const float pin_density = model->state.clock_pin_densities
        ? model->state.clock_pin_densities[idx]
        : model->config.clock_density;
    const float pin_duty = model->state.clock_pin_duties ? model->state.clock_pin_duties[idx] : 0.5f;
    const bool enqueue = !model->state.clock_pin_enqueue || model->state.clock_pin_enqueue[idx] != 0;
    if (power_set_activity(pin, pin_density, pin_duty, 2, true, model, scratch) && enqueue) {
        power_enqueue_adjacent(pin, model, scratch);
    }
}

__global__ void power_seed_case_kernel(PowerActivityCudaModel* model,
                                       PowerActivityScratchView* scratch) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    const int* case_values = model->state.case_values;
    if (pin >= model->n || !case_values || case_values[pin] < 0) return;
    if (power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true, model, scratch)) {
        power_enqueue_adjacent(pin, model, scratch);
    }
}

__global__ void power_seed_seq_feedback_state_kernel(PowerActivityCudaModel* model,
                                                     PowerActivityScratchView* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < model->state.num_feedback_seed_pins) {
        const int pin = model->state.feedback_seed_pins[idx];
        if (pin >= 0) {
            power_set_activity(pin, model->config.default_density, 0.5f, 1, false, model, scratch);
        }
    }
    if (idx < model->state.num_feedback_seed_seqs) {
        const int seq_id = model->state.feedback_seed_seqs[idx];
        if (seq_id >= 0 && atomicExch(&scratch->pending_seq[seq_id], 1) == 0) {
            atomicAdd(scratch->pending_seq_count, 1);
        }
    }
}

__device__ bool power_process_level_pin(int pin,
                                        const PowerActivityCudaModel* model,
                                        PowerActivityScratchView* scratch,
                                        bool defer_pending_seq) {
    const auto& graph = model->graph;
    const auto& expr = model->expr;
    const auto& state = model->state;
    const int* case_values = state.case_values;
    float* density = scratch->density;
    float* duty = scratch->duty;
    int* origin = scratch->origin;
    bool changed = false;
    if (case_values && case_values[pin] >= 0) {
        changed = power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true,
                                     model, scratch);
    } else if (graph.is_load_pin[pin]) {
        const int net = graph.pin2net_map[pin];
        const int driver = (net >= 0 && graph.net_driver_pin) ? graph.net_driver_pin[net] : -1;
        if (driver >= 0 && driver != pin && (!origin || origin[driver] != 0)) {
            changed = power_set_activity(pin, density[driver], duty[driver], 3, false,
                                         model, scratch);
        }
    }
    if ((!case_values || case_values[pin] < 0) && graph.is_driver_pin[pin]) {
        if (scratch->seq_pin_valid && scratch->seq_pin_valid[pin]) {
            changed = power_set_activity(pin, scratch->seq_pin_density[pin],
                                         scratch->seq_pin_duty[pin], 3, false,
                                         model, scratch) || changed;
        } else {
             const int expr_id = expr.pin_func_expr_id[pin];
             if (expr_id >= 0) {
                 float out_density = 0.0f, out_duty = 0.0f;
                 const int node_id = graph.pin2node_map ? graph.pin2node_map[pin] : -1;
                 if (power_eval_expr_activity(expr_id, expr.expr_ops, expr.expr_start, expr.expr_count,
                                              density, duty, out_density, out_duty,
                                              expr.node_port_pin_start, expr.node_port_pin_list,
                                              node_id)) {
                     changed = power_set_activity(pin, out_density, out_duty, 3, false,
                                                   model, scratch) || changed;
                 }
             }
        }
        changed = power_set_clock_gate_output(pin, model, scratch) || changed;
    }
    if (!changed) return false;
    if (graph.is_load_pin[pin]) {
        if (!defer_pending_seq && power_should_mark_pending_seq(density[pin])) {
            for (int i = state.pin_seq_list_start[pin]; i < state.pin_seq_list_start[pin + 1]; i++) {
                const int seq_id = state.pin_seq_list[i];
                if (seq_id >= 0 && atomicExch(&scratch->pending_seq[seq_id], 1) == 0)
                    atomicAdd(scratch->pending_seq_count, 1);
            }
        }
        power_enqueue_clock_gate_output(pin, model, scratch);
        if (expr.missing_func_out_start && expr.missing_func_out_list) {
            for (int i = expr.missing_func_out_start[pin]; i < expr.missing_func_out_start[pin + 1]; ++i) {
                const int out_pin = expr.missing_func_out_list[i];
                if (out_pin < 0) continue;
                  const int expr_id = expr.pin_func_expr_id[out_pin];
                  if (expr_id < 0) continue;
                  float out_density = 0.0f, out_duty = 0.0f;
                  const int node_id = graph.pin2node_map ? graph.pin2node_map[out_pin] : -1;
                  if (!power_eval_expr_activity(expr_id, expr.expr_ops, expr.expr_start, expr.expr_count,
                                                density, duty, out_density, out_duty,
                                                expr.node_port_pin_start, expr.node_port_pin_list,
                                                node_id))
                      continue;
                  if (power_set_activity(out_pin, out_density, out_duty, 3, false, model, scratch)) {
                      power_enqueue_adjacent(out_pin, model, scratch);
                }
            }
        }
    }
    power_enqueue_adjacent(pin, model, scratch);
    return true;
}

__global__ void power_visit_level_kernel(PowerActivityCudaModel* model,
                                         PowerActivityScratchView* scratch,
                                         int level_start,
                                         int num_level_pins,
                                         bool defer_pending_seq) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = model->graph.level_list[level_start + pos];
    if (pin < 0 || !scratch->visit_active || scratch->visit_active[pin] == 0) return;
    scratch->visit_active[pin] = 0;
    power_process_level_pin(pin, model, scratch, defer_pending_seq);
}

__global__ void power_visit_level_serial_kernel(PowerActivityCudaModel* model,
                                                PowerActivityScratchView* scratch,
                                                int level_start,
                                                int num_level_pins,
                                                bool defer_pending_seq) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int pos = num_level_pins - 1; pos >= 0; --pos) {
        const int pin = model->graph.level_list[level_start + pos];
        if (pin < 0) continue;
        if (atomicExch(&scratch->active[pin], 0) == 0) continue;
        power_process_level_pin(pin, model, scratch, defer_pending_seq);
    }
}

__global__ void power_visit_active_list_serial_kernel(PowerActivityCudaModel* model,
                                                      PowerActivityScratchView* scratch,
                                                      const int* active_pins,
                                                      const int* active_count,
                                                      bool defer_pending_seq) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !active_pins || !active_count) return;
    for (int idx = *active_count - 1; idx >= 0; --idx) {
        const int pin = active_pins[idx];
        if (pin < 0) continue;
        if (scratch->visit_active && scratch->visit_active[pin] == 0) continue;
        if (scratch->visit_active) scratch->visit_active[pin] = 0;
        power_process_level_pin(pin, model, scratch, defer_pending_seq);
    }
}

__global__ void power_mark_pending_seq_changes_kernel(PowerActivityCudaModel* model,
                                                      PowerActivityScratchView* scratch) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin >= model->n || !model->graph.is_load_pin || !model->graph.is_load_pin[pin]) return;
    if (!model->state.pin_seq_list_start || !model->state.pin_seq_list ||
        !scratch->pending_seq || !scratch->pending_seq_count)
        return;
    const bool changed = power_percent_change(scratch->density[pin], scratch->prev_density[pin]) > 0.01f
        || power_percent_change(scratch->duty[pin], scratch->prev_duty[pin]) > 0.01f
        || scratch->origin[pin] != scratch->prev_origin[pin];
    if (!changed || !power_should_mark_pending_seq(scratch->density[pin])) return;
    for (int i = model->state.pin_seq_list_start[pin]; i < model->state.pin_seq_list_start[pin + 1]; i++) {
        const int seq_id = model->state.pin_seq_list[i];
        if (seq_id >= 0 && atomicExch(&scratch->pending_seq[seq_id], 1) == 0)
            atomicAdd(scratch->pending_seq_count, 1);
    }
}

__global__ void power_seed_seq_kernel(PowerActivityCudaModel* model,
                                      PowerActivityScratchView* scratch) {
    const int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_id >= model->state.num_seqs) return;
    if (atomicExch(&scratch->pending_seq[seq_id], 0) == 0) return;
    atomicSub(scratch->pending_seq_count, 1);
    const auto seq = model->state.seqs[seq_id];
    float in_density = 0.0f, in_duty = 0.0f;
    float clk_density = model->config.clock_density, clk_duty = 0.5f;
      if ((g_power_require_known_seq_data
           && !power_expr_has_known_activity_input(seq.data_expr_id, model->expr.expr_ops,
                                                   model->expr.expr_start, model->expr.expr_count,
                                                   scratch->origin,
                                                   model->expr.node_port_pin_start,
                                                   model->expr.node_port_pin_list,
                                                   seq.node_id))
          || !power_eval_expr_activity(seq.data_expr_id, model->expr.expr_ops, model->expr.expr_start,
                                       model->expr.expr_count, scratch->density, scratch->duty,
                                       in_density, in_duty,
                                       model->expr.node_port_pin_start,
                                       model->expr.node_port_pin_list,
                                       seq.node_id)) return;
      power_eval_expr_activity(seq.clk_expr_id, model->expr.expr_ops, model->expr.expr_start,
                               model->expr.expr_count, scratch->density, scratch->duty,
                               clk_density, clk_duty,
                               model->expr.node_port_pin_start,
                               model->expr.node_port_pin_list,
                               seq.node_id);
    float out_density = in_density;
    float out_duty = in_duty;
    if (power_seq_density_exceeds_clock_limit(in_density, clk_density)) {
        out_density = seq.is_latch ? in_density * clk_duty
                                   : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
    }
    if (seq.q_pin >= 0) {
        scratch->seq_pin_density[seq.q_pin] = out_density;
        scratch->seq_pin_duty[seq.q_pin] = out_duty;
        scratch->seq_pin_valid[seq.q_pin] = 1;
        power_activate_pin(seq.q_pin, model, scratch);
    }
    if (seq.qn_pin >= 0) {
        const float qn_duty = 1.0f - out_duty;
        scratch->seq_pin_density[seq.qn_pin] = out_density;
        scratch->seq_pin_duty[seq.qn_pin] = qn_duty;
        scratch->seq_pin_valid[seq.qn_pin] = 1;
        power_activate_pin(seq.qn_pin, model, scratch);
    }
}

__global__ void power_seed_seq_ordered_kernel(PowerActivityCudaModel* model,
                                              PowerActivityScratchView* scratch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int seq_id = 0; seq_id < model->state.num_seqs; ++seq_id) {
        if (scratch->pending_seq[seq_id] == 0) continue;
        scratch->pending_seq[seq_id] = 0;
        const auto seq = model->state.seqs[seq_id];
        float in_density = 0.0f, in_duty = 0.0f;
        float clk_density = model->config.clock_density, clk_duty = 0.5f;
        if ((g_power_require_known_seq_data
               && !power_expr_has_known_activity_input(seq.data_expr_id, model->expr.expr_ops,
                                                       model->expr.expr_start, model->expr.expr_count,
                                                       scratch->origin,
                                                       model->expr.node_port_pin_start,
                                                       model->expr.node_port_pin_list,
                                                       seq.node_id))
              || !power_eval_expr_activity(seq.data_expr_id, model->expr.expr_ops,
                                           model->expr.expr_start, model->expr.expr_count,
                                           scratch->density, scratch->duty, in_density, in_duty,
                                           model->expr.node_port_pin_start,
                                           model->expr.node_port_pin_list,
                                           seq.node_id))
              continue;
          power_eval_expr_activity(seq.clk_expr_id, model->expr.expr_ops, model->expr.expr_start,
                                   model->expr.expr_count, scratch->density, scratch->duty,
                                   clk_density, clk_duty,
                                   model->expr.node_port_pin_start,
                                   model->expr.node_port_pin_list,
                                   seq.node_id);
        float out_density = in_density;
        float out_duty = in_duty;
        if (power_seq_density_exceeds_clock_limit(in_density, clk_density)) {
            out_density = seq.is_latch ? in_density * clk_duty
                                       : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
        }
        if (seq.q_pin >= 0) {
            if (g_power_direct_ordered_seq_seed)
                power_set_activity(seq.q_pin, out_density, out_duty, 3, false, model, scratch);
            scratch->seq_pin_density[seq.q_pin] = out_density;
            scratch->seq_pin_duty[seq.q_pin] = out_duty;
            scratch->seq_pin_valid[seq.q_pin] = 1;
            power_activate_pin(seq.q_pin, model, scratch);
        }
        if (seq.qn_pin >= 0) {
            const float qn_duty = 1.0f - out_duty;
            if (g_power_direct_ordered_seq_seed)
                power_set_activity(seq.qn_pin, out_density, qn_duty, 3, false, model, scratch);
            scratch->seq_pin_density[seq.qn_pin] = out_density;
            scratch->seq_pin_duty[seq.qn_pin] = qn_duty;
            scratch->seq_pin_valid[seq.qn_pin] = 1;
            power_activate_pin(seq.qn_pin, model, scratch);
        }
    }
    if (scratch->pending_seq_count) *scratch->pending_seq_count = 0;
}

__global__ void power_seed_seq_id_list_ordered_kernel(PowerActivityCudaModel* model,
                                                      PowerActivityScratchView* scratch,
                                                      const int* seq_ids,
                                                      int num_seq_ids) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !seq_ids) return;
    for (int idx = 0; idx < num_seq_ids; ++idx) {
        const int seq_id = seq_ids[idx];
        if (seq_id < 0) continue;
        if (scratch->pending_seq && scratch->pending_seq[seq_id] == 0) continue;
        if (scratch->pending_seq) scratch->pending_seq[seq_id] = 0;
        const auto seq = model->state.seqs[seq_id];
        float in_density = 0.0f, in_duty = 0.0f;
        float clk_density = model->config.clock_density, clk_duty = 0.5f;
        if ((g_power_require_known_seq_data
               && !power_expr_has_known_activity_input(seq.data_expr_id, model->expr.expr_ops,
                                                       model->expr.expr_start, model->expr.expr_count,
                                                       scratch->origin,
                                                       model->expr.node_port_pin_start,
                                                       model->expr.node_port_pin_list,
                                                       seq.node_id))
              || !power_eval_expr_activity(seq.data_expr_id, model->expr.expr_ops,
                                           model->expr.expr_start, model->expr.expr_count,
                                           scratch->density, scratch->duty, in_density, in_duty,
                                           model->expr.node_port_pin_start,
                                           model->expr.node_port_pin_list,
                                           seq.node_id))
              continue;
          power_eval_expr_activity(seq.clk_expr_id, model->expr.expr_ops, model->expr.expr_start,
                                   model->expr.expr_count, scratch->density, scratch->duty,
                                   clk_density, clk_duty,
                                   model->expr.node_port_pin_start,
                                   model->expr.node_port_pin_list,
                                   seq.node_id);
        float out_density = in_density;
        float out_duty = in_duty;
        if (power_seq_density_exceeds_clock_limit(in_density, clk_density)) {
            out_density = seq.is_latch ? in_density * clk_duty
                                       : 2.0f * in_duty * (1.0f - in_duty) * clk_density;
        }
        if (seq.q_pin >= 0) {
            if (g_power_direct_ordered_seq_seed)
                power_set_activity(seq.q_pin, out_density, out_duty, 3, false, model, scratch);
            scratch->seq_pin_density[seq.q_pin] = out_density;
            scratch->seq_pin_duty[seq.q_pin] = out_duty;
            scratch->seq_pin_valid[seq.q_pin] = 1;
            power_activate_pin(seq.q_pin, model, scratch);
        }
        if (seq.qn_pin >= 0) {
            const float qn_duty = 1.0f - out_duty;
            if (g_power_direct_ordered_seq_seed)
                power_set_activity(seq.qn_pin, out_density, qn_duty, 3, false, model, scratch);
            scratch->seq_pin_density[seq.qn_pin] = out_density;
            scratch->seq_pin_duty[seq.qn_pin] = qn_duty;
            scratch->seq_pin_valid[seq.qn_pin] = 1;
            power_activate_pin(seq.qn_pin, model, scratch);
        }
    }
    if (scratch->pending_seq_count) *scratch->pending_seq_count = 0;
}

}  // namespace gt
