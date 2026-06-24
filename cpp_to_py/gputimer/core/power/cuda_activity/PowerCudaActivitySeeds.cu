#include "PowerCudaActivityKernels.cuh"

namespace gt {

__global__ void power_snapshot_level_active_kernel(const PowerActivityDevice* model,
                                                   PowerActivityPropDevice* scratch,
                                                   int level_start,
                                                   int num_level_pins) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = model->graph.level_list[level_start + pos];
    if (pin < 0) return;
    scratch->visit_active[pin] =
        static_cast<uint8_t>(power_activity_flag_atomic_test_and_clear(scratch->active, pin));
}

__global__ void power_snapshot_level_active_list_kernel(const PowerActivityDevice* model,
                                                        PowerActivityPropDevice* scratch,
                                                        int level_start,
                                                        int num_level_pins,
                                                        int* active_count,
                                                        int* active_pins) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = model->graph.level_list[level_start + pos];
    if (pin < 0) return;
    const bool is_active = power_activity_flag_atomic_test_and_clear(scratch->active, pin);
    if (scratch->visit_active) scratch->visit_active[pin] = static_cast<uint8_t>(is_active);
    if (!is_active || !active_count || !active_pins) return;
    const int out_pos = atomicAdd(active_count, 1);
    active_pins[out_pos] = pin;
}



__global__ void power_seed_pi_kernel(PowerActivityDevice* model,
                                     PowerActivityPropDevice* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->seed.num_primary_inputs) return;
    const int pin = model->seed.primary_inputs[idx];
    if (pin < 0) return;
    PowerActivityOps activity(model, scratch);
    if (activity.setActivity(pin, model->config.default_density, 0.5f, 1, false)) {
        activity.enqueueAdjacent(pin);
    }
}

__global__ void power_seed_clock_active_kernel(PowerActivityDevice* model,
                                               PowerActivityPropDevice* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->seed.num_clock_pins) return;
    const int pin = model->seed.clock_pins[idx];
    if (pin < 0) return;
    const float pin_density = model->seed.clock_pin_densities
        ? model->seed.clock_pin_densities[idx]
        : model->config.clock_density;
    const float pin_duty = model->seed.clock_pin_duties ? model->seed.clock_pin_duties[idx] : 0.5f;
    const bool enqueue = !model->seed.clock_pin_enqueue || model->seed.clock_pin_enqueue[idx] != 0;
    PowerActivityOps activity(model, scratch);
    if (activity.setActivity(pin, pin_density, pin_duty, 2, true) && enqueue) {
        activity.enqueueAdjacent(pin);
    }
}

__global__ void power_seed_case_kernel(PowerActivityDevice* model,
                                       PowerActivityPropDevice* scratch) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    const int* case_values = model->seed.case_values;
    if (pin >= model->n || !case_values || case_values[pin] < 0) return;
    PowerActivityOps activity(model, scratch);
    if (activity.setActivity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true)) {
        activity.enqueueAdjacent(pin);
    }
}

__global__ void power_seed_seq_feedback_state_kernel(PowerActivityDevice* model,
                                                     PowerActivityPropDevice* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < model->seed.num_feedback_seed_pins) {
        const int pin = model->seed.feedback_seed_pins[idx];
        if (pin >= 0) {
            PowerActivityOps(model, scratch).setActivity(pin, model->config.default_density, 0.5f, 1, false);
        }
    }
    if (idx < model->seed.num_feedback_seed_seqs &&
        scratch->pending_seq && scratch->pending_seq_count) {
        const int seq_id = model->seed.feedback_seed_seqs[idx];
        if (seq_id >= 0 && atomicExch(&scratch->pending_seq[seq_id], 1) == 0) {
            atomicAdd(scratch->pending_seq_count, 1);
        }
    }
}

__device__ bool PowerActivityOps::processLevelPin(int pin, bool defer_pending_seq) const {
    bool changed = false;
    const int* case_values = model->seed.case_values;
    if (case_values && case_values[pin] >= 0) {
        changed = setActivity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true);
    } else if (model->graph.is_load_pin[pin]) {
        const int net = model->graph.pin2net_map[pin];
        const int driver = (net >= 0 && model->graph.net_driver_pin)
                               ? model->graph.net_driver_pin[net]
                               : -1;
        if (driver >= 0 && driver != pin &&
            (!scratch->origin || scratch->origin[driver] != 0)) {
            changed = setActivity(pin, scratch->density[driver], scratch->duty[driver], 3, false);
        }
    }
    if ((!case_values || case_values[pin] < 0) && model->graph.is_driver_pin[pin]) {
        if (scratch->seq_pin_valid && scratch->seq_pin_valid[pin]) {
            changed = setActivity(pin,
                                  scratch->seq_pin_density[pin],
                                  scratch->seq_pin_duty[pin],
                                  3,
                                  false) || changed;
        } else {
            const auto& expr = model->expr;
            const int expr_id = expr.pin_expr_id[pin];
            if (expr_id >= 0) {
                float value_density = 0.0f;
                float value_duty = 0.0f;
                const int node_id = model->graph.pin2node_map ? model->graph.pin2node_map[pin] : -1;
                PowerExprEval expr_device(expr.expr_ops,
                                        expr.expr_start,
                                        expr.expr_count,
                                        scratch->density,
                                        scratch->duty,
                                        expr.node_port_pin_start,
                                        expr.node_port_pin_list,
                                        node_id);
                if (expr_device.activity(expr_id, value_density, value_duty)) {
                    changed = setActivity(pin, value_density, value_duty, 3, false) || changed;
                }
            }
        }
        changed = setClockGateOutput(pin) || changed;
    }
    if (!changed) return false;
    if (model->graph.is_load_pin[pin]) {
        const auto& seed = model->seed;
        if (!defer_pending_seq && shouldMarkPendingSeq(scratch->density[pin]) &&
            seed.pin_seq_list_start && seed.pin_seq_list &&
            scratch->pending_seq && scratch->pending_seq_count) {
            for (int i = seed.pin_seq_list_start[pin]; i < seed.pin_seq_list_start[pin + 1]; i++) {
                const int seq_id = seed.pin_seq_list[i];
                if (seq_id >= 0 && atomicExch(&scratch->pending_seq[seq_id], 1) == 0)
                    atomicAdd(scratch->pending_seq_count, 1);
            }
        }
        enqueueClockGateOutput(pin);
        const auto& expr = model->expr;
        if (expr.missing_func_out_start && expr.missing_func_out_list) {
            for (int i = expr.missing_func_out_start[pin]; i < expr.missing_func_out_start[pin + 1]; ++i) {
                const int out_pin = expr.missing_func_out_list[i];
                if (out_pin < 0) continue;
                const int expr_id = expr.pin_expr_id[out_pin];
                if (expr_id < 0) continue;
                float value_density = 0.0f;
                float value_duty = 0.0f;
                const int node_id = model->graph.pin2node_map ? model->graph.pin2node_map[out_pin] : -1;
                PowerExprEval expr_device(expr.expr_ops,
                                        expr.expr_start,
                                        expr.expr_count,
                                        scratch->density,
                                        scratch->duty,
                                        expr.node_port_pin_start,
                                        expr.node_port_pin_list,
                                        node_id);
                if (!expr_device.activity(expr_id, value_density, value_duty))
                    continue;
                if (setActivity(out_pin, value_density, value_duty, 3, false)) {
                    enqueueAdjacent(out_pin);
                }
            }
        }
    }
    enqueueAdjacent(pin);
    return true;
}

__device__ void PowerActivityOps::seedSeqActivity(int seq_id, bool direct_ordered) const {
    const auto seq = model->seed.seqs[seq_id];
    PowerActivityValue in_value;
    PowerActivityValue clk_value{model->config.clock_density, 0.5f};
    const auto& expr = model->expr;
    PowerExprEval expr_device(expr.expr_ops,
                            expr.expr_start,
                            expr.expr_count,
                            scratch->density,
                            scratch->duty,
                            expr.node_port_pin_start,
                            expr.node_port_pin_list,
                            seq.node_id);
    if ((g_power_require_known_seq_data &&
         !expr_device.hasKnownActivityInput(seq.data_expr_id, scratch->origin)) ||
        !expr_device.activity(seq.data_expr_id, in_value.density, in_value.duty)) {
        return;
    }
    expr_device.activity(seq.clk_expr_id, clk_value.density, clk_value.duty);
    const float out_density = seqDensityExceedsClockLimit(in_value.density, clk_value.density)
        ? (seq.is_latch
               ? in_value.density * clk_value.duty
               : 2.0f * in_value.duty * (1.0f - in_value.duty) * clk_value.density)
        : in_value.density;
    const PowerActivityValue out_value(out_density, in_value.duty);
    if (seq.q_pin >= 0) {
        if (direct_ordered) setActivity(seq.q_pin, out_value.density, out_value.duty, 3, false);
        scratch->seq_pin_density[seq.q_pin] = out_value.density;
        scratch->seq_pin_duty[seq.q_pin] = out_value.duty;
        scratch->seq_pin_valid[seq.q_pin] = 1;
        activatePin(seq.q_pin);
    }
    if (seq.qn_pin >= 0) {
        const float qn_duty = 1.0f - out_value.duty;
        if (direct_ordered) setActivity(seq.qn_pin, out_value.density, qn_duty, 3, false);
        scratch->seq_pin_density[seq.qn_pin] = out_value.density;
        scratch->seq_pin_duty[seq.qn_pin] = qn_duty;
        scratch->seq_pin_valid[seq.qn_pin] = 1;
        activatePin(seq.qn_pin);
    }
}

__global__ void power_visit_level_kernel(PowerActivityDevice* model,
                                         PowerActivityPropDevice* scratch,
                                         int level_start,
                                         int num_level_pins,
                                         bool defer_pending_seq) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = model->graph.level_list[level_start + pos];
    if (pin < 0 || !scratch->visit_active || scratch->visit_active[pin] == 0) return;
    scratch->visit_active[pin] = 0;
    PowerActivityOps(model, scratch).processLevelPin(pin, defer_pending_seq);
}

__global__ void power_visit_level_serial_kernel(PowerActivityDevice* model,
                                                PowerActivityPropDevice* scratch,
                                                int level_start,
                                                int num_level_pins,
                                                bool defer_pending_seq) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int pos = num_level_pins - 1; pos >= 0; --pos) {
        const int pin = model->graph.level_list[level_start + pos];
        if (pin < 0) continue;
        if (!power_activity_flag_atomic_test_and_clear(scratch->active, pin)) continue;
        PowerActivityOps(model, scratch).processLevelPin(pin, defer_pending_seq);
    }
}

__global__ void power_visit_active_list_serial_kernel(PowerActivityDevice* model,
                                                      PowerActivityPropDevice* scratch,
                                                      const int* active_pins,
                                                      const int* active_count,
                                                      bool defer_pending_seq) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !active_pins || !active_count) return;
    for (int idx = *active_count - 1; idx >= 0; --idx) {
        const int pin = active_pins[idx];
        if (pin < 0) continue;
        if (scratch->visit_active && scratch->visit_active[pin] == 0) continue;
        if (scratch->visit_active) scratch->visit_active[pin] = 0;
        PowerActivityOps(model, scratch).processLevelPin(pin, defer_pending_seq);
    }
}

__global__ void power_mark_pending_seq_changes_kernel(PowerActivityDevice* model,
                                                      PowerActivityPropDevice* scratch) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin >= model->n || !model->graph.is_load_pin || !model->graph.is_load_pin[pin]) return;
    if (!model->seed.pin_seq_list_start || !model->seed.pin_seq_list ||
        !scratch->pending_seq || !scratch->pending_seq_count)
        return;
    const bool changed = PowerActivityOps::percentChange(scratch->density[pin], scratch->prev_density[pin]) > 0.01f
        || PowerActivityOps::percentChange(scratch->duty[pin], scratch->prev_duty[pin]) > 0.01f
        || scratch->origin[pin] != scratch->prev_origin[pin];
    if (!changed || !PowerActivityOps::shouldMarkPendingSeq(scratch->density[pin])) return;
    for (int i = model->seed.pin_seq_list_start[pin]; i < model->seed.pin_seq_list_start[pin + 1]; i++) {
        const int seq_id = model->seed.pin_seq_list[i];
        if (seq_id >= 0 && atomicExch(&scratch->pending_seq[seq_id], 1) == 0)
            atomicAdd(scratch->pending_seq_count, 1);
    }
}

__global__ void power_seed_seq_kernel(PowerActivityDevice* model,
                                      PowerActivityPropDevice* scratch) {
    const int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_id >= model->seed.num_seqs) return;
    if (atomicExch(&scratch->pending_seq[seq_id], 0) == 0) return;
    atomicSub(scratch->pending_seq_count, 1);
    PowerActivityOps(model, scratch).seedSeqActivity(seq_id, false);
}

__global__ void power_seed_seq_ordered_kernel(PowerActivityDevice* model,
                                              PowerActivityPropDevice* scratch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int seq_id = 0; seq_id < model->seed.num_seqs; ++seq_id) {
        if (scratch->pending_seq[seq_id] == 0) continue;
        scratch->pending_seq[seq_id] = 0;
        PowerActivityOps(model, scratch).seedSeqActivity(seq_id, g_power_direct_ordered_seq_seed != 0);
    }
    if (scratch->pending_seq_count) *scratch->pending_seq_count = 0;
}

__global__ void power_seed_seq_id_list_ordered_kernel(PowerActivityDevice* model,
                                                      PowerActivityPropDevice* scratch,
                                                      const int* seq_ids,
                                                      int num_seq_ids) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !seq_ids) return;
    for (int idx = 0; idx < num_seq_ids; ++idx) {
        const int seq_id = seq_ids[idx];
        if (seq_id < 0) continue;
        if (scratch->pending_seq && scratch->pending_seq[seq_id] == 0) continue;
        if (scratch->pending_seq) scratch->pending_seq[seq_id] = 0;
        PowerActivityOps(model, scratch).seedSeqActivity(seq_id, g_power_direct_ordered_seq_seed != 0);
    }
    if (scratch->pending_seq_count) *scratch->pending_seq_count = 0;
}

}  // namespace gt
