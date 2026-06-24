#include "PowerCudaActivityKernels.cuh"

#include <cooperative_groups.h>

namespace gt {

__device__ bool PowerLevelQueueOps::processPinFrontier(int pin) const {
    bool changed = false;
    const bool has_case_value = model->seed.case_values && model->seed.case_values[pin] >= 0;
    if (has_case_value) {
        changed = setActivity(pin,
                              0.0f,
                              model->seed.case_values[pin] ? 1.0f : 0.0f,
                              4,
                              true);
    }
    if (!has_case_value && model->graph.is_load_pin[pin]) {
        const int net = model->graph.pin2net_map[pin];
        const int driver = (net >= 0 && model->graph.net_driver_pin)
                               ? model->graph.net_driver_pin[net]
                               : -1;
        if (driver >= 0 && driver != pin &&
            (!scratch->origin || scratch->origin[driver] != 0)) {
            changed = setActivity(pin, scratch->density[driver], scratch->duty[driver], 3, false);
        }
    }
    if (!has_case_value && model->graph.is_driver_pin[pin]) {
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
    if (changed && model->graph.is_load_pin[pin] &&
        PowerActivityOps::shouldMarkPendingSeq(scratch->density[pin]) &&
        scratch->pending_seq && scratch->pending_seq_count) {
        const auto& seed = model->seed;
        if (!seed.pin_seq_list_start || !seed.pin_seq_list) return changed;
        for (int i = seed.pin_seq_list_start[pin]; i < seed.pin_seq_list_start[pin + 1]; i++) {
            const int seq_id = seed.pin_seq_list[i];
            if (seq_id >= 0 && atomicExch(&scratch->pending_seq[seq_id], 1) == 0) {
                atomicAdd(scratch->pending_seq_count, 1);
                if (queue && queue->pending_seq_list && queue->pending_seq_list_count &&
                    model->seed.num_seqs > 0) {
                    const int pos = atomicAdd(queue->pending_seq_list_count, 1);
                    if (pos < model->seed.num_seqs) queue->pending_seq_list[pos] = seq_id;
                }
            }
        }
    }
    return changed;
}

__device__ void PowerLevelQueueOps::enqueuePin(int pin) const {
    if (pin < 0 || !model->graph.pin_power_level || !queue || !queue->level_offsets) return;
    const int level = model->graph.pin_power_level[pin];
    if (level < 0 || level >= scratch->num_power_levels) return;
    if (!power_activity_flag_atomic_test_and_set(queue->queued, pin)) {
        const int pos = atomicAdd(&queue->level_counts[level], 1);
        const int cap = queue->level_offsets[level + 1] - queue->level_offsets[level];
        if (pos < cap) queue->level_queue[queue->level_offsets[level] + pos] = pin;
        else if (queue->overflow) atomicExch(queue->overflow, 1);
    }
}

__device__ void PowerLevelQueueOps::enqueueAdjacent(int pin) const {
    const auto& graph = model->graph;
    if (graph.is_load_pin && graph.pin2net_map && graph.net_driver_pin &&
        graph.flat_net2pin_start_map && graph.flat_net2pin_map) {
        const int net = graph.pin2net_map[pin];
        if (net >= 0 && graph.net_driver_pin[net] == pin) {
            const int start = graph.flat_net2pin_start_map[net];
            const int end = graph.flat_net2pin_start_map[net + 1];
            for (int pos = start; pos < end; ++pos) {
                const int sink = graph.flat_net2pin_map[pos];
                if (sink < 0 || sink == pin || !graph.is_load_pin[sink]) continue;
                enqueuePin(sink);
            }
        }
    }
    for (index_type i = graph.pin_forward_arc_list_end[pin];
         i < graph.pin_forward_arc_list_end[pin + 1]; i++) {
        const int arc = graph.pin_forward_arc_list[i];
        if (graph.arc_skip && graph.arc_skip[arc]) continue;
        const int to_pin = graph.timing_arc_to_pin_id[arc];
        if (to_pin < 0) continue;
        if (graph.arc_types && graph.arc_types[arc] == 1 &&
            graph.is_seq_output_pin && graph.is_seq_output_pin[to_pin] &&
            !power_activity_flag_test(graph.seq_output_arc_keep, arc))
            continue;
        enqueuePin(to_pin);
    }
}

__device__ void PowerLevelQueueOps::enqueueClockGateOutput(int pin) const {
    if (!model->graph.clock_gate_out_for_input) return;
    const int out_pin = model->graph.clock_gate_out_for_input[pin];
    if (out_pin < 0) return;
    enqueuePin(out_pin);
}

__device__ void PowerLevelQueueOps::enqueueMissingFuncOutputs(int pin) const {
    const auto& expr = model->expr;
    if (!expr.missing_func_out_start || !expr.missing_func_out_list) return;
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

__global__ void power_seed_case_level_queue_kernel(PowerActivityDevice* model,
                                                   PowerActivityPropDevice* scratch,
                                                   PowerActivityLevelQueueDevice* queue) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    const int* case_values = model->seed.case_values;
    if (pin >= model->n || !case_values || case_values[pin] < 0) return;
    PowerLevelQueueOps queue_ops(model, scratch, queue);
    if (queue_ops.setActivity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true)) {
        queue_ops.enqueueAdjacent(pin);
    }
}

__global__ void power_seed_pi_level_queue_kernel(PowerActivityDevice* model,
                                                 PowerActivityPropDevice* scratch,
                                                 PowerActivityLevelQueueDevice* queue) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->seed.num_primary_inputs) return;
    const int pin = model->seed.primary_inputs[idx];
    if (pin < 0) return;
    PowerLevelQueueOps queue_ops(model, scratch, queue);
    if (queue_ops.setActivity(pin, model->config.default_density, 0.5f, 1, false)) {
        queue_ops.enqueueAdjacent(pin);
    }
}

__global__ void power_seed_clock_level_queue_kernel(PowerActivityDevice* model,
                                                    PowerActivityPropDevice* scratch,
                                                    PowerActivityLevelQueueDevice* queue) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->seed.num_clock_pins) return;
    const int pin = model->seed.clock_pins[idx];
    if (pin < 0) return;
    const float pin_density = model->seed.clock_pin_densities
        ? model->seed.clock_pin_densities[idx]
        : model->config.clock_density;
    const float pin_duty = model->seed.clock_pin_duties ? model->seed.clock_pin_duties[idx] : 0.5f;
    const bool enqueue = !model->seed.clock_pin_enqueue || model->seed.clock_pin_enqueue[idx] != 0;
    PowerLevelQueueOps queue_ops(model, scratch, queue);
    if (queue_ops.setActivity(pin, pin_density, pin_duty, 2, true) && enqueue) {
        queue_ops.enqueueAdjacent(pin);
    }
}

__global__ void power_seed_roots_level_queue_ordered_kernel(PowerActivityDevice* model,
                                                            PowerActivityPropDevice* scratch,
                                                            PowerActivityLevelQueueDevice* queue) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (model->seed.case_values) {
        PowerLevelQueueOps queue_ops(model, scratch, queue);
        for (int pin = 0; pin < model->n; ++pin) {
            if (model->seed.case_values[pin] < 0) continue;
            if (queue_ops.setActivity(pin, 0.0f, model->seed.case_values[pin] ? 1.0f : 0.0f,
                                      4, true)) {
                queue_ops.enqueueAdjacent(pin);
            }
        }
    }
    PowerLevelQueueOps queue_ops(model, scratch, queue);
    for (int idx = 0; idx < model->seed.num_primary_inputs; ++idx) {
        const int pin = model->seed.primary_inputs ? model->seed.primary_inputs[idx] : -1;
        if (pin < 0 || pin >= model->n) continue;
        if (queue_ops.setActivity(pin, model->config.default_density, 0.5f, 1, false)) {
            queue_ops.enqueueAdjacent(pin);
        }
    }
    for (int idx = 0; idx < model->seed.num_clock_pins; ++idx) {
        const int pin = model->seed.clock_pins ? model->seed.clock_pins[idx] : -1;
        if (pin < 0 || pin >= model->n) continue;
        const float pin_density = model->seed.clock_pin_densities
            ? model->seed.clock_pin_densities[idx]
            : model->config.clock_density;
        const float pin_duty = model->seed.clock_pin_duties ? model->seed.clock_pin_duties[idx] : 0.5f;
        const bool enqueue = !model->seed.clock_pin_enqueue || model->seed.clock_pin_enqueue[idx] != 0;
        if (queue_ops.setActivity(pin, pin_density, pin_duty, 2, true) && enqueue) {
            queue_ops.enqueueAdjacent(pin);
        }
    }
}

__device__ void PowerLevelQueueOps::seedFrontierSeq(int seq_id) const {
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
        !expr_device.activity(seq.data_expr_id, in_value.density, in_value.duty))
        return;
    expr_device.activity(seq.clk_expr_id, clk_value.density, clk_value.duty);
    const float out_density = PowerActivityOps::seqDensityExceedsClockLimit(in_value.density, clk_value.density)
        ? (seq.is_latch
               ? in_value.density * clk_value.duty
               : 2.0f * in_value.duty * (1.0f - in_value.duty) * clk_value.density)
        : in_value.density;
    const PowerActivityValue out_value(out_density, in_value.duty);
    if (seq.q_pin >= 0) {
        scratch->seq_pin_density[seq.q_pin] = out_value.density;
        scratch->seq_pin_duty[seq.q_pin] = out_value.duty;
        scratch->seq_pin_valid[seq.q_pin] = 1;
        enqueuePin(seq.q_pin);
    }
    if (seq.qn_pin >= 0) {
        const float qn_duty = 1.0f - out_value.duty;
        scratch->seq_pin_density[seq.qn_pin] = out_value.density;
        scratch->seq_pin_duty[seq.qn_pin] = qn_duty;
        scratch->seq_pin_valid[seq.qn_pin] = 1;
        enqueuePin(seq.qn_pin);
    }
}

__global__ void power_activity_level_queue_persistent_kernel(PowerActivityDevice* model,
                                                             PowerActivityPropDevice* scratch,
                                                             PowerActivityLevelQueueDevice* queue,
                                                             int max_seq_passes) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int num_power_levels = scratch->num_power_levels;
    PowerLevelQueueOps queue_ops(model, scratch, queue);

    for (int pass = 0; pass < max_seq_passes; ++pass) {
        for (int level = 0; level < num_power_levels; ++level) {
            grid.sync();
            const int count = queue->level_counts[level];
            const int offset = queue->level_offsets[level];
            for (int idx = tid; idx < count; idx += stride) {
                const int pin = queue->level_queue[offset + idx];
                if (pin < 0 || pin >= model->n) continue;
                power_activity_flag_atomic_test_and_clear(queue->queued, pin);
                const bool changed = PowerLevelQueueOps(model, scratch, nullptr).processPinFrontier(pin);
                if (changed) {
                    if (model->graph.is_load_pin[pin]) {
                        queue_ops.enqueueClockGateOutput(pin);
                        queue_ops.enqueueMissingFuncOutputs(pin);
                    }
                    queue_ops.enqueueAdjacent(pin);
                }
            }
            grid.sync();
            if (tid == 0) queue->level_counts[level] = 0;
        }
        grid.sync();
        const int pending = *scratch->pending_seq_count;
        if (pending <= 0) break;
        for (int seq_id = tid; seq_id < model->seed.num_seqs; seq_id += stride) {
            if (atomicExch(&scratch->pending_seq[seq_id], 0) == 0) continue;
            atomicSub(scratch->pending_seq_count, 1);
            queue_ops.seedFrontierSeq(seq_id);
        }
    }
}

__global__ void power_activity_level_queue_ordered_kernel(PowerActivityDevice* model,
                                                          PowerActivityPropDevice* scratch,
                                                          PowerActivityLevelQueueDevice* queue,
                                                          int max_seq_passes) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    PowerLevelQueueOps queue_ops(model, scratch, queue);

    auto drain = [&]() {
        bool any = true;
        while (any) {
            any = false;
            for (int level = 0; level < scratch->num_power_levels; ++level) {
                const int offset = queue->level_offsets[level];
                while (queue->level_counts[level] > 0) {
                    any = true;
                    const int idx = --queue->level_counts[level];
                    const int pin = queue->level_queue[offset + idx];
                    if (pin < 0 || pin >= model->n) continue;
                    power_activity_flag_clear(queue->queued, pin);
                    const bool changed = queue_ops.processPinFrontier(pin);
                    if (!changed) continue;
                    if (model->graph.is_load_pin[pin]) {
                        queue_ops.enqueueClockGateOutput(pin);
                        queue_ops.enqueueMissingFuncOutputs(pin);
                    }
                    queue_ops.enqueueAdjacent(pin);
                }
            }
        }
    };

    drain();
    for (int pass = 1; pass < max_seq_passes; ++pass) {
        if (*scratch->pending_seq_count <= 0) break;
        const int pending_items = queue->pending_seq_list_count ? *queue->pending_seq_list_count : 0;
        const int seq_scan_count = pending_items > 0 ? pending_items : model->seed.num_seqs;
        for (int idx = 0; idx < seq_scan_count; ++idx) {
            const int seq_id = pending_items > 0 ? queue->pending_seq_list[idx] : idx;
            if (seq_id < 0 || seq_id >= model->seed.num_seqs) continue;
            if (scratch->pending_seq[seq_id] == 0) continue;
            scratch->pending_seq[seq_id] = 0;
            *scratch->pending_seq_count -= 1;
            queue_ops.seedFrontierSeq(seq_id);
        }
        if (queue->pending_seq_list_count) *queue->pending_seq_list_count = 0;
        drain();
    }
}

}  // namespace gt
