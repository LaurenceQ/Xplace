#include "PowerCudaActivityKernels.cuh"

#include <cooperative_groups.h>

namespace gt {

__device__ bool power_process_pin_frontier(int pin,
                                           const PowerActivityCudaModel* model,
                                           PowerActivityScratchView* scratch,
                                           PowerActivityQueueView* queue) {
    const auto& graph = model->graph;
    const auto& expr = model->expr;
    const auto& state = model->state;
    float* density = scratch->density;
    float* duty = scratch->duty;
    int* origin = scratch->origin;
    bool changed = false;
    const bool has_case_value = state.case_values && state.case_values[pin] >= 0;
    if (has_case_value) {
        changed = power_set_activity(pin, 0.0f, state.case_values[pin] ? 1.0f : 0.0f, 4, true,
                                     model, scratch);
    }
    if (!has_case_value && graph.is_load_pin[pin]) {
        const int net = graph.pin2net_map[pin];
        const int driver = (net >= 0 && graph.net_driver_pin) ? graph.net_driver_pin[net] : -1;
        if (driver >= 0 && driver != pin && (!origin || origin[driver] != 0)) {
            changed = power_set_activity(pin, density[driver], duty[driver], 3, false,
                                         model, scratch);
        }
    }
    if (!has_case_value && graph.is_driver_pin[pin]) {
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
    if (changed && graph.is_load_pin[pin] && power_should_mark_pending_seq(density[pin])) {
        for (int i = state.pin_seq_list_start[pin]; i < state.pin_seq_list_start[pin + 1]; i++) {
            const int seq_id = state.pin_seq_list[i];
            if (seq_id >= 0 && atomicExch(&scratch->pending_seq[seq_id], 1) == 0) {
                atomicAdd(scratch->pending_seq_count, 1);
                if (queue && queue->pending_seq_list && queue->pending_seq_list_count &&
                    model->state.num_seqs > 0) {
                    const int pos = atomicAdd(queue->pending_seq_list_count, 1);
                    if (pos < model->state.num_seqs) queue->pending_seq_list[pos] = seq_id;
                }
            }
        }
    }
    return changed;
}

__device__ void power_enqueue_pin_level_queue(int pin,
                                              const PowerActivityCudaModel* model,
                                              const PowerActivityScratchView* scratch,
                                              PowerActivityQueueView* queue) {
    if (pin < 0 || !model->graph.pin_power_level || !queue || !queue->level_offsets) return;
    const int level = model->graph.pin_power_level[pin];
    if (level < 0 || level >= scratch->num_power_levels) return;
    if (atomicExch(&queue->queued[pin], 1) == 0) {
        const int pos = atomicAdd(&queue->level_counts[level], 1);
        const int cap = queue->level_offsets[level + 1] - queue->level_offsets[level];
        if (pos < cap) queue->level_queue[queue->level_offsets[level] + pos] = pin;
        else if (queue->overflow) atomicExch(queue->overflow, 1);
    }
}

__device__ void power_enqueue_adjacent_level_queue(int pin,
                                                   const PowerActivityCudaModel* model,
                                                   const PowerActivityScratchView* scratch,
                                                   PowerActivityQueueView* queue) {
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
                power_enqueue_pin_level_queue(sink, model, scratch, queue);
            }
        }
    }
    for (index_type i = graph.pin_forward_arc_list_end[pin];
         i < graph.pin_forward_arc_list_end[pin + 1]; i++) {
        const int arc = graph.pin_forward_arc_list[i];
        if (graph.arc_id2test_id && graph.arc_id2test_id[arc] != -1) continue;
        const int to_pin = graph.timing_arc_to_pin_id[arc];
        if (to_pin < 0) continue;
        if (graph.arc_types && graph.arc_types[arc] == 1 &&
            graph.is_seq_output_pin && graph.is_seq_output_pin[to_pin])
            continue;
        power_enqueue_pin_level_queue(to_pin, model, scratch, queue);
    }
}

__device__ void power_enqueue_clock_gate_output_level_queue(int pin,
                                                            const PowerActivityCudaModel* model,
                                                            const PowerActivityScratchView* scratch,
                                                            PowerActivityQueueView* queue) {
    if (!model->graph.clock_gate_out_for_input) return;
    const int out_pin = model->graph.clock_gate_out_for_input[pin];
    if (out_pin < 0) return;
    power_enqueue_pin_level_queue(out_pin, model, scratch, queue);
}

__device__ void power_enqueue_missing_func_outputs_level_queue(int pin,
                                                               const PowerActivityCudaModel* model,
                                                               PowerActivityScratchView* scratch,
                                                               PowerActivityQueueView* queue) {
    const auto& expr = model->expr;
    if (!expr.missing_func_out_start || !expr.missing_func_out_list) return;
    for (int i = expr.missing_func_out_start[pin]; i < expr.missing_func_out_start[pin + 1]; ++i) {
        const int out_pin = expr.missing_func_out_list[i];
        if (out_pin < 0) continue;
         const int expr_id = expr.pin_func_expr_id[out_pin];
         if (expr_id < 0) continue;
         float out_density = 0.0f;
         float out_duty = 0.0f;
         const int node_id = model->graph.pin2node_map ? model->graph.pin2node_map[out_pin] : -1;
         if (!power_eval_expr_activity(expr_id, expr.expr_ops, expr.expr_start, expr.expr_count,
                                       scratch->density, scratch->duty, out_density, out_duty,
                                       expr.node_port_pin_start, expr.node_port_pin_list,
                                       node_id))
             continue;
         if (power_set_activity(out_pin, out_density, out_duty, 3, false, model, scratch)) {
             power_enqueue_adjacent_level_queue(out_pin, model, scratch, queue);
        }
    }
}

__global__ void power_seed_case_level_queue_kernel(PowerActivityCudaModel* model,
                                                   PowerActivityScratchView* scratch,
                                                   PowerActivityQueueView* queue) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    const int* case_values = model->state.case_values;
    if (pin >= model->n || !case_values || case_values[pin] < 0) return;
    if (power_set_activity(pin, 0.0f, case_values[pin] ? 1.0f : 0.0f, 4, true, model, scratch)) {
        power_enqueue_adjacent_level_queue(pin, model, scratch, queue);
    }
}

__global__ void power_seed_pi_level_queue_kernel(PowerActivityCudaModel* model,
                                                 PowerActivityScratchView* scratch,
                                                 PowerActivityQueueView* queue) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->state.num_primary_inputs) return;
    const int pin = model->state.primary_inputs[idx];
    if (pin < 0) return;
    if (power_set_activity(pin, model->config.default_density, 0.5f, 1, false, model, scratch)) {
        power_enqueue_adjacent_level_queue(pin, model, scratch, queue);
    }
}

__global__ void power_seed_clock_level_queue_kernel(PowerActivityCudaModel* model,
                                                    PowerActivityScratchView* scratch,
                                                    PowerActivityQueueView* queue) {
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
        power_enqueue_adjacent_level_queue(pin, model, scratch, queue);
    }
}

__global__ void power_seed_roots_level_queue_ordered_kernel(PowerActivityCudaModel* model,
                                                            PowerActivityScratchView* scratch,
                                                            PowerActivityQueueView* queue) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (model->state.case_values) {
        for (int pin = 0; pin < model->n; ++pin) {
            if (model->state.case_values[pin] < 0) continue;
            if (power_set_activity(pin, 0.0f, model->state.case_values[pin] ? 1.0f : 0.0f,
                                   4, true, model, scratch)) {
                power_enqueue_adjacent_level_queue(pin, model, scratch, queue);
            }
        }
    }
    for (int idx = 0; idx < model->state.num_primary_inputs; ++idx) {
        const int pin = model->state.primary_inputs ? model->state.primary_inputs[idx] : -1;
        if (pin < 0 || pin >= model->n) continue;
        if (power_set_activity(pin, model->config.default_density, 0.5f, 1, false, model, scratch)) {
            power_enqueue_adjacent_level_queue(pin, model, scratch, queue);
        }
    }
    for (int idx = 0; idx < model->state.num_clock_pins; ++idx) {
        const int pin = model->state.clock_pins ? model->state.clock_pins[idx] : -1;
        if (pin < 0 || pin >= model->n) continue;
        const float pin_density = model->state.clock_pin_densities
            ? model->state.clock_pin_densities[idx]
            : model->config.clock_density;
        const float pin_duty = model->state.clock_pin_duties ? model->state.clock_pin_duties[idx] : 0.5f;
        const bool enqueue = !model->state.clock_pin_enqueue || model->state.clock_pin_enqueue[idx] != 0;
        if (power_set_activity(pin, pin_density, pin_duty, 2, true, model, scratch) && enqueue) {
            power_enqueue_adjacent_level_queue(pin, model, scratch, queue);
        }
    }
}

__device__ void power_seed_frontier_seq(int seq_id,
                                        const PowerActivityCudaModel* model,
                                        PowerActivityScratchView* scratch,
                                        PowerActivityQueueView* queue) {
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
          return;
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
        power_enqueue_pin_level_queue(seq.q_pin, model, scratch, queue);
    }
    if (seq.qn_pin >= 0) {
        const float qn_duty = 1.0f - out_duty;
        scratch->seq_pin_density[seq.qn_pin] = out_density;
        scratch->seq_pin_duty[seq.qn_pin] = qn_duty;
        scratch->seq_pin_valid[seq.qn_pin] = 1;
        power_enqueue_pin_level_queue(seq.qn_pin, model, scratch, queue);
    }
}

__global__ void power_activity_level_queue_persistent_kernel(PowerActivityCudaModel* model,
                                                             PowerActivityScratchView* scratch,
                                                             PowerActivityQueueView* queue,
                                                             int max_seq_passes) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int num_power_levels = scratch->num_power_levels;

    for (int pass = 0; pass < max_seq_passes; ++pass) {
        for (int level = 0; level < num_power_levels; ++level) {
            grid.sync();
            const int count = queue->level_counts[level];
            const int offset = queue->level_offsets[level];
            for (int idx = tid; idx < count; idx += stride) {
                const int pin = queue->level_queue[offset + idx];
                if (pin < 0 || pin >= model->n) continue;
                atomicExch(&queue->queued[pin], 0);
                const bool changed = power_process_pin_frontier(pin, model, scratch, nullptr);
                if (changed) {
                    if (model->graph.is_load_pin[pin]) {
                        power_enqueue_clock_gate_output_level_queue(pin, model, scratch, queue);
                        power_enqueue_missing_func_outputs_level_queue(pin, model, scratch, queue);
                    }
                    power_enqueue_adjacent_level_queue(pin, model, scratch, queue);
                }
            }
            grid.sync();
            if (tid == 0) queue->level_counts[level] = 0;
        }
        grid.sync();
        const int pending = *scratch->pending_seq_count;
        if (pending <= 0) break;
        for (int seq_id = tid; seq_id < model->state.num_seqs; seq_id += stride) {
            if (atomicExch(&scratch->pending_seq[seq_id], 0) == 0) continue;
            atomicSub(scratch->pending_seq_count, 1);
            power_seed_frontier_seq(seq_id, model, scratch, queue);
        }
    }
}

__global__ void power_activity_level_queue_ordered_kernel(PowerActivityCudaModel* model,
                                                          PowerActivityScratchView* scratch,
                                                          PowerActivityQueueView* queue,
                                                          int max_seq_passes) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

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
                    queue->queued[pin] = 0;
                    const bool changed = power_process_pin_frontier(pin, model, scratch, queue);
                    if (!changed) continue;
                    if (model->graph.is_load_pin[pin]) {
                        power_enqueue_clock_gate_output_level_queue(pin, model, scratch, queue);
                        power_enqueue_missing_func_outputs_level_queue(pin, model, scratch, queue);
                    }
                    power_enqueue_adjacent_level_queue(pin, model, scratch, queue);
                }
            }
        }
    };

    drain();
    for (int pass = 1; pass < max_seq_passes; ++pass) {
        if (*scratch->pending_seq_count <= 0) break;
        const int pending_items = queue->pending_seq_list_count ? *queue->pending_seq_list_count : 0;
        const int seq_scan_count = pending_items > 0 ? pending_items : model->state.num_seqs;
        for (int idx = 0; idx < seq_scan_count; ++idx) {
            const int seq_id = pending_items > 0 ? queue->pending_seq_list[idx] : idx;
            if (seq_id < 0 || seq_id >= model->state.num_seqs) continue;
            if (scratch->pending_seq[seq_id] == 0) continue;
            scratch->pending_seq[seq_id] = 0;
            *scratch->pending_seq_count -= 1;
            power_seed_frontier_seq(seq_id, model, scratch, queue);
        }
        if (queue->pending_seq_list_count) *queue->pending_seq_list_count = 0;
        drain();
    }
}

}  // namespace gt
