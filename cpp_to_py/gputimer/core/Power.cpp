#include "GPUTimer.h"

#include "DmpModel.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
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
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace gt {

void clear_power_cuda_error();

void run_power_activity_cuda_launcher(int n,
                                      const std::vector<int>& level_list_end_cpu,
                                      index_type* d_level_list,
                                      const int* d_pin_power_level,
                                      index_type* d_pin_forward_arc_list_end,
                                      index_type* d_pin_forward_arc_list,
                                      index_type* d_timing_arc_to_pin_id,
                                      int* d_arc_types,
                                      int* d_arc_id2test_id,
                                      const int* d_pin2net_map,
                                      const int* d_net_driver_pin,
                                      const int* d_flat_net2pin_start_map,
                                      const int* d_flat_net2pin_map,
                                      uint8_t* d_is_load_pin,
                                      uint8_t* d_is_driver_pin,
                                      uint8_t* d_is_cell_pin,
                                      uint8_t* d_is_seq_output_pin,
                                      int* d_clock_gate_out_for_input,
                                      int* d_clock_gate_clock_for_out,
                                      int* d_clock_gate_enable_for_out,
                                      int* d_primary_inputs,
                                      int num_primary_inputs,
                                      int* d_case_values,
                                      int* d_clock_pins,
                                      int num_clock_pins,
                                      GpuPowerExprOpHost* d_expr_ops,
                                      int* d_expr_start,
                                      int* d_expr_count,
                                      int* d_pin_func_expr_id,
                                      GpuPowerSeqHost* d_seqs,
                                      int num_seqs,
                                      int* d_pin_seq_list_start,
                                      int* d_pin_seq_list,
                                      int* d_feedback_seed_pins,
                                      int num_feedback_seed_pins,
                                      int* d_feedback_seed_seqs,
                                      int num_feedback_seed_seqs,
                                      float default_density,
                                      float clock_density,
                                      float time_unit,
                                      int max_activity_passes,
                                      float* d_out,
                                      int num_nodes,
                                      const int* d_pin2node_map,
                                      const float* d_pinLoad,
                                      const double* d_dmp_C1,
                                      const double* d_dmp_C2,
                                      const float* d_pinSlew,
                                      const float* d_power_clock_slews,
                                      bool allow_clock_activity_override,
                                      GpuPowerInternalHost* d_internal_rows,
                                      int num_internal_rows,
                                      int num_internal_denom_groups,
                                      GPUPowerLutAllocator* d_power_allocator,
                                      float cap_unit,
                                      float voltage,
                                      float* d_inst_switching,
                                      float* d_pin_switching,
                                      float* d_inst_internal,
                                      float* d_internal_row_power,
                                      GpuPowerLeakageRowHost* d_leakage_rows,
                                      int num_leakage_rows,
                                      GpuPowerLeakageGroupHost* d_leakage_groups,
                                      int num_leakage_groups,
                                      float* d_inst_leakage,
                                      float* d_leakage_row_power);

namespace {

struct CpuActivity {
    float density = 0.0f;
    float duty = 0.0f;
    int origin = 0;  // 0 unknown, 1 input, 2 clock, 3 propagated, 4 constant.
};

static bool evalPowerExprWithPortValues(const PowerExpr& expr,
                                        const std::vector<int8_t>& port_values,
                                        int8_t& value) {
    std::vector<int8_t> stack;
    stack.reserve(expr.ops().size());
    auto pop = [&]() -> int8_t {
        if (stack.empty()) return -1;
        int8_t v = stack.back();
        stack.pop_back();
        return v;
    };

    for (const auto& op : expr.ops()) {
        switch (op.opcode) {
            case PowerExprOpcode::port:
                if (op.port_id >= 0 && op.port_id < static_cast<int>(port_values.size()))
                    stack.push_back(port_values[op.port_id]);
                else
                    stack.push_back(-1);
                break;
            case PowerExprOpcode::const_zero:
                stack.push_back(0);
                break;
            case PowerExprOpcode::const_one:
                stack.push_back(1);
                break;
            case PowerExprOpcode::logical_not: {
                const int8_t a = pop();
                stack.push_back(a < 0 ? -1 : static_cast<int8_t>(!a));
                break;
            }
            case PowerExprOpcode::logical_and: {
                const int8_t b = pop();
                const int8_t a = pop();
                if (a == 0 || b == 0) stack.push_back(0);
                else if (a == 1 && b == 1) stack.push_back(1);
                else stack.push_back(-1);
                break;
            }
            case PowerExprOpcode::logical_or: {
                const int8_t b = pop();
                const int8_t a = pop();
                if (a == 1 || b == 1) stack.push_back(1);
                else if (a == 0 && b == 0) stack.push_back(0);
                else stack.push_back(-1);
                break;
            }
            case PowerExprOpcode::logical_xor: {
                const int8_t b = pop();
                const int8_t a = pop();
                if (a < 0 || b < 0) stack.push_back(-1);
                else stack.push_back(static_cast<int8_t>((a != 0) ^ (b != 0)));
                break;
            }
        }
    }

    if (stack.size() != 1 || stack.back() < 0) return false;
    value = stack.back();
    return true;
}

static bool evalPowerExprActivity(const PowerExpr& expr,
                                  const LibertyCell* cell,
                                  const gp::GPNode& node,
                                  const std::vector<CpuActivity>& pin_activity,
                                  float& density,
                                  float& duty) {
    std::vector<int> ports;
    for (const auto& op : expr.ops()) {
        if (op.opcode != PowerExprOpcode::port || op.port_id < 0) continue;
        if (std::find(ports.begin(), ports.end(), op.port_id) == ports.end())
            ports.push_back(op.port_id);
    }
    if (ports.size() > 16) return false;

    std::vector<float> duties;
    std::vector<float> densities;
    duties.reserve(ports.size());
    densities.reserve(ports.size());
    for (int port_id : ports) {
        if (!cell || port_id < 0 || port_id >= static_cast<int>(cell->ports_.size())) return false;
        const std::string& port_name = cell->ports_[port_id]->name;
        auto pin_itr = node.portMap.find(port_name);
        if (pin_itr == node.portMap.end()) return false;
        int pin_id = pin_itr->second;
        if (pin_id < 0 || pin_id >= static_cast<int>(pin_activity.size())) return false;
        duties.push_back(std::clamp(pin_activity[pin_id].duty, 0.0f, 1.0f));
        densities.push_back(pin_activity[pin_id].density);
    }

    auto eval_assignment = [&](uint64_t bits, int force_var, int force_val, int8_t& value) -> bool {
        std::vector<int8_t> port_values(cell ? cell->ports_.size() : 0, -1);
        for (size_t i = 0; i < ports.size(); i++) {
            int bit = (bits >> i) & 1ULL;
            if (static_cast<int>(i) == force_var) bit = force_val;
            port_values[ports[i]] = static_cast<int8_t>(bit);
        }
        return evalPowerExprWithPortValues(expr, port_values, value);
    };

    const uint64_t states = 1ULL << ports.size();
    float true_duty = 0.0f;
    float false_duty = 0.0f;
    for (uint64_t bits = 0; bits < states; bits++) {
        float prob = 1.0f;
        for (size_t i = 0; i < ports.size(); i++) {
            prob *= ((bits >> i) & 1ULL) ? duties[i] : (1.0f - duties[i]);
        }
        int8_t value = -1;
        if (!eval_assignment(bits, -1, 0, value)) return false;
        if (value) true_duty += prob;
        else false_duty += prob;
    }
    // OpenSTA evaluates BDDs recursively. A naive true-minterm sum can round
    // near-1 duties to exactly 1.0 (for example OR chains), which erases rare
    // sensitization paths. Use the smaller complement side when it is more
    // stable: duty = 1 - P(function is false).
    duty = (false_duty < true_duty) ? (1.0f - false_duty) : true_duty;
    duty = std::clamp(duty, 0.0f, 1.0f);

    density = 0.0f;
    for (size_t var = 0; var < ports.size(); var++) {
        float diff_true = 0.0f;
        float diff_false = 0.0f;
        for (uint64_t bits = 0; bits < states; bits++) {
            // Boolean difference is a function of all variables except `var`.
            // Enumerate each cofactor assignment once by fixing var's bit in the
            // iteration mask to zero; eval_assignment() then forces var to 0/1.
            if ((bits >> var) & 1ULL) continue;
            float prob = 1.0f;
            for (size_t i = 0; i < ports.size(); i++) {
                if (i == var) continue;
                prob *= ((bits >> i) & 1ULL) ? duties[i] : (1.0f - duties[i]);
            }
            int8_t value0 = -1, value1 = -1;
            if (!eval_assignment(bits, static_cast<int>(var), 0, value0)) return false;
            if (!eval_assignment(bits, static_cast<int>(var), 1, value1)) return false;
            if (value0 != value1) diff_true += prob;
            else diff_false += prob;
        }
        float diff_duty = (diff_false < diff_true) ? (1.0f - diff_false) : diff_true;
        diff_duty = std::clamp(diff_duty, 0.0f, 1.0f);
        density += densities[var] * diff_duty;
    }

    return std::isfinite(density) && std::isfinite(duty);
}

}  // namespace

tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> GPUTimer::report_power_liberty_inventory() {
    int64_t internal_groups = static_cast<int64_t>(gtdb.liberty_internal_powers.size());
    int64_t internal_rise_luts = 0;
    int64_t internal_fall_luts = 0;
    int64_t internal_when_exprs = 0;
    for (auto* internal_power : gtdb.liberty_internal_powers) {
        if (!internal_power) continue;
        if (internal_power->power_[RISE] && internal_power->power_[RISE]->set_) internal_rise_luts++;
        if (internal_power->power_[FALL] && internal_power->power_[FALL]->set_) internal_fall_luts++;
        if (!internal_power->when_expr_.empty()) internal_when_exprs++;
    }

    int64_t leakage_groups = static_cast<int64_t>(gtdb.liberty_leakage_powers.size());
    int64_t leakage_when_exprs = 0;
    for (auto* leakage_power : gtdb.liberty_leakage_powers) {
        if (leakage_power && !leakage_power->when_expr_.empty()) leakage_when_exprs++;
    }

    int64_t output_functions = 0;
    for (uint8_t has_function : gtdb.liberty_port_has_function) {
        if (has_function) output_functions++;
    }

    return {internal_groups, internal_rise_luts, internal_fall_luts, internal_when_exprs, leakage_groups, leakage_when_exprs, output_functions};
}

int64_t GPUTimer::report_power_seq_inventory() {
    int64_t seqs = 0;
    for (const auto* cell_type : gtdb.rawdb.celltypes) {
        if (cell_type && cell_type->liberty_cell)
            seqs += static_cast<int64_t>(cell_type->liberty_cell->sequentials_.size());
    }
    return seqs;
}

torch::Tensor GPUTimer::report_power_activity_cpu() {
    const int n = static_cast<int>(gtdb.pin_names.size());
    std::vector<CpuActivity> act(n);

    float min_period_sec = std::numeric_limits<float>::infinity();
    for (const auto& kv : gtdb.clocks) {
        float period = kv.second.period();
        float period_sec = period;
        if (gtdb.sdc_time_unit.has_value()) period_sec *= *gtdb.sdc_time_unit;
        else period_sec *= gtdb.time_unit;
        if (period_sec > 0.0f) min_period_sec = std::min(min_period_sec, period_sec);
    }
    if (!std::isfinite(min_period_sec) || min_period_sec <= 0.0f) {
        min_period_sec = gtdb.time_unit > 0.0f ? gtdb.time_unit : 1.0e-9f;
    }
    const float default_density = 0.1f / min_period_sec;
    const float clock_density = 2.0f / min_period_sec;

    std::vector<int> pin_level(n, 0);
    int max_pin_level = 0;
    if (gtdb.pin_num_fanin.size() == static_cast<size_t>(n)
        && gtdb.pin_fanout_list_end.size() == static_cast<size_t>(n + 1)) {
        std::vector<int> indeg = gtdb.pin_num_fanin;
        std::deque<int> frontier;
        for (int i = 0; i < n; i++) {
            if (indeg[i] == 0) frontier.push_back(i);
        }
        std::vector<uint8_t> seen(n, 0);
        while (!frontier.empty()) {
            int pin_id = frontier.front();
            frontier.pop_front();
            if (pin_id < 0 || pin_id >= n || seen[pin_id]) continue;
            seen[pin_id] = 1;
            max_pin_level = std::max(max_pin_level, pin_level[pin_id]);
            int start = gtdb.pin_fanout_list_end[pin_id];
            int end = gtdb.pin_fanout_list_end[pin_id + 1];
            for (int idx = start; idx < end; idx++) {
                int fanout = gtdb.pin_fanout_list[idx];
                if (fanout < 0 || fanout >= n) continue;
                pin_level[fanout] = std::max(pin_level[fanout], pin_level[pin_id] + 1);
                if (--indeg[fanout] == 0) frontier.push_back(fanout);
            }
        }
    }
    std::vector<std::deque<int>> level_queues(max_pin_level + 2);
    std::vector<uint8_t> in_queue(n, 0);
    std::vector<uint8_t> force_propagate_on_visit(n, 0);
    std::vector<int> pin_to_node(n, -1);
    std::vector<int> pin_to_net(n, -1);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id >= 0 && pin_id < n) {
            pin_to_node[pin_id] = static_cast<int>(pin.getParNodeId());
            pin_to_net[pin_id] = static_cast<int>(pin.getParNetId());
        }
    }

    auto enqueue = [&](int pin_id, bool force_propagate = false) {
        if (pin_id < 0 || pin_id >= n) return;
        if (force_propagate) force_propagate_on_visit[pin_id] = 1;
        if (in_queue[pin_id]) return;
        int level = std::clamp(pin_level[pin_id], 0, max_pin_level + 1);
        level_queues[level].push_back(pin_id);
        in_queue[pin_id] = 1;
    };

    auto percent_change = [](float value, float prev) -> float {
        if (prev == 0.0f) return value == 0.0f ? 0.0f : 1.0f;
        return std::abs(value - prev) / std::abs(prev);
    };

    std::string trace_pin_name;
    if (const char* trace_name = std::getenv("XPLACE_POWER_ACTIVITY_TRACE_PIN")) trace_pin_name = trace_name;
    auto trace_matches = [&](int pin_id) -> bool {
        if (trace_pin_name.empty() || pin_id < 0 || pin_id >= n) return false;
        std::string name = gtdb.pin_names[pin_id];
        std::replace(name.begin(), name.end(), ':', '/');
        return gtdb.pin_names[pin_id] == trace_pin_name || name == trace_pin_name;
    };

    auto set_activity = [&](int pin_id, float density, float duty, int origin, bool force, bool enqueue_on_change = true) -> bool {
        if (pin_id < 0 || pin_id >= n) return false;
        const float prev_density = act[pin_id].density;
        const float prev_duty = act[pin_id].duty;
        const int prev_origin = act[pin_id].origin;
        const float duty_clamped = std::clamp(duty, 0.0f, 1.0f);
        float density_clamped = std::clamp(density, 0.0f, clock_density);
        // Match OpenSTA PwrActivity::check(): densities below 1e-10 are
        // numerical noise and are clipped to zero before change detection.
        if (std::abs(density_clamped) < 1.0e-10f) density_clamped = 0.0f;
        const bool changed = percent_change(density_clamped, prev_density) > 0.01f
            || percent_change(duty_clamped, prev_duty) > 0.01f
            || prev_origin != origin;
        act[pin_id].density = density_clamped;
        act[pin_id].duty = duty_clamped;
        act[pin_id].origin = origin;
        if (trace_matches(pin_id)) {
            std::cerr << "[power_activity_trace_set] pin=" << trace_pin_name
                      << " level=" << pin_level[pin_id]
                      << " prev_density=" << prev_density
                      << " prev_duty=" << prev_duty
                      << " density=" << density_clamped
                      << " duty=" << duty_clamped
                      << " origin=" << origin
                      << " changed=" << changed
                      << " enqueue=" << (changed && enqueue_on_change)
                      << std::endl;
        }
        if (changed && enqueue_on_change) enqueue(pin_id, true);
        return changed;
    };

    auto normalize_expr = [](std::string expr) {
        expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());
        if (expr.size() >= 2 && expr.front() == '"' && expr.back() == '"')
            expr = expr.substr(1, expr.size() - 2);
        return expr;
    };

    std::vector<uint8_t> pending_reg_flag(gtdb.gpdb.getNodes().size(), 0);
    std::vector<int> pending_regs;
    auto mark_pending_reg = [&](int node_id) {
        if (node_id < 0 || node_id >= static_cast<int>(pending_reg_flag.size())) return;
        if (pending_reg_flag[node_id]) return;
        pending_reg_flag[node_id] = 1;
        pending_regs.push_back(node_id);
    };

    auto get_cell = [&](int node_id) -> LibertyCell* {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return nullptr;
        int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
        auto* cell_type = gtdb.rawdb.celltypes[libcell_id];
        return cell_type ? cell_type->liberty_cell : nullptr;
    };
    auto is_io_node = [&](int node_id) -> bool {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return false;
        const std::string& node_type = gtdb.gpdb.getNodes()[node_id].getNodeType();
        return node_type == "IOPin" || node_type == "FloatIOPin";
    };

    std::vector<uint8_t> is_load_pin(n, 0);
    std::vector<uint8_t> is_driver_pin(n, 0);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id < 0 || pin_id >= n) continue;
        int node_id = pin_to_node[pin_id];
        if (is_io_node(node_id)
            && std::find(gtdb.primary_inputs.begin(), gtdb.primary_inputs.end(), pin_id) != gtdb.primary_inputs.end()) {
            is_driver_pin[pin_id] = 1;
            continue;
        }
        if (is_io_node(node_id)
            && std::find(gtdb.primary_outputs.begin(), gtdb.primary_outputs.end(), pin_id) != gtdb.primary_outputs.end()) {
            is_load_pin[pin_id] = 1;
            continue;
        }
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
        LibertyPort* port = cell->ports_[port_offset];
        if (!port) continue;
        if (port->direction_ == CellPortDirection::input) is_load_pin[pin_id] = 1;
        else if (port->direction_ == CellPortDirection::output) is_driver_pin[pin_id] = 1;
    }
    std::vector<uint8_t> is_seq_output_pin(n, 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n || !is_driver_pin[pin_id]) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if ((!seq_out.empty() && func == seq_out) ||
                    (!seq_out_inv.empty() && func == seq_out_inv)) {
                    is_seq_output_pin[pin_id] = 1;
                }
            }
        }
    }

    std::vector<int> net_driver_pin(gtdb.gpdb.getNets().size(), -1);
    for (const auto& net : gtdb.gpdb.getNets()) {
        const int net_id = static_cast<int>(net.getId());
        if (net_id < 0 || net_id >= static_cast<int>(net_driver_pin.size())) continue;
        for (int pin_id : net.pins()) {
            if (pin_id >= 0 && pin_id < n && is_driver_pin[pin_id]) {
                net_driver_pin[net_id] = pin_id;
                break;
            }
        }
    }
    std::vector<int> clock_gate_out_for_input(n, -1);
    std::vector<int> clock_gate_clock_for_out(n, -1);
    std::vector<int> clock_gate_enable_for_out(n, -1);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        int clk_pin = -1;
        int enable_pin = -1;
        int out_pin = -1;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port) continue;
            if (port->is_clock_gate_clock_) clk_pin = pin_id;
            if (port->is_clock_gate_enable_) enable_pin = pin_id;
            if (port->is_clock_gate_out_) out_pin = pin_id;
        }
        if (out_pin >= 0 && clk_pin >= 0 && enable_pin >= 0) {
            clock_gate_clock_for_out[out_pin] = clk_pin;
            clock_gate_enable_for_out[out_pin] = enable_pin;
            clock_gate_out_for_input[clk_pin] = out_pin;
            clock_gate_out_for_input[enable_pin] = out_pin;
        }
    }

    auto build_clock_pins = [&]() {
        const int num_nets = static_cast<int>(gtdb.gpdb.getNets().size());
        std::vector<uint8_t> is_clock_net(num_nets, 0);
        auto mark_net = [&](int net_id) -> bool {
            if (net_id < 0 || net_id >= num_nets || is_clock_net[net_id]) return false;
            is_clock_net[net_id] = 1;
            return true;
        };

        if (gtdb.net_is_clock.size() == static_cast<size_t>(num_nets)) {
            for (int net_id = 0; net_id < num_nets; net_id++) {
                if (gtdb.net_is_clock[net_id]) mark_net(net_id);
            }
        }
        if (gtdb.pin_is_clk.size() == static_cast<size_t>(n)) {
            for (int pin_id = 0; pin_id < n; pin_id++) {
                if (gtdb.pin_is_clk[pin_id]) mark_net(pin_to_net[pin_id]);
            }
        }
        std::vector<int> clock_pins;
        for (int net_id = 0; net_id < num_nets; net_id++) {
            if (!is_clock_net[net_id]) continue;
            for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
                if (pin_id >= 0 && pin_id < n) clock_pins.push_back(pin_id);
            }
        }
        std::sort(clock_pins.begin(), clock_pins.end());
        clock_pins.erase(std::unique(clock_pins.begin(), clock_pins.end()), clock_pins.end());
        return clock_pins;
    };

    auto enqueue_adjacent_vertices = [&](int pin_id) {
        if (pin_id < 0 || pin_id >= n) return;
        if (gtdb.pin_forward_arc_list_end.size() != static_cast<size_t>(n + 1)) return;
        int start = gtdb.pin_forward_arc_list_end[pin_id];
        int end = gtdb.pin_forward_arc_list_end[pin_id + 1];
        for (int idx = start; idx < end; idx++) {
            int arc_id = gtdb.pin_forward_arc_list[idx];
            if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) continue;
            if (arc_id < static_cast<int>(gtdb.arc_id2test_id.size()) && gtdb.arc_id2test_id[arc_id] != -1) continue;  // timing checks.
            int to_pin = gtdb.timing_arc_to_pin_id[arc_id];
            if (std::getenv("XPLACE_POWER_ACTIVITY_SKIP_BACK_LEVEL_ARCS")
                && arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1
                && to_pin >= 0 && to_pin < n && pin_level[to_pin] <= pin_level[pin_id])
                continue;
            // OpenSTA ActivitySrchPred excludes reg clk->Q/latch D->Q. Xplace does not expose
            // TimingRole here, so conservatively skip cell arcs into sequential outputs; they
            // are propagated only by seed_reg_outputs().
            if (arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1) {
                int to_node = to_pin >= 0 && to_pin < n ? pin_to_node[to_pin] : -1;
                LibertyCell* to_cell = get_cell(to_node);
                if (to_cell && !to_cell->sequentials_.empty() && to_pin >= 0 && to_pin < n && is_driver_pin[to_pin])
                    continue;
            }
            enqueue(to_pin);
        }
    };

    auto eval_cell_outputs = [&](int node_id) {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell) return;

        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;

            PowerExpr expr;
            if (!expr.compile(port->function_expr_, cell)) continue;
            float density = 0.0f;
            float duty = 0.0f;
            if (evalPowerExprActivity(expr, cell, node, act, density, duty)) {
                set_activity(pin_id, density, duty, 3, false);
            }
        }
    };

    auto eval_output_pin_activity = [&](int pin_id, bool& changed) -> bool {
        changed = false;
        int node_id = pin_id >= 0 && pin_id < n ? pin_to_node[pin_id] : -1;
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return false;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell) return false;
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) return false;
        LibertyPort* port = cell->ports_[port_offset];
        if (!port || port->direction_ != CellPortDirection::output) return false;
        bool computed = false;
        if (port->has_function_) {
            PowerExpr expr;
            if (expr.compile(port->function_expr_, cell)) {
                float density = 0.0f;
                float duty = 0.0f;
                if (evalPowerExprActivity(expr, cell, node, act, density, duty)) {
                    changed = set_activity(pin_id, density, duty, 3, false, false);
                    computed = true;
                }
            }
        }
        const int cg_clk = clock_gate_clock_for_out[pin_id];
        const int cg_en = clock_gate_enable_for_out[pin_id];
        if (cg_clk >= 0 && cg_en >= 0) {
            const float density = act[cg_clk].density * act[cg_en].duty +
                                  act[cg_en].density * act[cg_clk].duty;
            const float duty = act[cg_clk].duty * act[cg_en].duty;
            changed = set_activity(pin_id, density, duty, 3, false, false) || changed;
            computed = true;
        }
        return computed;
    };

    auto seed_reg_outputs = [&](int node_id) {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) return;

        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            PowerExpr data_expr;
            PowerExpr clk_expr;
            if (!data_expr.compile(seq->next_state_expr_, cell)) continue;
            if (!clk_expr.compile(seq->clocked_on_expr_, cell)) continue;

            float in_density = 0.0f, in_duty = 0.0f;
            float clk_density_eval = 0.0f, clk_duty = 0.5f;
            if (!evalPowerExprActivity(data_expr, cell, node, act, in_density, in_duty)) continue;
            if (!evalPowerExprActivity(clk_expr, cell, node, act, clk_density_eval, clk_duty)) {
                clk_density_eval = clock_density;
                clk_duty = 0.5f;
            }

            float out_density = in_density;
            float out_duty = in_duty;
            if (in_density > clk_density_eval / 2.0f) {
                if (!seq->is_latch_)
                    out_density = 2.0f * in_duty * (1.0f - in_duty) * clk_density_eval;
                else
                    out_density = in_density * clk_duty;
            }

            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if (!seq_out.empty() && func == seq_out) {
                    set_activity(pin_id, out_density, out_duty, 3, false);
                } else if (!seq_out_inv.empty() && func == seq_out_inv) {
                    const float inv_duty = 1.0f - out_duty;
                    set_activity(pin_id, out_density, inv_duty, 3, false);
                }
            }
        }
    };

    bool level_lifo = true;
    if (const char* order_env = std::getenv("XPLACE_POWER_ACTIVITY_LEVEL_ORDER")) {
        std::string order(order_env);
        std::transform(order.begin(), order.end(), order.begin(), [](unsigned char c) { return std::tolower(c); });
        if (order == "fifo") level_lifo = false;
    }

    auto run_queue = [&]() {
        for (int level = 0; level <= max_pin_level + 1; level++) {
            auto& queue = level_queues[level];
            while (!queue.empty()) {
                int pin_id;
                if (level_lifo) {
                    pin_id = queue.back();
                    queue.pop_back();
                } else {
                    pin_id = queue.front();
                    queue.pop_front();
                }
                bool force_visit = force_propagate_on_visit[pin_id] != 0;
                force_propagate_on_visit[pin_id] = 0;
                in_queue[pin_id] = 0;

                bool changed = false;
                if (is_load_pin[pin_id]) {
                    int net_id = pin_to_net[pin_id];
                    const int driver_pin = (net_id >= 0 && net_id < static_cast<int>(net_driver_pin.size()))
                        ? net_driver_pin[net_id] : -1;
                    if (driver_pin >= 0 && driver_pin < n && driver_pin != pin_id) {
                        if (trace_matches(pin_id)) {
                            std::cerr << "[power_activity_trace_net_sink] sink=" << trace_pin_name
                                      << " from=" << gtdb.pin_names[driver_pin]
                                      << " driver_level=" << pin_level[driver_pin]
                                      << " sink_level=" << pin_level[pin_id]
                                      << " density=" << act[driver_pin].density
                                      << " duty=" << act[driver_pin].duty
                                      << std::endl;
                        }
                        changed = set_activity(pin_id, act[driver_pin].density, act[driver_pin].duty, 3, false, false);
                    }
                }

                if (is_driver_pin[pin_id]) {
                    bool output_changed = false;
                    bool computed = eval_output_pin_activity(pin_id, output_changed);
                    if (computed)
                        changed = changed || output_changed;
                    else
                        changed = changed || force_visit;
                }

                if (changed) {
                    int node_id = pin_id >= 0 && pin_id < n ? pin_to_node[pin_id] : -1;
                    LibertyCell* cell = get_cell(node_id);
                    if (is_load_pin[pin_id] && cell && !cell->sequentials_.empty())
                        mark_pending_reg(node_id);
                    if (is_load_pin[pin_id] && pin_id >= 0 && pin_id < n &&
                        clock_gate_out_for_input[pin_id] >= 0) {
                        enqueue(clock_gate_out_for_input[pin_id]);
                    }
                    enqueue_adjacent_vertices(pin_id);
                }
            }
        }
    };

    std::vector<int> clock_pins = build_clock_pins();
    std::vector<uint8_t> is_clock_pin(n, 0);
    for (int pin_id : clock_pins) {
        if (pin_id >= 0 && pin_id < n) is_clock_pin[pin_id] = 1;
    }
    std::vector<uint8_t> is_primary_input(n, 0);
    for (int pin_id : gtdb.primary_inputs) {
        if (pin_id >= 0 && pin_id < n) is_primary_input[pin_id] = 1;
        if (pin_id >= 0 && pin_id < n && is_driver_pin[pin_id]
            && set_activity(pin_id, default_density, 0.5f, 1, false, false))
            enqueue_adjacent_vertices(pin_id);
    }

    for (int pin_id : gtdb.pin_frontiers) {
        if (pin_id < 0 || pin_id >= n) continue;
        if (is_primary_input[pin_id] || is_clock_pin[pin_id]) continue;
        if (set_activity(pin_id, default_density, 0.5f, 1, false, false))
            enqueue_adjacent_vertices(pin_id);
    }

    // Constant-generator outputs are roots in OpenSTA's power graph.
    for (int pin_id = 0; pin_id < n; pin_id++) {
        if (!is_driver_pin[pin_id] || is_primary_input[pin_id] || is_clock_pin[pin_id]) continue;
        int node_id = pin_to_node[pin_id];
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
        bool has_input_pin = false;
        for (int node_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
            if (node_pin >= 0 && node_pin < n && is_load_pin[node_pin]) {
                has_input_pin = true;
                break;
            }
        }
        if (!has_input_pin && set_activity(pin_id, default_density, 0.5f, 1, false, false))
            enqueue_adjacent_vertices(pin_id);
    }

    for (int pin_id : clock_pins) {
        if (set_activity(pin_id, clock_density, 0.5f, 2, true, false))
            enqueue_adjacent_vertices(pin_id);
    }

    // Initial combinational propagation from roots/clock network.
    run_queue();

    int max_activity_passes = 50;
    if (const char* env = std::getenv("XPLACE_POWER_ACTIVITY_MAX_PASSES")) {
        max_activity_passes = std::max(1, std::atoi(env));
    }
    auto trace_pending_regs = [&](int pass) {
        if (!std::getenv("XPLACE_POWER_ACTIVITY_TRACE_REGS")) return;
        for (int node_id : pending_regs) {
            if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
            std::cerr << "[power_activity_reg] pass=" << pass
                      << " node=" << node_id
                      << " inst=" << gtdb.gpdb.getNodes()[node_id].getName()
                      << std::endl;
        }
    };
    int trace_pin_id = -1;
    if (const char* trace_name = std::getenv("XPLACE_POWER_ACTIVITY_TRACE_PIN")) {
        std::string trace(trace_name);
        for (int i = 0; i < n; i++) {
            std::string name = gtdb.pin_names[i];
            std::replace(name.begin(), name.end(), ':', '/');
            if (gtdb.pin_names[i] == trace || name == trace) {
                trace_pin_id = i;
                break;
            }
        }
        if (trace_pin_id >= 0) {
            std::cerr << "[power_activity_trace] pass=0 pin=" << trace_name << " density=" << act[trace_pin_id].density << " duty=" << act[trace_pin_id].duty << " pending=" << pending_regs.size() << std::endl;
        }
    }
    trace_pending_regs(0);
    for (int pass = 1; !pending_regs.empty() && pass < max_activity_passes; pass++) {
        std::vector<int> regs = std::move(pending_regs);
        pending_regs.clear();
        for (int node_id : regs) {
            if (node_id >= 0 && node_id < static_cast<int>(pending_reg_flag.size()))
                pending_reg_flag[node_id] = 0;
            seed_reg_outputs(node_id);
        }
        run_queue();
        trace_pending_regs(pass);
        if (trace_pin_id >= 0) {
            std::cerr << "[power_activity_trace] pass=" << pass << " density=" << act[trace_pin_id].density << " duty=" << act[trace_pin_id].duty << " pending=" << pending_regs.size() << std::endl;
        }
    }

    auto out = torch::empty({n, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto acc = out.accessor<float, 2>();
    for (int i = 0; i < n; i++) {
        acc[i][0] = act[i].density;
        acc[i][1] = act[i].duty;
        acc[i][2] = static_cast<float>(act[i].origin);
    }
    return out;
}

torch::Tensor GPUTimer::compute_power_activity_cuda(torch::Tensor* inst_switching_cpu, torch::Tensor* pin_switching_cpu, torch::Tensor* inst_internal_cpu, torch::Tensor* internal_row_power_cpu, torch::Tensor* internal_row_meta_cpu, torch::Tensor* inst_leakage_cpu, torch::Tensor* leakage_row_power_cpu, torch::Tensor* leakage_row_meta_cpu) {
    const int n = static_cast<int>(gtdb.pin_names.size());
    if (n <= 0) return torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("report_power_activity_cuda requires CUDA");
    }
    // Some existing init kernels leave a stale CUDA error status that CPU reports ignore.
    // Clear it before allocating/uploading the Plan-A power activity data structures.
    clear_power_cuda_error();

    float min_period_sec = std::numeric_limits<float>::infinity();
    for (auto& kv : gtdb.clocks) {
        float period_sec = kv.second.period() * gtdb.time_unit;
        if (period_sec > 0.0f) min_period_sec = std::min(min_period_sec, period_sec);
    }
    if (!std::isfinite(min_period_sec) || min_period_sec <= 0.0f)
        min_period_sec = gtdb.time_unit > 0.0f ? gtdb.time_unit : 1.0e-9f;
    const float default_density = 0.1f / min_period_sec;
    const float clock_density = 2.0f / min_period_sec;

    std::vector<int> h_pin_to_node(n, -1);
    std::vector<int> h_pin_to_net(n, -1);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id >= 0 && pin_id < n) {
            h_pin_to_node[pin_id] = static_cast<int>(pin.getParNodeId());
            h_pin_to_net[pin_id] = static_cast<int>(pin.getParNetId());
        }
    }

    auto get_cell = [&](int node_id) -> LibertyCell* {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return nullptr;
        int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
        auto* cell_type = gtdb.rawdb.celltypes[libcell_id];
        return cell_type ? cell_type->liberty_cell : nullptr;
    };
    auto is_io_node = [&](int node_id) -> bool {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return false;
        const std::string& node_type = gtdb.gpdb.getNodes()[node_id].getNodeType();
        return node_type == "IOPin" || node_type == "FloatIOPin";
    };
    auto normalize_expr = [](std::string expr) {
        expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());
        if (expr.size() >= 2 && expr.front() == '"' && expr.back() == '"') expr = expr.substr(1, expr.size() - 2);
        return expr;
    };

    std::vector<uint8_t> h_is_load_pin(n, 0), h_is_driver_pin(n, 0), h_is_cell_pin(n, 0), h_is_seq_output_pin(n, 0);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id < 0 || pin_id >= n) continue;
        const int node_id = h_pin_to_node[pin_id];
        if (!is_io_node(node_id)) h_is_cell_pin[pin_id] = 1;
        if (is_io_node(node_id)
            && std::find(gtdb.primary_inputs.begin(), gtdb.primary_inputs.end(), pin_id) != gtdb.primary_inputs.end()) {
            h_is_driver_pin[pin_id] = 1;
            continue;
        }
        if (is_io_node(node_id)
            && std::find(gtdb.primary_outputs.begin(), gtdb.primary_outputs.end(), pin_id) != gtdb.primary_outputs.end()) {
            h_is_load_pin[pin_id] = 1;
            continue;
        }
        LibertyCell* cell = get_cell(node_id);
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (!cell || port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
        LibertyPort* port = cell->ports_[port_offset];
        if (!port) continue;
        if (port->direction_ == CellPortDirection::input) h_is_load_pin[pin_id] = 1;
        else if (port->direction_ == CellPortDirection::output) h_is_driver_pin[pin_id] = 1;
    }

    std::vector<int> h_net_driver_pin(gtdb.gpdb.getNets().size(), -1);
    for (const auto& net : gtdb.gpdb.getNets()) {
        const int net_id = static_cast<int>(net.getId());
        if (net_id < 0 || net_id >= static_cast<int>(h_net_driver_pin.size())) continue;
        for (int pin_id : net.pins()) {
            if (pin_id >= 0 && pin_id < n && h_is_driver_pin[pin_id]) {
                h_net_driver_pin[net_id] = pin_id;
                break;
            }
        }
    }
    std::vector<int> h_clock_gate_out_for_input(n, -1);
    std::vector<int> h_clock_gate_clock_for_out(n, -1);
    std::vector<int> h_clock_gate_enable_for_out(n, -1);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        int clk_pin = -1;
        int enable_pin = -1;
        int out_pin = -1;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port) continue;
            if (port->is_clock_gate_clock_) clk_pin = pin_id;
            if (port->is_clock_gate_enable_) enable_pin = pin_id;
            if (port->is_clock_gate_out_) out_pin = pin_id;
        }
        if (out_pin >= 0 && clk_pin >= 0 && enable_pin >= 0) {
            h_clock_gate_clock_for_out[out_pin] = clk_pin;
            h_clock_gate_enable_for_out[out_pin] = enable_pin;
            h_clock_gate_out_for_input[clk_pin] = out_pin;
            h_clock_gate_out_for_input[enable_pin] = out_pin;
        }
    }

    auto build_clock_pins = [&]() {
        const int num_nets = static_cast<int>(gtdb.gpdb.getNets().size());
        std::vector<uint8_t> is_clock_net(num_nets, 0);
        auto mark_net = [&](int net_id) -> bool {
            if (net_id < 0 || net_id >= num_nets || is_clock_net[net_id]) return false;
            is_clock_net[net_id] = 1;
            return true;
        };

        if (gtdb.net_is_clock.size() == static_cast<size_t>(num_nets)) {
            for (int net_id = 0; net_id < num_nets; net_id++) {
                if (gtdb.net_is_clock[net_id]) mark_net(net_id);
            }
        }
        if (gtdb.pin_is_clk.size() == static_cast<size_t>(n)) {
            for (int pin_id = 0; pin_id < n; pin_id++) {
                if (gtdb.pin_is_clk[pin_id]) mark_net(h_pin_to_net[pin_id]);
            }
        }
        std::vector<int> clock_pins;
        for (int net_id = 0; net_id < num_nets; net_id++) {
            if (!is_clock_net[net_id]) continue;
            for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
                if (pin_id >= 0 && pin_id < n) clock_pins.push_back(pin_id);
            }
        }
        std::sort(clock_pins.begin(), clock_pins.end());
        clock_pins.erase(std::unique(clock_pins.begin(), clock_pins.end()), clock_pins.end());
        return clock_pins;
    };
    std::vector<int> h_clock_pins = build_clock_pins();

    std::vector<GpuPowerExprOpHost> h_expr_ops;
    std::vector<int> h_expr_start;
    std::vector<int> h_expr_count;
    auto add_expr = [&](const std::string& expr_str, LibertyCell* cell, const gp::GPNode& node) -> int {
        if (!cell) return -1;
        PowerExpr expr;
        if (!expr.compile(expr_str, cell)) return -1;
        std::vector<GpuPowerExprOpHost> local_ops;
        local_ops.reserve(expr.ops().size());
        for (const auto& op : expr.ops()) {
            GpuPowerExprOpHost out;
            switch (op.opcode) {
                case PowerExprOpcode::port: {
                    if (op.port_id < 0 || op.port_id >= static_cast<int>(cell->ports_.size())) return -1;
                    const std::string& port_name = cell->ports_[op.port_id]->name;
                    auto pin_itr = node.portMap.find(port_name);
                    if (pin_itr == node.portMap.end()) return -1;
                    out.op = 0;
                    out.arg = pin_itr->second;
                    break;
                }
                case PowerExprOpcode::const_zero: out.op = 1; out.arg = -1; break;
                case PowerExprOpcode::const_one: out.op = 2; out.arg = -1; break;
                case PowerExprOpcode::logical_not: out.op = 3; out.arg = -1; break;
                case PowerExprOpcode::logical_and: out.op = 4; out.arg = -1; break;
                case PowerExprOpcode::logical_or: out.op = 5; out.arg = -1; break;
                case PowerExprOpcode::logical_xor: out.op = 6; out.arg = -1; break;
            }
            local_ops.push_back(out);
        }
        if (local_ops.empty()) return -1;
        int expr_id = static_cast<int>(h_expr_start.size());
        h_expr_start.push_back(static_cast<int>(h_expr_ops.size()));
        h_expr_count.push_back(static_cast<int>(local_ops.size()));
        h_expr_ops.insert(h_expr_ops.end(), local_ops.begin(), local_ops.end());
        return expr_id;
    };

    auto expr_contains_pin = [&](int expr_id, int pin_id) -> bool {
        if (expr_id < 0 || expr_id >= static_cast<int>(h_expr_start.size())) return false;
        const int start = h_expr_start[expr_id];
        const int count = h_expr_count[expr_id];
        for (int k = 0; k < count; ++k) {
            if (h_expr_ops[start + k].op == 0 && h_expr_ops[start + k].arg == pin_id) return true;
        }
        return false;
    };

    std::vector<int> h_pin_func_expr_id(n, -1);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n || !h_is_driver_pin[pin_id]) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
            h_pin_func_expr_id[pin_id] = add_expr(port->function_expr_, cell, node);
        }
    }

    std::vector<GpuPowerSeqHost> h_seqs;
    std::vector<std::vector<int>> node_seq_ids(gtdb.gpdb.getNodes().size());
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            GpuPowerSeqHost rec;
            rec.data_expr_id = add_expr(seq->next_state_expr_, cell, node);
            rec.clk_expr_id = add_expr(seq->clocked_on_expr_, cell, node);
            rec.is_latch = seq->is_latch_ ? 1 : 0;
            if (rec.data_expr_id < 0) continue;
            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if (!seq_out.empty() && func == seq_out) rec.q_pin = pin_id;
                else if (!seq_out_inv.empty() && func == seq_out_inv) rec.qn_pin = pin_id;
            }
            if (rec.q_pin < 0 && rec.qn_pin < 0) continue;
            int seq_id = static_cast<int>(h_seqs.size());
            h_seqs.push_back(rec);
            if (rec.q_pin >= 0) h_is_seq_output_pin[rec.q_pin] = 1;
            if (rec.qn_pin >= 0) h_is_seq_output_pin[rec.qn_pin] = 1;
            node_seq_ids[node_id].push_back(seq_id);
        }
    }

    std::vector<std::vector<int>> pin_seq_ids(n);
    for (int pin = 0; pin < n; pin++) {
        int node_id = h_pin_to_node[pin];
        if (node_id >= 0 && node_id < static_cast<int>(node_seq_ids.size()) && !node_seq_ids[node_id].empty()) {
            if (h_is_load_pin[pin]) pin_seq_ids[pin] = node_seq_ids[node_id];
        }
    }
    std::vector<int> h_pin_seq_list_start(n + 1, 0);
    std::vector<int> h_pin_seq_list;
    for (int pin = 0; pin < n; pin++) {
        h_pin_seq_list_start[pin] = static_cast<int>(h_pin_seq_list.size());
        h_pin_seq_list.insert(h_pin_seq_list.end(), pin_seq_ids[pin].begin(), pin_seq_ids[pin].end());
    }
    h_pin_seq_list_start[n] = static_cast<int>(h_pin_seq_list.size());

    std::vector<uint8_t> h_is_clock_pin(n, 0);
    for (int pin_id : h_clock_pins) if (pin_id >= 0 && pin_id < n) h_is_clock_pin[pin_id] = 1;

    const bool seed_seq_feedback_outputs =
        std::getenv("XPLACE_POWER_SEED_SEQ_FEEDBACK_OUTPUTS") != nullptr;
    const bool seed_seq_feedback_d_only =
        std::getenv("XPLACE_POWER_SEED_SEQ_FEEDBACK_D_ONLY") != nullptr;
    const bool init_seq_feedback_state =
        std::getenv("XPLACE_POWER_INIT_SEQ_FEEDBACK_STATE") != nullptr;
    const bool skip_all_seq_output_arcs =
        std::getenv("XPLACE_POWER_SKIP_ALL_SEQ_OUTPUT_ARCS") != nullptr;
    std::vector<int> h_power_arc_types = gtdb.arc_types;
    auto timing_arc_ptr = [&](int arc_id) -> TimingArc* {
        const int base = arc_id * 2;
        int timing_id = -1;
        if (base + static_cast<int>(MAX) < static_cast<int>(gtdb.timing_arc_id_map.size()))
            timing_id = gtdb.timing_arc_id_map[base + static_cast<int>(MAX)];
        if (timing_id < 0 &&
            base + static_cast<int>(MIN) < static_cast<int>(gtdb.timing_arc_id_map.size()))
            timing_id = gtdb.timing_arc_id_map[base + static_cast<int>(MIN)];
        if (timing_id < 0 || timing_id >= static_cast<int>(gtdb.liberty_timing_arcs.size()))
            return nullptr;
        return gtdb.liberty_timing_arcs[timing_id];
    };
    auto skip_seq_output_arc_for_power = [&](int arc_id, int from_pin, int to_pin) -> bool {
        if (to_pin < 0 || to_pin >= n || !h_is_seq_output_pin[to_pin]) return false;
        TimingArc* timing_arc = timing_arc_ptr(arc_id);
        if (timing_arc) {
            const TimingType type = timing_arc->timing_type_;
            if (type == TimingType::rising_edge || type == TimingType::falling_edge)
                return true;
            if (type == TimingType::clear || type == TimingType::preset)
                return false;
        }
        if (from_pin >= 0 && from_pin < static_cast<int>(gtdb.pin_is_clk.size()) && gtdb.pin_is_clk[from_pin])
            return true;
        return false;
    };
    for (int from_pin = 0; from_pin < n; ++from_pin) {
        if (from_pin + 1 >= static_cast<int>(gtdb.pin_forward_arc_list_end.size())) break;
        const int start = gtdb.pin_forward_arc_list_end[from_pin];
        const int end = gtdb.pin_forward_arc_list_end[from_pin + 1];
        for (int idx = start; idx < end; ++idx) {
            const int arc_id = gtdb.pin_forward_arc_list[idx];
            if (arc_id < 0 || arc_id >= static_cast<int>(h_power_arc_types.size())) continue;
            if (h_power_arc_types[arc_id] != 1) continue;
            if (arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) continue;
            const int to_pin = gtdb.timing_arc_to_pin_id[arc_id];
            if (to_pin < 0 || to_pin >= n || !h_is_seq_output_pin[to_pin]) continue;
            if (!skip_all_seq_output_arcs && !skip_seq_output_arc_for_power(arc_id, from_pin, to_pin))
                h_power_arc_types[arc_id] = 0;
        }
    }

    std::vector<float> h_power_clock_slews(n * NUM_ATTR, nanf(""));
    if (ideal_clock) {
        std::array<float, NUM_ATTR> fallback_clock_slews;
        fallback_clock_slews.fill(nanf(""));
        if (!gtdb.clock_transitions.empty()) {
            fallback_clock_slews = gtdb.clock_transitions.begin()->second;
        }
        for (float& slew : fallback_clock_slews) {
            if (!std::isfinite(slew)) slew = 0.0f;
        }
        for (int pin_id : h_clock_pins) {
            if (pin_id < 0 || pin_id >= n) continue;
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                float slew = nanf("");
                const int idx = pin_id * NUM_ATTR + attr;
                if (idx >= 0 && idx < static_cast<int>(gtdb.pin_clock_slews.size()))
                    slew = gtdb.pin_clock_slews[idx];
                if (!std::isfinite(slew)) slew = fallback_clock_slews[attr];
                h_power_clock_slews[idx] = slew;
            }
        }
    }

    std::vector<uint8_t> h_is_primary_input(n, 0);
    std::vector<int> h_primary_inputs;
    h_primary_inputs.reserve(gtdb.primary_inputs.size());
    int root_primary_count = 0;
    int root_zero_indeg_count = 0;
    int root_const_output_count = 0;
    int root_seq_feedback_count = 0;
    int state_seq_feedback_count = 0;
    std::vector<int> h_feedback_seed_pins;
    std::vector<int> h_feedback_seed_seqs;
    bool seed_default_inputs = true;
    if (const char* env = std::getenv("XPLACE_POWER_SEED_INPUTS")) {
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        seed_default_inputs = !(value.empty() || value == "0" || value == "false" || value == "no");
    }
    for (auto pin : gtdb.primary_inputs) {
        const int pin_id = static_cast<int>(pin);
        if (pin_id >= 0 && pin_id < n) h_is_primary_input[pin_id] = 1;
        if (seed_default_inputs && pin_id >= 0 && pin_id < n && h_is_driver_pin[pin_id]) {
            h_primary_inputs.push_back(pin_id);
            root_primary_count++;
        }
    }
    if (seed_default_inputs) {
        for (int pin_id : gtdb.pin_frontiers) {
            if (pin_id < 0 || pin_id >= n) continue;
            if (h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
            h_primary_inputs.push_back(pin_id);
            root_zero_indeg_count++;
        }
    }
    if (seed_default_inputs && (seed_seq_feedback_outputs || init_seq_feedback_state)) {
        std::vector<uint8_t> seed_seen(n, 0);
        std::vector<uint8_t> state_pin_seen(n, 0);
        std::vector<uint8_t> state_seq_seen(h_seqs.size(), 0);
        auto collect_feedback_data_pins = [&](int expr_id, int driver_pin, std::vector<int>* data_pins) -> bool {
            if (expr_id < 0 || driver_pin < 0 || driver_pin >= n) return false;
            const int driver_net = h_pin_to_net[driver_pin];
            if (driver_net < 0 || driver_net >= static_cast<int>(h_net_driver_pin.size())) return false;
            if (h_net_driver_pin[driver_net] != driver_pin) return false;
            bool matched = false;
            const int start = h_expr_start[expr_id];
            const int end = start + h_expr_count[expr_id];
            for (int op_i = start; op_i < end; ++op_i) {
                if (h_expr_ops[op_i].op != 0) continue;
                const int data_pin = h_expr_ops[op_i].arg;
                if (data_pin < 0 || data_pin >= n || h_pin_to_net[data_pin] != driver_net) continue;
                if (seed_seq_feedback_d_only) {
                    const int node_id = h_pin_to_node[data_pin];
                    LibertyCell* cell = get_cell(node_id);
                    const int port_offset = gtdb.pin_id2port_offset_id[data_pin];
                    if (!cell || port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size()))
                        continue;
                    LibertyPort* port = cell->ports_[port_offset];
                    if (!port || port->name != "D") continue;
                }
                if (data_pin >= 0 && data_pin < n) {
                    matched = true;
                    if (data_pins) data_pins->push_back(data_pin);
                }
            }
            return matched;
        };
        for (int seq_id = 0; seq_id < static_cast<int>(h_seqs.size()); ++seq_id) {
            const auto& seq = h_seqs[seq_id];
            std::vector<int> data_pins;
            const bool q_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.q_pin, &data_pins);
            if (q_feedback && seed_seq_feedback_outputs && !seed_seen[seq.q_pin]) {
                h_primary_inputs.push_back(seq.q_pin);
                seed_seen[seq.q_pin] = 1;
                root_seq_feedback_count++;
            }
            const bool qn_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.qn_pin, &data_pins);
            if (qn_feedback && seed_seq_feedback_outputs && !seed_seen[seq.qn_pin]) {
                h_primary_inputs.push_back(seq.qn_pin);
                seed_seen[seq.qn_pin] = 1;
                root_seq_feedback_count++;
            }
            if (init_seq_feedback_state && (q_feedback || qn_feedback)) {
                if (!state_seq_seen[seq_id]) {
                    h_feedback_seed_seqs.push_back(seq_id);
                    state_seq_seen[seq_id] = 1;
                    state_seq_feedback_count++;
                }
                for (int data_pin : data_pins) {
                    if (data_pin >= 0 && data_pin < n && !state_pin_seen[data_pin]) {
                        h_feedback_seed_pins.push_back(data_pin);
                        state_pin_seen[data_pin] = 1;
                    }
                }
            }
        }
    }
    // Constant-generator outputs are roots in OpenSTA's power graph.
    for (int pin_id = 0; pin_id < n; pin_id++) {
        if (!h_is_driver_pin[pin_id] || h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
        int node_id = h_pin_to_node[pin_id];
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
        bool has_input_pin = false;
        for (int node_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
            if (node_pin >= 0 && node_pin < n && h_is_load_pin[node_pin]) {
                has_input_pin = true;
                break;
            }
        }
        if (seed_default_inputs && !has_input_pin) {
            h_primary_inputs.push_back(pin_id);
            root_const_output_count++;
        }
    }
    std::sort(h_primary_inputs.begin(), h_primary_inputs.end());
    h_primary_inputs.erase(std::unique(h_primary_inputs.begin(), h_primary_inputs.end()), h_primary_inputs.end());
    if (std::getenv("XPLACE_POWER_PRINT_ROOT_STATS")) {
        std::fprintf(stderr,
                     "[power_activity_roots] seeds=%zu primary=%d timing_roots=%d const_outputs=%d\n",
                     h_primary_inputs.size(), root_primary_count, root_zero_indeg_count,
                     root_const_output_count);
        if (seed_seq_feedback_outputs) {
            std::fprintf(stderr,
                         "[power_activity_roots] seq_feedback=%d\n",
                         root_seq_feedback_count);
        }
        if (init_seq_feedback_state) {
            std::fprintf(stderr,
                         "[power_activity_roots] seq_feedback_state=%d pins=%zu\n",
                         state_seq_feedback_count, h_feedback_seed_pins.size());
        }
    }

    auto positive_unate_for_power = [](LibertyCell* cell, LibertyPort* from, LibertyPort* to) -> bool {
        if (!cell || !from || !to) return true;
        for (TimingArc* arc : from->timing_arcs_) {
            if (!arc || arc->to_port_ != to) continue;
            return arc->timing_sense_ == TimingSense::positive_unate ||
                   arc->timing_sense_ == TimingSense::non_unate ||
                   arc->timing_sense_ == TimingSense::unknown;
        }
        for (TimingArc* arc : to->timing_arcs_) {
            if (!arc || arc->from_port_ != from) continue;
            return arc->timing_sense_ == TimingSense::positive_unate ||
                   arc->timing_sense_ == TimingSense::non_unate ||
                   arc->timing_sense_ == TimingSense::unknown;
        }
        return true;
    };

    auto compile_when_expr = [&](InternalPower* ip, LibertyCell* cell, const gp::GPNode& node) -> int {
        if (!ip || ip->when_expr_.empty()) return -1;
        return add_expr(ip->when_expr_, cell, node);
    };

    std::vector<GpuPowerInternalHost> h_internal_rows;
    const char* debug_power_node_env = std::getenv("XPLACE_POWER_DEBUG_NODE");
    std::unordered_map<std::string, int> internal_denom_group;
    auto get_denom_group = [&](int to_pin, const std::string& related_pg) -> int {
        std::string key = std::to_string(to_pin) + "|" + related_pg;
        auto it = internal_denom_group.find(key);
        if (it != internal_denom_group.end()) return it->second;
        int id = static_cast<int>(internal_denom_group.size());
        internal_denom_group.emplace(std::move(key), id);
        return id;
    };

    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
        const int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id + 1 >= static_cast<int>(gtdb.liberty_cell_type2port_list_end.size())) continue;
        const int port_base = gtdb.liberty_cell_type2port_list_end[libcell_id];
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port) continue;
            const int port_global = port_base + port_offset;
            const int range_idx = port_global * 2 + static_cast<int>(MAX);
            if (range_idx + 1 >= static_cast<int>(gtdb.liberty_port2internal_power_list_end.size())) continue;
            const int ip_start = gtdb.liberty_port2internal_power_list_end[range_idx];
            const int ip_end = gtdb.liberty_port2internal_power_list_end[range_idx + 1];
            if (ip_start == ip_end) continue;

            if (h_is_load_pin[pin_id]) {
                for (int ip_id = ip_start; ip_id < ip_end; ++ip_id) {
                    InternalPower* ip = gtdb.liberty_internal_powers[ip_id];
                    if (!ip) continue;
                    GpuPowerInternalHost row;
                    row.internal_power_id = ip_id;
                    row.node_id = node_id;
                    row.to_pin = pin_id;
                    row.kind = 0;
                    row.energy_unit = ip->energy_unit_;
                    row.duty_mode = 0;
                    int when_expr_id = compile_when_expr(ip, cell, node);
                    if (when_expr_id >= 0) {
                        row.duty_mode = 1;
                        row.duty_expr_id = when_expr_id;
                        for (int op_i = h_expr_start[when_expr_id]; op_i < h_expr_start[when_expr_id] + h_expr_count[when_expr_id]; ++op_i) {
                            const int out_pin = h_expr_ops[op_i].op == 0 ? h_expr_ops[op_i].arg : -1;
                            if (out_pin >= 0 && out_pin < n && h_is_driver_pin[out_pin]) {
                                const int func_expr_id = h_pin_func_expr_id[out_pin];
                                if (expr_contains_pin(func_expr_id, pin_id)) {
                                    row.duty_mode = 2;
                                    row.duty_expr_id = func_expr_id;
                                    row.duty_pin = pin_id;
                                    break;
                                }
                            }
                        }
                    }
                    if (debug_power_node_env && node.getName().find(debug_power_node_env) != std::string::npos) {
                        std::fprintf(stderr,
                                     "[XPLACE_POWER_DEBUG_NODE] node=%s port=%s kind=input ip=%d when='%s' duty_mode=%d duty_expr=%d\n",
                                     node.getName().c_str(),
                                     port->name.c_str(),
                                     ip_id,
                                     ip->when_expr_.c_str(),
                                     row.duty_mode,
                                     row.duty_expr_id);
                    }
                    h_internal_rows.push_back(row);
                }
            }

            if (h_is_driver_pin[pin_id]) {
                const int func_expr_id = h_pin_func_expr_id[pin_id];
                for (int ip_id = ip_start; ip_id < ip_end; ++ip_id) {
                    InternalPower* ip = gtdb.liberty_internal_powers[ip_id];
                    if (!ip) continue;
                    GpuPowerInternalHost row;
                    row.internal_power_id = ip_id;
                    row.node_id = node_id;
                    row.to_pin = pin_id;
                    row.kind = 1;
                    row.energy_unit = ip->energy_unit_;
                    row.duty_mode = 4;
                    LibertyPort* from_port = ip->related_port_;
                    if (from_port && node.portMap.find(from_port->name) != node.portMap.end()) {
                        row.from_pin = node.portMap.at(from_port->name);
                        row.positive_unate = positive_unate_for_power(cell, from_port, port) ? 1 : 0;
                        const int when_expr_id = compile_when_expr(ip, cell, node);
                        if (expr_contains_pin(func_expr_id, row.from_pin)) {
                            row.duty_mode = 2;
                            row.duty_expr_id = func_expr_id;
                            row.duty_pin = row.from_pin;
                        } else if (when_expr_id >= 0) {
                            row.duty_mode = 1;
                            row.duty_expr_id = when_expr_id;
                        } else {
                            row.duty_mode = 3;
                        }
                        const std::string pg = ip->related_pg_pin_ ? ip->related_pg_pin_->name : ip->related_pg_pin_name_;
                        row.denom_group = get_denom_group(pin_id, pg);
                    }
                    if (debug_power_node_env && node.getName().find(debug_power_node_env) != std::string::npos) {
                        std::fprintf(stderr,
                                     "[XPLACE_POWER_DEBUG_NODE] node=%s port=%s kind=output ip=%d related=%s when='%s' duty_mode=%d duty_expr=%d from_pin=%d\n",
                                     node.getName().c_str(),
                                     port->name.c_str(),
                                     ip_id,
                                     ip->related_port_name_.c_str(),
                                     ip->when_expr_.c_str(),
                                     row.duty_mode,
                                     row.duty_expr_id,
                                     row.from_pin);
                    }
                    h_internal_rows.push_back(row);
                }
            }
        }
    }

    const float max_power_unit = (gtdb.cell_libs_[MAX] && gtdb.cell_libs_[MAX]->power_unit_.has_value())
        ? static_cast<float>(gtdb.cell_libs_[MAX]->power_unit_->value()) : 1.0f;
    std::vector<GpuPowerLeakageRowHost> h_leakage_rows;
    std::vector<GpuPowerLeakageGroupHost> h_leakage_groups;
    std::unordered_map<std::string, int> leakage_group_map;
    auto get_leakage_group = [&](int node_id, const std::string& pg, float cell_leakage_w) -> int {
        std::string key = std::to_string(node_id) + "|" + pg;
        auto it = leakage_group_map.find(key);
        if (it != leakage_group_map.end()) return it->second;
        GpuPowerLeakageGroupHost group;
        group.node_id = node_id;
        group.cell_leakage = cell_leakage_w;
        int id = static_cast<int>(h_leakage_groups.size());
        h_leakage_groups.push_back(group);
        leakage_group_map.emplace(std::move(key), id);
        return id;
    };
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
        const int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id * 2 + static_cast<int>(MAX) + 1 >= static_cast<int>(gtdb.liberty_cell_type2leakage_power_list_end.size())) continue;
        const int leak_range_idx = libcell_id * 2 + static_cast<int>(MAX);
        const int leak_start = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx];
        const int leak_end = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx + 1];
        // OpenSTA uses scene_cell(max) for leakage_power groups, but the
        // default/cell_leakage fallback comes from the original cell pointer.
        // In this Xplace setup that corresponds to the MIN/early Liberty view.
        LibertyCell* cell_leakage_cell = (gtdb.cell_libs_[MIN] ? gtdb.cell_libs_[MIN]->get_cell(cell->name) : nullptr);
        if (!cell_leakage_cell) cell_leakage_cell = cell;
        LibertyCell* leak_expr_cell = (gtdb.cell_libs_[MAX] ? gtdb.cell_libs_[MAX]->get_cell(cell->name) : nullptr);
        if (!leak_expr_cell) leak_expr_cell = cell;
        const float cell_leakage_w = cell_leakage_cell->leakage_power_.value_or(0.0f) * max_power_unit;
        if (leak_start == leak_end) {
            get_leakage_group(node_id, "", cell_leakage_w);
            continue;
        }
        for (int leak_id = leak_start; leak_id < leak_end; ++leak_id) {
            LeakagePower* lp = gtdb.liberty_leakage_powers[leak_id];
            if (!lp) continue;
            const std::string pg = lp->related_pg_pin_ ? lp->related_pg_pin_->name : lp->related_pg_pin_name_;
            const int group_id = get_leakage_group(node_id, pg, cell_leakage_w);
            GpuPowerLeakageRowHost row;
            row.node_id = node_id;
            row.group_id = group_id;
            row.leakage_power_id = leak_id;
            row.when_expr_id = lp->when_expr_.empty() ? -1 : add_expr(lp->when_expr_, leak_expr_cell, node);
            row.leakage = lp->value_ * max_power_unit;
            h_leakage_rows.push_back(row);
        }
    }

    auto iopt_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto i64opt_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (internal_row_meta_cpu) {
        std::vector<int64_t> meta;
        meta.reserve(h_internal_rows.size() * 6);
        for (const auto& row : h_internal_rows) {
            meta.push_back(row.node_id);
            meta.push_back(row.to_pin);
            meta.push_back(row.from_pin);
            meta.push_back(row.kind);
            meta.push_back(row.internal_power_id);
            meta.push_back(row.duty_mode);
        }
        if (h_internal_rows.empty()) *internal_row_meta_cpu = torch::empty({0, 6}, i64opt_cpu);
        else *internal_row_meta_cpu = torch::from_blob(meta.data(), {(long)h_internal_rows.size(), 6}, i64opt_cpu).clone();
    }
    if (leakage_row_meta_cpu) {
        std::vector<int64_t> meta;
        meta.reserve(h_leakage_rows.size() * 4);
        for (const auto& row : h_leakage_rows) {
            meta.push_back(row.node_id);
            meta.push_back(row.group_id);
            meta.push_back(row.leakage_power_id);
            meta.push_back(row.when_expr_id);
        }
        if (h_leakage_rows.empty()) *leakage_row_meta_cpu = torch::empty({0, 4}, i64opt_cpu);
        else *leakage_row_meta_cpu = torch::from_blob(meta.data(), {(long)h_leakage_rows.size(), 4}, i64opt_cpu).clone();
    }
    auto bopt_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto fopt_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto to_cuda_int = [&](const std::vector<int>& v) {
        std::vector<int> tmp = v.empty() ? std::vector<int>{0} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, iopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_index = [&](const std::vector<index_type>& v) {
        std::vector<index_type> tmp = v.empty() ? std::vector<index_type>{0} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, iopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_u8 = [&](const std::vector<uint8_t>& v) {
        std::vector<uint8_t> tmp = v.empty() ? std::vector<uint8_t>{0} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, bopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_float = [&](const std::vector<float>& v) {
        auto fopt_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        std::vector<float> tmp = v.empty() ? std::vector<float>{nanf("")} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, fopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_bytes = [&](const auto& v) {
        using VecT = std::decay_t<decltype(v)>;
        using ElemT = typename VecT::value_type;
        std::vector<ElemT> tmp = v.empty() ? std::vector<ElemT>(1) : v;
        auto cpu = torch::from_blob(reinterpret_cast<uint8_t*>(tmp.data()), {(long)(tmp.size() * sizeof(ElemT))}, bopt_cpu).clone();
        return cpu.to(torch::kCUDA);
    };

    auto d_pin_forward_arc_list_end = to_cuda_index(gtdb.pin_forward_arc_list_end);
    auto d_pin_forward_arc_list = to_cuda_index(gtdb.pin_forward_arc_list);
    auto d_timing_arc_to_pin_id = to_cuda_index(gtdb.timing_arc_to_pin_id);
    auto d_arc_types = to_cuda_int(h_power_arc_types);
    std::vector<int> h_power_arc_id2test_id = gtdb.arc_id2test_id;
    auto d_arc_id2test_id = to_cuda_int(h_power_arc_id2test_id);
    auto d_net_driver_pin = to_cuda_int(h_net_driver_pin);
    auto d_is_load_pin = to_cuda_u8(h_is_load_pin);
    auto d_is_driver_pin = to_cuda_u8(h_is_driver_pin);
    auto d_is_cell_pin = to_cuda_u8(h_is_cell_pin);
    auto d_is_seq_output_pin = to_cuda_u8(h_is_seq_output_pin);
    auto d_clock_gate_out_for_input = to_cuda_int(h_clock_gate_out_for_input);
    auto d_clock_gate_clock_for_out = to_cuda_int(h_clock_gate_clock_for_out);
    auto d_clock_gate_enable_for_out = to_cuda_int(h_clock_gate_enable_for_out);
    auto d_clock_pins = to_cuda_int(h_clock_pins);
    auto d_power_clock_slews = to_cuda_float(h_power_clock_slews);
    auto d_expr_ops = to_cuda_bytes(h_expr_ops);
    auto d_expr_start = to_cuda_int(h_expr_start);
    auto d_expr_count = to_cuda_int(h_expr_count);
    auto d_pin_func_expr_id = to_cuda_int(h_pin_func_expr_id);
    auto d_seqs = to_cuda_bytes(h_seqs);
    auto d_pin_seq_list_start = to_cuda_int(h_pin_seq_list_start);
    auto d_pin_seq_list = to_cuda_int(h_pin_seq_list);
    auto d_feedback_seed_pins = to_cuda_int(h_feedback_seed_pins);
    auto d_feedback_seed_seqs = to_cuda_int(h_feedback_seed_seqs);
    auto d_internal_rows = to_cuda_bytes(h_internal_rows);
    auto d_leakage_rows = to_cuda_bytes(h_leakage_rows);
    auto d_leakage_groups = to_cuda_bytes(h_leakage_groups);

    // Power-specific CUDA levelization: use the same propagation edge predicate
    // as power_enqueue_adjacent() (skip constraints/tests and sequential Q/Q_N arcs).
    levelize_power(d_is_seq_output_pin.data_ptr<uint8_t>());
    if (!power_level_list || power_level_list_end_cpu.empty()) {
        throw std::runtime_error("levelize_power failed to build power level list");
    }

    std::vector<uint8_t> h_seed_seen(n, 0);
    std::vector<int> h_seed_inputs;
    h_seed_inputs.reserve(h_primary_inputs.size());
    for (int pin_id : h_primary_inputs) {
        if (pin_id < 0 || pin_id >= n || h_seed_seen[pin_id]) continue;
        h_seed_seen[pin_id] = 1;
        h_seed_inputs.push_back(pin_id);
    }
    h_primary_inputs.swap(h_seed_inputs);
    auto d_primary_inputs = to_cuda_int(h_primary_inputs);

    const bool use_timing_levels_for_power =
        std::getenv("XPLACE_POWER_USE_TIMING_LEVELS") != nullptr;
    const std::vector<int>* activity_level_list_end_cpu = &power_level_list_end_cpu;
    index_type* activity_level_list = power_level_list;
    std::vector<int> h_pin_power_level;
    if (use_timing_levels_for_power) {
        if (!level_list || level_list_end_cpu.empty()) {
            throw std::runtime_error("timing level list is unavailable for power activity");
        }
        activity_level_list_end_cpu = &level_list_end_cpu;
        activity_level_list = level_list;
        h_pin_power_level = pin_level_cpu;
        if (static_cast<int>(h_pin_power_level.size()) != n) h_pin_power_level.assign(n, -1);
    } else {
        h_pin_power_level = power_pin_level_cpu;
        if (static_cast<int>(h_pin_power_level.size()) != n) h_pin_power_level.assign(n, -1);
    }
    auto d_pin_power_level = to_cuda_int(h_pin_power_level);

    int max_activity_passes = 50;
    if (const char* env = std::getenv("XPLACE_POWER_ACTIVITY_MAX_PASSES"))
        max_activity_passes = std::max(1, std::atoi(env));

    auto out_gpu = torch::empty({n, 3}, fopt_cuda);
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
    float power_voltage = 1.0f;
    if (const char* env_voltage = std::getenv("XPLACE_POWER_VOLTAGE")) {
        const float v = std::strtof(env_voltage, nullptr);
        if (std::isfinite(v) && v > 0.0f) power_voltage = v;
    } else {
        // OpenSTA switching power uses the max scene/corner.  Prefer the MAX
        // Liberty operating-condition voltage, then fall back to any available lib.
        auto read_lib_voltage = [](const std::shared_ptr<CellLib>& lib, float& out) -> bool {
            if (!lib) return false;
            auto it = lib->default_values.find("voltage");
            if (it != lib->default_values.end() && it->second.has_value() && *(it->second) > 0.0f) {
                out = *(it->second);
                return true;
            }
            return false;
        };
        if (!read_lib_voltage(gtdb.cell_libs_[MAX], power_voltage)) {
            for (const auto& lib : gtdb.cell_libs_) {
                if (read_lib_voltage(lib, power_voltage)) break;
            }
        }
    }
    if (inst_switching_cpu || pin_switching_cpu) {
        inst_switching_gpu = torch::zeros({num_nodes}, fopt_cuda);
        pin_switching_gpu = torch::zeros({n}, fopt_cuda);
        inst_switching_ptr = inst_switching_gpu.data_ptr<float>();
        pin_switching_ptr = pin_switching_gpu.data_ptr<float>();
    }
    if (inst_internal_cpu || internal_row_power_cpu) {
        inst_internal_gpu = torch::zeros({num_nodes}, fopt_cuda);
        inst_internal_ptr = inst_internal_gpu.data_ptr<float>();
    }
    if (internal_row_power_cpu) {
        internal_row_power_gpu = torch::zeros({static_cast<long>(h_internal_rows.size())}, fopt_cuda);
        internal_row_power_ptr = internal_row_power_gpu.data_ptr<float>();
    }
    if (inst_leakage_cpu) {
        inst_leakage_gpu = torch::zeros({num_nodes}, fopt_cuda);
        inst_leakage_ptr = inst_leakage_gpu.data_ptr<float>();
    }
    if (leakage_row_power_cpu) {
        leakage_row_power_gpu = torch::zeros({static_cast<long>(h_leakage_rows.size())}, fopt_cuda);
        leakage_row_power_ptr = leakage_row_power_gpu.data_ptr<float>();
    }

    const double* dmp_C1_ptr = nullptr;
    const double* dmp_C2_ptr = nullptr;
    bool use_dmp_power_load = true;
    if (const char* env = std::getenv("XPLACE_POWER_USE_DMP_LOAD")) {
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        use_dmp_power_load = !(value.empty() || value == "0" || value == "false" || value == "no");
    }
    if (use_dmp_power_load && h_dmp_db && h_dmp_db->C1 && h_dmp_db->C2) {
        dmp_C1_ptr = h_dmp_db->C1;
        dmp_C2_ptr = h_dmp_db->C2;
    }

    run_power_activity_cuda_launcher(
        n, *activity_level_list_end_cpu,
        activity_level_list,
        d_pin_power_level.data_ptr<int>(),
        d_pin_forward_arc_list_end.data_ptr<index_type>(),
        d_pin_forward_arc_list.data_ptr<index_type>(),
        d_timing_arc_to_pin_id.data_ptr<index_type>(),
        d_arc_types.data_ptr<int>(),
        d_arc_id2test_id.data_ptr<int>(),
        pin2net_map,
        d_net_driver_pin.data_ptr<int>(),
        flat_net2pin_start_map,
        flat_net2pin_map,
        d_is_load_pin.data_ptr<uint8_t>(),
        d_is_driver_pin.data_ptr<uint8_t>(),
        d_is_cell_pin.data_ptr<uint8_t>(),
        d_is_seq_output_pin.data_ptr<uint8_t>(),
        d_clock_gate_out_for_input.data_ptr<int>(),
        d_clock_gate_clock_for_out.data_ptr<int>(),
        d_clock_gate_enable_for_out.data_ptr<int>(),
        d_primary_inputs.data_ptr<int>(),
        static_cast<int>(h_primary_inputs.size()),
        nullptr,
        d_clock_pins.data_ptr<int>(),
        static_cast<int>(h_clock_pins.size()),
        reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>()),
        d_expr_start.data_ptr<int>(),
        d_expr_count.data_ptr<int>(),
        d_pin_func_expr_id.data_ptr<int>(),
        reinterpret_cast<GpuPowerSeqHost*>(d_seqs.data_ptr<uint8_t>()),
        static_cast<int>(h_seqs.size()),
        d_pin_seq_list_start.data_ptr<int>(),
        d_pin_seq_list.data_ptr<int>(),
        d_feedback_seed_pins.data_ptr<int>(),
        static_cast<int>(h_feedback_seed_pins.size()),
        d_feedback_seed_seqs.data_ptr<int>(),
        static_cast<int>(h_feedback_seed_seqs.size()),
        default_density,
        clock_density,
        gtdb.time_unit,
        max_activity_passes,
        out_gpu.data_ptr<float>(),
        num_nodes,
        pin2node_map,
        pinLoad,
        dmp_C1_ptr,
        dmp_C2_ptr,
        pinSlew,
        d_power_clock_slews.data_ptr<float>(),
        ideal_clock,
        reinterpret_cast<GpuPowerInternalHost*>(d_internal_rows.data_ptr<uint8_t>()),
        static_cast<int>(h_internal_rows.size()),
        static_cast<int>(internal_denom_group.size()),
        d_power_allocator,
        cap_unit,
        power_voltage,
        inst_switching_ptr,
        pin_switching_ptr,
        inst_internal_ptr,
        internal_row_power_ptr,
        reinterpret_cast<GpuPowerLeakageRowHost*>(d_leakage_rows.data_ptr<uint8_t>()),
        static_cast<int>(h_leakage_rows.size()),
        reinterpret_cast<GpuPowerLeakageGroupHost*>(d_leakage_groups.data_ptr<uint8_t>()),
        static_cast<int>(h_leakage_groups.size()),
        inst_leakage_ptr,
        leakage_row_power_ptr);

    if (inst_switching_cpu) *inst_switching_cpu = inst_switching_gpu.to(torch::kCPU);
    if (pin_switching_cpu) *pin_switching_cpu = pin_switching_gpu.to(torch::kCPU);
    if (inst_internal_cpu) *inst_internal_cpu = inst_internal_gpu.to(torch::kCPU);
    if (internal_row_power_cpu) *internal_row_power_cpu = internal_row_power_gpu.to(torch::kCPU);
    if (inst_leakage_cpu) *inst_leakage_cpu = inst_leakage_gpu.to(torch::kCPU);
    if (leakage_row_power_cpu) *leakage_row_power_cpu = leakage_row_power_gpu.to(torch::kCPU);
    const bool want_activity_cpu = !inst_switching_cpu && !pin_switching_cpu &&
        !inst_internal_cpu && !internal_row_power_cpu && !internal_row_meta_cpu &&
        !inst_leakage_cpu && !leakage_row_power_cpu && !leakage_row_meta_cpu;
    if (want_activity_cpu) {
        return out_gpu.to(torch::kCPU);
    }
    return torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
}

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
    torch::Tensor pin_switching_cpu;
    torch::Tensor inst_internal_cpu;
    torch::Tensor inst_leakage_cpu;
    compute_power_activity_cuda(&inst_switching_cpu, &pin_switching_cpu, &inst_internal_cpu, nullptr, nullptr, &inst_leakage_cpu);
    torch::Tensor inst_total_cpu = inst_internal_cpu + inst_switching_cpu + inst_leakage_cpu;
    return {inst_internal_cpu, inst_switching_cpu, inst_leakage_cpu, inst_total_cpu};
}

}  // namespace gt
