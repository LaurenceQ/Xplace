#include "PowerActivityHostUtils.h"

#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "gputimer/core/power/common/PowerHostCommon.h"
#include "io_parser/gp/GPDatabase.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <deque>

namespace gt {

std::string normalizePowerExprString(std::string expr) {
    expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());
    if (expr.size() >= 2 && expr.front() == '"' && expr.back() == '"')
        expr = expr.substr(1, expr.size() - 2);
    return expr;
}

int parsePowerConstNetValue(std::string name) {
    name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char c) { return std::isspace(c); }),
               name.end());
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (name == "0" || name == "1'b0" || name == "1'd0" || name == "1'h0") return 0;
    if (name == "1" || name == "1'b1" || name == "1'd1" || name == "1'h1") return 1;
    const size_t quote = name.find('\'');
    if (quote != std::string::npos && quote + 2 < name.size()) {
        const std::string digits = name.substr(quote + 2);
        if (!digits.empty() && digits.find_first_not_of("0") == std::string::npos) return 0;
        if (!digits.empty() && digits.find_first_not_of("1") == std::string::npos) return 1;
    }
    return -1;
}

LibertyCell* powerCellForNode(GTDatabase& gtdb, int node_id) {
    if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return nullptr;
    const int libcell_id = gtdb.cell_node_type_map[node_id];
    if (libcell_id < 0 || libcell_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
    auto* cell_type = gtdb.rawdb.celltypes[libcell_id];
    return cell_type ? cell_type->liberty_cell : nullptr;
}

LibertyCell* powerCellForLibcell(GTDatabase& gtdb, int libcell_id) {
    if (libcell_id < 0 || libcell_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
    auto* cell_type = gtdb.rawdb.celltypes[libcell_id];
    return cell_type ? cell_type->liberty_cell : nullptr;
}

bool powerIsIoNode(GTDatabase& gtdb, int node_id) {
    if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return false;
    const std::string& node_type = gtdb.gpdb.getNodes()[node_id].getNodeType();
    return node_type == "IOPin" || node_type == "FloatIOPin";
}

std::pair<float, float> powerClockActivityForPin(GTDatabase& gtdb,
                                                 int pin_id,
                                                 double sdc_time_scale,
                                                 float clock_density) {
    float density = clock_density;
    float duty = 0.5f;
    const float period = gtdb.ClockPeriodForPin(pin_id);
    if (std::isfinite(period) && period > 0.0f && sdc_time_scale > 0.0) {
        density = powerDensityForPeriod(2.0, period, sdc_time_scale);
        const float rise = gtdb.ClockRiseEdgeForPin(pin_id);
        const float fall = gtdb.ClockFallEdgeForPin(pin_id);
        if (std::isfinite(rise) && std::isfinite(fall)) {
            const float candidate_duty = (fall - rise) / period;
            if (std::isfinite(candidate_duty) && candidate_duty >= 0.0f && candidate_duty <= 1.0f)
                duty = candidate_duty;
        }
    }
    return {density, duty};
}

void buildPowerPinNodeNetMaps(GTDatabase& gtdb,
                              int n,
                              std::vector<int>& pin_to_node,
                              std::vector<int>& pin_to_net) {
    pin_to_node.assign(n, -1);
    pin_to_net.assign(n, -1);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        const int pin_id = static_cast<int>(pin.getId());
        if (pin_id >= 0 && pin_id < n) {
            pin_to_node[pin_id] = static_cast<int>(pin.getParNodeId());
            pin_to_net[pin_id] = static_cast<int>(pin.getParNetId());
        }
    }
}

void classifyPowerPins(GTDatabase& gtdb,
                       int n,
                       const std::vector<int>& pin_to_node,
                       std::vector<uint8_t>& is_load_pin,
                       std::vector<uint8_t>& is_driver_pin,
                       std::vector<uint8_t>* is_cell_pin) {
    is_load_pin.assign(n, 0);
    is_driver_pin.assign(n, 0);
    if (is_cell_pin) is_cell_pin->assign(n, 0);
    const int node_count = static_cast<int>(gtdb.gpdb.getNodes().size());
    std::vector<uint8_t> is_io_node(node_count, 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        if (node_id < 0 || node_id >= node_count) continue;
        const std::string& node_type = node.getNodeType();
        is_io_node[node_id] = (node_type == "IOPin" || node_type == "FloatIOPin") ? 1 : 0;
    }
    std::vector<uint8_t> is_primary_input_pin(n, 0);
    std::vector<uint8_t> is_primary_output_pin(n, 0);
    for (int pin_id : gtdb.primary_inputs) {
        if (pin_id >= 0 && pin_id < n) is_primary_input_pin[pin_id] = 1;
    }
    for (int pin_id : gtdb.primary_outputs) {
        if (pin_id >= 0 && pin_id < n) is_primary_output_pin[pin_id] = 1;
    }
    const int num_libcell_slots =
        std::max(0, static_cast<int>(gtdb.liberty_cell_type2port_list_end.size()) - 1);
    const int num_port_slots = gtdb.liberty_cell_type2port_list_end.empty()
        ? 0
        : gtdb.liberty_cell_type2port_list_end.back();
    std::vector<uint8_t> port_direction(std::max(0, num_port_slots), 0);
    for (int libcell_id = 0; libcell_id < num_libcell_slots; ++libcell_id) {
        LibertyCell* cell = powerCellForLibcell(gtdb, libcell_id);
        if (!cell) continue;
        const int port_start = gtdb.liberty_cell_type2port_list_end[libcell_id];
        const int port_end = gtdb.liberty_cell_type2port_list_end[libcell_id + 1];
        const int port_count = std::min(static_cast<int>(cell->ports_.size()), port_end - port_start);
        for (int port_offset = 0; port_offset < port_count; ++port_offset) {
            LibertyPort* port = cell->ports_[port_offset];
            if (!port) continue;
            const int port_global = port_start + port_offset;
            if (port_global < 0 || port_global >= static_cast<int>(port_direction.size())) continue;
            if (port->direction_ == CellPortDirection::input) port_direction[port_global] = 1;
            else if (port->direction_ == CellPortDirection::output) port_direction[port_global] = 2;
        }
    }
    for (const auto& pin : gtdb.gpdb.getPins()) {
        const int pin_id = static_cast<int>(pin.getId());
        if (pin_id < 0 || pin_id >= n) continue;
        const int node_id = pin_to_node[pin_id];
        const bool io_node = node_id >= 0 && node_id < node_count && is_io_node[node_id] != 0;
        if (is_cell_pin && !io_node) (*is_cell_pin)[pin_id] = 1;
        if (io_node && is_primary_input_pin[pin_id]) {
            is_driver_pin[pin_id] = 1;
            continue;
        }
        if (io_node && is_primary_output_pin[pin_id]) {
            is_load_pin[pin_id] = 1;
            continue;
        }
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
        const int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id + 1 >= static_cast<int>(gtdb.liberty_cell_type2port_list_end.size())) continue;
        const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        const int port_global = gtdb.liberty_cell_type2port_list_end[libcell_id] + port_offset;
        if (port_offset < 0 || port_global < 0 || port_global >= static_cast<int>(port_direction.size())) continue;
        if (port_direction[port_global] == 1) is_load_pin[pin_id] = 1;
        else if (port_direction[port_global] == 2) is_driver_pin[pin_id] = 1;
    }
}

void markPowerSeqOutputPins(GTDatabase& gtdb,
                            int n,
                            const std::vector<int>& pin_to_node,
                            const std::vector<uint8_t>& is_driver_pin,
                            std::vector<uint8_t>& is_seq_output_pin) {
    is_seq_output_pin.assign(n, 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = powerCellForNode(gtdb, node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            const std::string seq_out = normalizePowerExprString(seq->output_name_);
            const std::string seq_out_inv = normalizePowerExprString(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n || !is_driver_pin[pin_id]) continue;
                const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || !port->has_function_) continue;
                const std::string func = normalizePowerExprString(port->function_expr_);
                if ((!seq_out.empty() && func == seq_out) ||
                    (!seq_out_inv.empty() && func == seq_out_inv)) {
                    is_seq_output_pin[pin_id] = 1;
                }
            }
        }
    }
}

void buildPowerNetDriverPins(GTDatabase& gtdb,
                             int n,
                             const std::vector<uint8_t>& is_driver_pin,
                             std::vector<int>& net_driver_pin) {
    net_driver_pin.assign(gtdb.gpdb.getNets().size(), -1);
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
}

void buildPowerClockGateMaps(GTDatabase& gtdb,
                             int n,
                             const std::vector<int>& pin_to_node,
                             std::vector<int>& clock_gate_out_for_input,
                             std::vector<int>& clock_gate_clock_for_out,
                             std::vector<int>& clock_gate_enable_for_out,
                             std::vector<uint8_t>& is_clock_gate_clock_pin) {
    clock_gate_out_for_input.assign(n, -1);
    clock_gate_clock_for_out.assign(n, -1);
    clock_gate_enable_for_out.assign(n, -1);
    is_clock_gate_clock_pin.assign(n, 0);
    const int num_libcell_slots =
        std::max(0, static_cast<int>(gtdb.liberty_cell_type2port_list_end.size()) - 1);
    const int num_port_slots = gtdb.liberty_cell_type2port_list_end.empty()
        ? 0
        : gtdb.liberty_cell_type2port_list_end.back();
    std::vector<uint8_t> is_clock_gate_clock_port(std::max(0, num_port_slots), 0);
    std::vector<uint8_t> is_clock_gate_enable_port(std::max(0, num_port_slots), 0);
    std::vector<uint8_t> is_clock_gate_out_port(std::max(0, num_port_slots), 0);
    for (int libcell_id = 0; libcell_id < num_libcell_slots; ++libcell_id) {
        LibertyCell* cell = powerCellForLibcell(gtdb, libcell_id);
        if (!cell) continue;
        const int port_start = gtdb.liberty_cell_type2port_list_end[libcell_id];
        const int port_end = gtdb.liberty_cell_type2port_list_end[libcell_id + 1];
        const int port_count = std::min(static_cast<int>(cell->ports_.size()), port_end - port_start);
        for (int port_offset = 0; port_offset < port_count; ++port_offset) {
            LibertyPort* port = cell->ports_[port_offset];
            if (!port) continue;
            const int port_global = port_start + port_offset;
            if (port_global < 0 || port_global >= num_port_slots) continue;
            if (port->is_clock_gate_clock_) is_clock_gate_clock_port[port_global] = 1;
            if (port->is_clock_gate_enable_) is_clock_gate_enable_port[port_global] = 1;
            if (port->is_clock_gate_out_) is_clock_gate_out_port[port_global] = 1;
        }
    }
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
        const int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id + 1 >= static_cast<int>(gtdb.liberty_cell_type2port_list_end.size())) continue;
        const int port_start = gtdb.liberty_cell_type2port_list_end[libcell_id];
        int clk_pin = -1;
        int enable_pin = -1;
        int out_pin = -1;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            const int port_global = port_start + port_offset;
            if (port_offset < 0 || port_global < 0 || port_global >= num_port_slots) continue;
            if (is_clock_gate_clock_port[port_global]) clk_pin = pin_id;
            if (is_clock_gate_enable_port[port_global]) enable_pin = pin_id;
            if (is_clock_gate_out_port[port_global]) out_pin = pin_id;
        }
        if (out_pin >= 0 && clk_pin >= 0 && enable_pin >= 0) {
            clock_gate_clock_for_out[out_pin] = clk_pin;
            clock_gate_enable_for_out[out_pin] = enable_pin;
            clock_gate_out_for_input[clk_pin] = out_pin;
            clock_gate_out_for_input[enable_pin] = out_pin;
            is_clock_gate_clock_pin[clk_pin] = 1;
        }
    }
}


PowerCpuActivityLevels buildPowerCpuActivityLevels(GTDatabase& gtdb, int n) {
    std::vector<int> pin_level(n, 0);
    int max_level = 0;
    if (gtdb.pin_num_fanin.size() == static_cast<size_t>(n)
        && gtdb.pin_fanout_list_end.size() == static_cast<size_t>(n + 1)) {
        std::vector<int> indeg = gtdb.pin_num_fanin;
        std::deque<int> frontier;
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (indeg[pin_id] == 0) frontier.push_back(pin_id);
        }
        std::vector<uint8_t> seen(n, 0);
        while (!frontier.empty()) {
            const int pin_id = frontier.front();
            frontier.pop_front();
            if (pin_id < 0 || pin_id >= n || seen[pin_id]) continue;
            seen[pin_id] = 1;
            max_level = std::max(max_level, pin_level[pin_id]);
            const int start = gtdb.pin_fanout_list_end[pin_id];
            const int end = gtdb.pin_fanout_list_end[pin_id + 1];
            for (int idx = start; idx < end; ++idx) {
                const int fanout = gtdb.pin_fanout_list[idx];
                if (fanout < 0 || fanout >= n) continue;
                pin_level[fanout] = std::max(pin_level[fanout], pin_level[pin_id] + 1);
                if (--indeg[fanout] == 0) frontier.push_back(fanout);
            }
        }
    }
    std::vector<std::vector<int>> by_level(std::max(1, max_level + 1));
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        const int level = std::clamp(pin_level[pin_id], 0, max_level);
        by_level[level].push_back(pin_id);
    }
    std::vector<int> level_list;
    std::vector<int> level_list_end;
    level_list.reserve(n);
    level_list_end.reserve(by_level.size() + 1);
    level_list_end.push_back(0);
    for (const auto& pins : by_level) {
        level_list.insert(level_list.end(), pins.begin(), pins.end());
        level_list_end.push_back(static_cast<int>(level_list.size()));
    }
    return PowerCpuActivityLevels(std::move(pin_level),
                                  std::move(level_list),
                                  std::move(level_list_end),
                                  max_level);
}

void buildPowerNodePortPinMap(GTDatabase& gtdb,
                              std::vector<int>& node_port_pin_start,
                              std::vector<int>& node_port_pin_list) {
    const int node_count = static_cast<int>(gtdb.gpdb.getNodes().size());
    node_port_pin_start.assign(node_count + 1, 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = powerCellForNode(gtdb, node_id);
        const int port_count = cell ? static_cast<int>(cell->ports_.size()) : 0;
        if (node_id >= 0 && node_id < node_count)
            node_port_pin_start[node_id + 1] = port_count;
    }
    for (int node_id = 0; node_id < node_count; ++node_id)
        node_port_pin_start[node_id + 1] += node_port_pin_start[node_id];
    node_port_pin_list.assign(node_port_pin_start.back(), -1);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        const int pin_id = static_cast<int>(pin.getId());
        if (pin_id < 0 || pin_id >= static_cast<int>(gtdb.pin_id2port_offset_id.size())) continue;
        const int node_id = static_cast<int>(pin.getParNodeId());
        if (node_id < 0 || node_id >= node_count) continue;
        const int port_id = gtdb.pin_id2port_offset_id[pin_id];
        const int start = node_port_pin_start[node_id];
        const int end = node_port_pin_start[node_id + 1];
        if (port_id >= 0 && start + port_id < end) {
            node_port_pin_list[start + port_id] = pin_id;
        }
    }
}

std::vector<int> buildPowerClockPins(GTDatabase& gtdb,
                                     int n,
                                     const std::vector<int>& pin_to_node,
                                     const std::vector<int>& pin_to_net,
                                     const std::vector<uint8_t>& is_load_pin,
                                     const std::vector<uint8_t>& is_driver_pin,
                                     const std::vector<uint8_t>& is_clock_gate_clock_pin) {
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
    for (int pin_id = 0; pin_id < n; pin_id++) {
        if (is_clock_gate_clock_pin[pin_id]) mark_net(pin_to_net[pin_id]);
    }

    std::vector<uint8_t> forward_clock_net(num_nets, 0);
    std::deque<int> forward_queue;
    auto mark_forward_net = [&](int net_id) {
        if (net_id < 0 || net_id >= num_nets || forward_clock_net[net_id]) return;
        forward_clock_net[net_id] = 1;
        forward_queue.push_back(net_id);
    };
    if (gtdb.net_is_clock.size() == static_cast<size_t>(num_nets)) {
        for (int net_id = 0; net_id < num_nets; net_id++) {
            if (gtdb.net_is_clock[net_id]) mark_forward_net(net_id);
        }
    }
    std::vector<int> extra_clock_pins;
    std::vector<uint8_t> extra_clock_pin_seen(n, 0);
    auto add_extra_clock_pin = [&](int pin_id) {
        if (pin_id < 0 || pin_id >= n || extra_clock_pin_seen[pin_id]) return;
        extra_clock_pin_seen[pin_id] = 1;
        extra_clock_pins.push_back(pin_id);
    };
    auto is_core_comb_node = [&](int node_id, LibertyCell* cell) {
        if (!cell || !cell->sequentials_.empty()) return false;
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return false;
        const int cell_type_id = gtdb.cell_node_type_map[node_id];
        if (cell_type_id < 0 || cell_type_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return false;
        db::CellType* cell_type = gtdb.rawdb.celltypes[cell_type_id];
        return cell_type && cell_type->cls == "CORE";
    };
    auto is_clock_transparent_from_pin = [&](int node_id, LibertyCell* cell, int in_pin_id) {
        if (!is_core_comb_node(node_id, cell)) return false;
        if (in_pin_id < 0 || in_pin_id >= n ||
            in_pin_id >= static_cast<int>(gtdb.pin_id2port_offset_id.size()))
            return false;
        const int in_port = gtdb.pin_id2port_offset_id[in_pin_id];
        if (in_port < 0 || in_port >= static_cast<int>(cell->ports_.size())) return false;

        bool output_seen = false;
        for (int out_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
            if (out_pin < 0 || out_pin >= n || !is_driver_pin[out_pin]) continue;
            if (out_pin >= static_cast<int>(gtdb.pin_id2port_offset_id.size())) return false;
            const int out_port = gtdb.pin_id2port_offset_id[out_pin];
            if (out_port < 0 || out_port >= static_cast<int>(cell->ports_.size())) return false;
            LibertyPort* port = cell->ports_[out_port];
            if (!port || !port->has_function_) return false;
            PowerExpr expr;
            if (!expr.compile(port->function_expr_, cell)) return false;
            const auto& ops = expr.ops();
            const bool direct =
                ops.size() == 1 && ops[0].opcode == PowerExprOpcode::port && ops[0].port_id == in_port;
            const bool inverted =
                ops.size() == 2 && ops[0].opcode == PowerExprOpcode::port && ops[0].port_id == in_port &&
                ops[1].opcode == PowerExprOpcode::logical_not;
            if (!direct && !inverted) return false;
            output_seen = true;
        }
        return output_seen;
    };
    for (size_t queue_pos = 0; queue_pos < forward_queue.size(); ++queue_pos) {
        const int net_id = forward_queue[queue_pos];
        if (net_id < 0 || net_id >= num_nets) continue;
        for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
            if (pin_id < 0 || pin_id >= n || !is_load_pin[pin_id]) continue;
            const int node_id = pin_to_node[pin_id];
            LibertyCell* cell = powerCellForNode(gtdb, node_id);
            if (!is_core_comb_node(node_id, cell)) continue;
            add_extra_clock_pin(pin_id);
            if (!is_clock_transparent_from_pin(node_id, cell, pin_id)) continue;
            for (int out_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
                if (out_pin >= 0 && out_pin < n && is_driver_pin[out_pin])
                    mark_forward_net(pin_to_net[out_pin]);
            }
        }
    }
    std::vector<int> clock_pins;
    for (int net_id = 0; net_id < num_nets; net_id++) {
        if (!is_clock_net[net_id]) continue;
        for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
            if (pin_id >= 0 && pin_id < n
                && (is_load_pin[pin_id] || powerIsIoNode(gtdb, pin_to_node[pin_id])))
                clock_pins.push_back(pin_id);
        }
    }
    clock_pins.insert(clock_pins.end(), extra_clock_pins.begin(), extra_clock_pins.end());
    std::sort(clock_pins.begin(), clock_pins.end());
    clock_pins.erase(std::unique(clock_pins.begin(), clock_pins.end()), clock_pins.end());
    return clock_pins;
}

}  // namespace gt
