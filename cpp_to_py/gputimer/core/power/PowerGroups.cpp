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

torch::Tensor GPUTimer::report_power_group_codes() {
    constexpr int sequential_code = 0;
    constexpr int combinational_code = 1;
    constexpr int clock_code = 2;
    constexpr int macro_code = 3;
    constexpr int pad_code = 4;

    auto out = torch::empty({num_nodes}, torch::dtype(torch::kInt64).device(torch::kCPU));
    auto acc = out.accessor<int64_t, 1>();

    const auto& nodes = gtdb.gpdb.getNodes();
    const int n = static_cast<int>(gtdb.pin_names.size());

    auto get_cell_type = [&](int node_id) -> db::CellType* {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return nullptr;
        const int cell_type_id = gtdb.cell_node_type_map[node_id];
        if (cell_type_id < 0 || cell_type_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
        return gtdb.rawdb.celltypes[cell_type_id];
    };
    auto get_cell = [&](int node_id) -> LibertyCell* {
        db::CellType* cell_type = get_cell_type(node_id);
        return cell_type ? cell_type->liberty_cell : nullptr;
    };
    auto is_io_node = [](const gp::GPNode& node) {
        const std::string& node_type = node.getNodeType();
        return node_type == "IOPin" || node_type == "FloatIOPin";
    };
    auto is_pad_cell_type = [](const db::CellType* cell_type) {
        if (!cell_type) return false;
        return cell_type->cls.find("PAD") == 0;
    };
    auto is_macro_cell_type = [](const db::CellType* cell_type) {
        if (!cell_type) return false;
        return cell_type->cls != "CORE";
    };

    std::vector<int> pin_to_net(n, -1);
    std::vector<int> pin_to_node(n, -1);
    std::vector<uint8_t> is_driver_pin(n, 0);
    std::vector<uint8_t> is_load_pin(n, 0);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        const int pin_id = static_cast<int>(pin.getId());
        if (pin_id >= 0 && pin_id < n) {
            pin_to_net[pin_id] = static_cast<int>(pin.getParNetId());
            pin_to_node[pin_id] = static_cast<int>(pin.getParNodeId());
            if (pin_id < static_cast<int>(gtdb.pin_id2port_offset_id.size())) {
                LibertyCell* cell = get_cell(pin_to_node[pin_id]);
                const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (cell && port_offset >= 0 && port_offset < static_cast<int>(cell->ports_.size())) {
                    const LibertyPort* port = cell->ports_[port_offset];
                    if (port) {
                        if (port->direction_ == CellPortDirection::input
                            || port->direction_ == CellPortDirection::inout)
                            is_load_pin[pin_id] = 1;
                        if (port->direction_ == CellPortDirection::output
                            || port->direction_ == CellPortDirection::inout)
                            is_driver_pin[pin_id] = 1;
                    }
                }
            }
        }
    }

    const int num_nets = static_cast<int>(gtdb.gpdb.getNets().size());
    std::vector<uint8_t> is_clock_net(num_nets, 0);
    std::vector<uint8_t> forward_clock_net(num_nets, 0);
    std::deque<int> forward_queue;
    auto mark_clock_net = [&](int net_id) -> bool {
        if (net_id < 0 || net_id >= num_nets || is_clock_net[net_id]) return false;
        is_clock_net[net_id] = 1;
        return true;
    };
    auto mark_forward_net = [&](int net_id) {
        if (net_id < 0 || net_id >= num_nets || forward_clock_net[net_id]) return;
        mark_clock_net(net_id);
        forward_clock_net[net_id] = 1;
        forward_queue.push_back(net_id);
    };
    if (gtdb.net_is_clock.size() == static_cast<size_t>(num_nets)) {
        for (int net_id = 0; net_id < num_nets; ++net_id) {
            if (gtdb.net_is_clock[net_id]) mark_forward_net(net_id);
        }
    }
    if (gtdb.pin_is_clk.size() == static_cast<size_t>(n)) {
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (gtdb.pin_is_clk[pin_id]) mark_forward_net(pin_to_net[pin_id]);
        }
    }

    auto is_core_comb_node = [&](int node_id, LibertyCell* cell) {
        if (!cell || !cell->sequentials_.empty()) return false;
        db::CellType* cell_type = get_cell_type(node_id);
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
        if (node_id < 0 || node_id >= static_cast<int>(nodes.size())) return false;
        for (int out_pin : nodes[node_id].pins()) {
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
            LibertyCell* cell = get_cell(node_id);
            if (!is_clock_transparent_from_pin(node_id, cell, pin_id)) continue;
            for (int out_pin : nodes[node_id].pins()) {
                if (out_pin >= 0 && out_pin < n && is_driver_pin[out_pin])
                    mark_forward_net(pin_to_net[out_pin]);
            }
        }
    }

    std::vector<uint8_t> is_clock_pin(n, 0);
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        if (!is_clock_net[net_id]) continue;
        for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
            if (pin_id >= 0 && pin_id < n) is_clock_pin[pin_id] = 1;
        }
    }
    auto in_clock_network = [&](const gp::GPNode& node, LibertyCell* cell) {
        if (!cell) return false;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            if (pin_id >= static_cast<int>(gtdb.pin_id2port_offset_id.size())) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            const LibertyPort* port = cell->ports_[port_offset];
            if (!port) continue;
            const bool is_output = port->direction_ == CellPortDirection::output
                || port->direction_ == CellPortDirection::inout;
            if (is_output && !is_clock_pin[pin_id]) return false;
        }
        return true;
    };

    for (int node_id = 0; node_id < num_nodes; ++node_id) {
        int group_code = combinational_code;
        if (node_id >= 0 && node_id < static_cast<int>(nodes.size())) {
            const gp::GPNode& node = nodes[node_id];
            db::CellType* cell_type = get_cell_type(node_id);
            LibertyCell* cell = get_cell(node_id);
            if (is_io_node(node) || is_pad_cell_type(cell_type)) {
                group_code = pad_code;
            } else if (is_macro_cell_type(cell_type)) {
                group_code = macro_code;
            } else if (cell && in_clock_network(node, cell)) {
                group_code = clock_code;
            } else if (cell && !cell->sequentials_.empty()) {
                group_code = sequential_code;
            }
        }
        acc[node_id] = group_code;
    }
    return out;
}

}  // namespace gt
