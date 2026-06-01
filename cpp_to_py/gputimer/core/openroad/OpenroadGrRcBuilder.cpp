#include "gputimer/core/openroad/OpenroadRcInternal.h"

namespace gt {

using namespace openroad_rc;

HostRcGraph GPUTimer::build_openroad_gr_rc(const std::string& file) {
    std::ifstream input(file);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open OpenROAD GR RC dump: " + file);
    }

    torch::Tensor flat_net2pin_start_map_at =
        timing_raw_db.flat_net2pin_start_map.cpu().contiguous();
    torch::Tensor flat_net2pin_map_at =
        timing_raw_db.flat_net2pin_map.cpu().contiguous();
    torch::Tensor pin2net_map_at =
        timing_raw_db.pin2net_map.cpu().contiguous();
    const int* flat_net2pin_start_map = flat_net2pin_start_map_at.data_ptr<int>();
    const int* flat_net2pin_map = flat_net2pin_map_at.data_ptr<int>();
    const int* pin2net_map = pin2net_map_at.data_ptr<int>();

    std::unordered_map<std::string, int> net_name_to_index;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        add_gr_name_alias(net_name_to_index, gtdb.net_names[i], i);
    }

    std::unordered_map<std::string, int> global_pin_name_to_id;
    for (int i = 0; i < static_cast<int>(gtdb.pin_names.size()); ++i) {
        add_gr_name_alias(global_pin_name_to_id, gtdb.pin_names[i], i);
    }

    std::vector<LocalSpefNetRc> parsed_nets(num_nets);
    std::vector<std::unordered_map<std::string, int>> edge_key_to_id(num_nets);
    std::vector<uint8_t> parsed_net(num_nets, 0);
    OpenroadGrRcBuildStats stats;
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> gr_corners;
    std::unordered_map<std::string, int> gr_corner_order;

    auto resolve_net = [&](const std::string& name) {
        auto iter = net_name_to_index.find(name);
        if (iter != net_name_to_index.end()) {
            return iter->second;
        }
        std::string normalized = normalized_spef_name(name);
        iter = net_name_to_index.find(normalized);
        return iter == net_name_to_index.end() ? -1 : iter->second;
    };

    auto resolve_pin = [&](const std::string& name) {
        auto iter = global_pin_name_to_id.find(name);
        if (iter != global_pin_name_to_id.end()) {
            return iter->second;
        }
        std::string normalized = normalized_spef_name(name);
        iter = global_pin_name_to_id.find(normalized);
        return iter == global_pin_name_to_id.end() ? -1 : iter->second;
    };

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::vector<std::string> fields = split_tsv(line);
        if (fields.size() < 3) {
            continue;
        }

        if (fields[0] == "WARN") {
            continue;
        }
        if (gr_corner_order.find(fields[1]) == gr_corner_order.end()) {
            const int order = static_cast<int>(gr_corners.size());
            gr_corner_order.emplace(fields[1], order);
            gr_corners.emplace_back(fields[1]);
        }
        rows.emplace_back(std::move(fields));
    }

    if (gr_corners.size() != 1) {
        throw std::runtime_error(
            "OpenROAD GR RC dump must contain exactly one OpenSTA scene for this DMP flow: " + file);
    }
    logger.info("OpenROAD GR RC scene: %s (wire RC is scalar and copied to all DMP attrs)",
                gr_corners[0].c_str());

    for (const std::vector<std::string>& fields : rows) {
        const std::string& tag = fields[0];

        const int net_idx = resolve_net(fields[2]);
        if (net_idx < 0 || net_idx >= num_nets) {
            stats.unknown_nets++;
            continue;
        }
        if (!parsed_net[net_idx]) {
            parsed_net[net_idx] = 1;
            stats.parsed_nets++;
        }
        LocalSpefNetRc& local = parsed_nets[net_idx];

        if (tag == "NET") {
            continue;
        }

        if (tag == "NODE") {
            if (fields.size() < 10) {
                stats.malformed_rows++;
                continue;
            }
            int node_id = -1;
            float cap_f = 0.0f;
            if (!parse_int_field(fields[3], node_id) ||
                !parse_float_field(fields[9], cap_f) || node_id < 0) {
                stats.malformed_rows++;
                continue;
            }
            ensure_local_node(local, node_id);
            if (fields[4] == "route") {
                local.node_names[node_id] =
                    fields[6] + "," + fields[7] + ",M" + fields[8];
            } else {
                local.node_names[node_id] = fields[5];
            }
            set_attr_cap(local.node_cap, node_id, cap_f / gtdb.cap_unit);
            if (fields[4] == "pin") {
                const int pin_id = resolve_pin(fields[5]);
                if (pin_id >= 0 && pin2net_map[pin_id] == net_idx) {
                    local.node2pin[node_id] = pin_id;
                } else if (pin_id >= 0) {
                    stats.pin_net_mismatches++;
                } else if (!fields[5].empty()) {
                    stats.unresolved_pin_nodes++;
                }
            }
            stats.node_rows++;
            continue;
        }

        if (tag == "PIN") {
            if (fields.size() < 5) {
                stats.malformed_rows++;
                continue;
            }
            int node_id = -1;
            if (!parse_int_field(fields[3], node_id) || node_id < 0) {
                stats.malformed_rows++;
                continue;
            }
            ensure_local_node(local, node_id);
            local.node_names[node_id] = fields[4];
            const int pin_id = resolve_pin(fields[4]);
            if (pin_id >= 0 && pin2net_map[pin_id] == net_idx) {
                local.node2pin[node_id] = pin_id;
            } else if (pin_id >= 0) {
                stats.pin_net_mismatches++;
            } else if (!fields[4].empty()) {
                stats.unresolved_pin_nodes++;
            }
            stats.pin_rows++;
            continue;
        }

        if (tag == "CAP") {
            if (fields.size() < 5) {
                stats.malformed_rows++;
                continue;
            }
            int node_id = -1;
            float cap_f = 0.0f;
            if (!parse_int_field(fields[3], node_id) ||
                !parse_float_field(fields[4], cap_f) || node_id < 0) {
                stats.malformed_rows++;
                continue;
            }
            ensure_local_node(local, node_id);
            set_attr_cap(local.node_cap, node_id, cap_f / gtdb.cap_unit);
            stats.cap_rows++;
            continue;
        }

        if (tag == "RES") {
            if (fields.size() < 8) {
                stats.malformed_rows++;
                continue;
            }
            int from = -1;
            int to = -1;
            float res_ohm = 0.0f;
            if (!parse_int_field(fields[5], from) ||
                !parse_int_field(fields[6], to) ||
                !parse_float_field(fields[7], res_ohm) ||
                from < 0 || to < 0) {
                stats.malformed_rows++;
                continue;
            }
            ensure_local_node(local, from);
            ensure_local_node(local, to);
            if (from == to) {
                stats.skipped_self_resistors++;
                continue;
            }
            auto& edge_map = edge_key_to_id[net_idx];
            const std::string edge_key = openroad_gr_edge_key(from, to, fields[3]);
            int edge_id = -1;
            auto edge_iter = edge_map.find(edge_key);
            if (edge_iter == edge_map.end()) {
                edge_id = static_cast<int>(local.edge_from.size());
                edge_map.emplace(edge_key, edge_id);
                local.edge_from.emplace_back(from);
                local.edge_to.emplace_back(to);
                local.edge_res.emplace_back(0.0f);
            } else {
                edge_id = edge_iter->second;
            }
            local.edge_res[edge_id] = res_ohm / gtdb.res_unit;
            stats.resistors++;
            continue;
        }
    }

    HostRcGraph graph;
    graph.includes_pin_caps.assign(num_nets, 0);
    graph.net2node_start.emplace_back(0);
    graph.net2edge_start.emplace_back(0);

    auto append_pin_node = [&](LocalSpefNetRc& local, int pin_id) {
        const int node_id = static_cast<int>(local.node2pin.size());
        local.node2pin.emplace_back(pin_id);
        local.node_names.emplace_back(pin_id >= 0 &&
                                      pin_id < static_cast<int>(gtdb.pin_names.size())
                                          ? gtdb.pin_names[pin_id]
                                          : "");
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            local.node_cap.emplace_back(0.0f);
        }
        return node_id;
    };

    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        LocalSpefNetRc local;
        const int pin_begin = flat_net2pin_start_map[net_idx];
        const int pin_end = flat_net2pin_start_map[net_idx + 1];
        const int driver_pin = pin_begin < pin_end ? flat_net2pin_map[pin_begin] : -1;

        if (parsed_net[net_idx] && driver_pin >= 0) {
            const LocalSpefNetRc& source = parsed_nets[net_idx];
            std::vector<int> old_to_new(source.node2pin.size(), -1);
            int driver_old_node = -1;
            for (int node = 0; node < static_cast<int>(source.node2pin.size()); ++node) {
                if (source.node2pin[node] == driver_pin) {
                    driver_old_node = node;
                    break;
                }
            }

            auto append_old_node = [&](int old_node) {
                const int new_node = static_cast<int>(local.node2pin.size());
                old_to_new[old_node] = new_node;
                local.node2pin.emplace_back(source.node2pin[old_node]);
                local.node_names.emplace_back(source.node_names[old_node]);
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    local.node_cap.emplace_back(source.node_cap[old_node * NUM_ATTR + attr]);
                }
            };

            if (driver_old_node >= 0) {
                append_old_node(driver_old_node);
            } else {
                append_pin_node(local, driver_pin);
                stats.missing_driver_nodes++;
            }

            for (int old_node = 0; old_node < static_cast<int>(source.node2pin.size()); ++old_node) {
                if (old_node != driver_old_node) {
                    append_old_node(old_node);
                }
            }

            for (std::size_t edge = 0; edge < source.edge_from.size(); ++edge) {
                const int old_from = source.edge_from[edge];
                const int old_to = source.edge_to[edge];
                if (old_from < 0 || old_to < 0 ||
                    old_from >= static_cast<int>(old_to_new.size()) ||
                    old_to >= static_cast<int>(old_to_new.size())) {
                    continue;
                }
                const int from = old_to_new[old_from];
                const int to = old_to_new[old_to];
                if (from < 0 || to < 0 || from == to) {
                    stats.skipped_self_resistors++;
                    continue;
                }
                local.edge_from.emplace_back(from);
                local.edge_to.emplace_back(to);
                local.edge_res.emplace_back(source.edge_res[edge]);
            }
        } else if (!parsed_net[net_idx]) {
            stats.missing_nets++;
        }

        if (local.node2pin.empty() && driver_pin >= 0) {
            append_pin_node(local, driver_pin);
        }

        std::unordered_set<int> present_pins;
        for (int pin : local.node2pin) {
            if (pin >= 0) {
                present_pins.insert(pin);
            }
        }
        for (int pin_pos = pin_begin; pin_pos < pin_end; ++pin_pos) {
            const int pin_id = flat_net2pin_map[pin_pos];
            if (present_pins.insert(pin_id).second) {
                const int node = append_pin_node(local, pin_id);
                if (node > 0) {
                    local.edge_from.emplace_back(0);
                    local.edge_to.emplace_back(node);
                    local.edge_res.emplace_back(0.0f);
                    stats.repaired_edges++;
                }
                if (parsed_net[net_idx]) {
                    stats.missing_net_pins++;
                } else {
                    stats.fallback_net_pins++;
                }
            }
        }

        if (!local.node2pin.empty()) {
            std::vector<uint8_t> seen(local.node2pin.size(), 0);
            std::vector<int> stack;
            seen[0] = 1;
            stack.emplace_back(0);
            for (std::size_t cursor = 0; cursor < stack.size(); ++cursor) {
                const int node = stack[cursor];
                for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
                    int next = -1;
                    if (local.edge_from[edge] == node) next = local.edge_to[edge];
                    if (local.edge_to[edge] == node) next = local.edge_from[edge];
                    if (next >= 0 && next < static_cast<int>(seen.size()) && !seen[next]) {
                        seen[next] = 1;
                        stack.emplace_back(next);
                    }
                }
            }
            for (int node = 1; node < static_cast<int>(seen.size()); ++node) {
                if (!seen[node]) {
                    local.edge_from.emplace_back(0);
                    local.edge_to.emplace_back(node);
                    local.edge_res.emplace_back(0.0f);
                    stats.repaired_edges++;
                }
            }
        }

        const int skipped_loop_edges = prune_to_rooted_tree(local);
        if (skipped_loop_edges > 0) {
            stats.skipped_loop_edges += skipped_loop_edges;
        }

        for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
            graph.edge_from.emplace_back(graph.num_nodes + local.edge_from[edge]);
            graph.edge_to.emplace_back(graph.num_nodes + local.edge_to[edge]);
            graph.edge_res.emplace_back(local.edge_res[edge]);
            graph.num_edges++;
        }
        for (int node = 0; node < static_cast<int>(local.node2pin.size()); ++node) {
            graph.node2pin.emplace_back(local.node2pin[node]);
            graph.node_names.emplace_back(local.node_names[node]);
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                graph.node_cap.emplace_back(local.node_cap[node * NUM_ATTR + attr]);
            }
        }
        graph.num_nodes += static_cast<int>(local.node2pin.size());
        graph.net2node_start.emplace_back(graph.num_nodes);
        graph.net2edge_start.emplace_back(graph.num_edges);
    }

    graph.skipped_loop_edges = stats.skipped_loop_edges;
    graph.repaired_edges = stats.repaired_edges;

    logger.info("OpenROAD GR RC graph: file=%s parsed_nets=%d missing_nets=%d unknown_nets=%d nodes=%d edges=%d",
                file.c_str(), stats.parsed_nets,
                stats.missing_nets, stats.unknown_nets, graph.num_nodes,
                graph.num_edges);
    logger.info("OpenROAD GR RC details: node_rows=%d pin_rows=%d cap_rows=%d resistors=%d self_res=%d unresolved_pin_nodes=%d pin_net_mismatches=%d missing_driver_nodes=%d missing_net_pins=%d fallback_net_pins=%d repaired_edges=%d loop_edges=%d malformed_rows=%d",
                stats.node_rows, stats.pin_rows, stats.cap_rows, stats.resistors,
                stats.skipped_self_resistors, stats.unresolved_pin_nodes,
                stats.pin_net_mismatches, stats.missing_driver_nodes, stats.missing_net_pins,
                stats.fallback_net_pins, stats.repaired_edges, stats.skipped_loop_edges,
                stats.malformed_rows);

    if (stats.unknown_nets > 0 || stats.malformed_rows > 0 ||
        stats.skipped_self_resistors > 0 ||
        stats.unresolved_pin_nodes > 0 || stats.pin_net_mismatches > 0 ||
        stats.missing_driver_nodes > 0 || stats.missing_net_pins > 0) {
        std::ostringstream msg;
        msg << "OpenROAD GR RC dump cannot be used for semantic timing alignment: "
            << "missing_nets=" << stats.missing_nets
            << " unknown_nets=" << stats.unknown_nets
            << " "
            << "malformed_rows=" << stats.malformed_rows
            << " self_res=" << stats.skipped_self_resistors
            << " unresolved_pin_nodes=" << stats.unresolved_pin_nodes
            << " pin_net_mismatches=" << stats.pin_net_mismatches
            << " missing_driver_nodes=" << stats.missing_driver_nodes
            << " missing_net_pins=" << stats.missing_net_pins
            << " fallback_net_pins=" << stats.fallback_net_pins
            << " repaired_edges=" << stats.repaired_edges
            << " loop_edges=" << stats.skipped_loop_edges
            << ". Regenerate/check my_dump_gr_rc rather than accepting repaired RC.";
        throw std::runtime_error(msg.str());
    }

    return graph;
}

}  // namespace gt
