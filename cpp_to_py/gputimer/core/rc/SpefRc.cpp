#include "gputimer/core/GPUTimer.h"
#include "common/db/Database.h"
#include "gputimer/core/rc/RcModels.h"
#include "gputimer/db/GTDatabase.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace gt {

void GPUTimer::read_spef(const std::string& file) {
    logger.info("reading spef: %s", file.c_str());
    if (not std::filesystem::exists(file)) {
        logger.error("can't find %s", file.c_str());
        std::exit(EXIT_FAILURE);
    }

    // Invoke the read function and check the return value
    if (not spef.read(file)) {
        if (spef.error) {
            logger.error("failed to read SPEF line=%zu byte=%zu text=%s",
                         spef.error->line_number,
                         spef.error->byte_in_line,
                         spef.error->line.c_str());
        } else {
            logger.error("failed to read SPEF");
        }
        std::exit(EXIT_FAILURE);
    }


    if (spef.time_unit == "1 PS") gtdb.spef_time_unit = 1e-12;
    if (spef.time_unit == "1 NS") gtdb.spef_time_unit = 1e-9;
    if (spef.time_unit == "1 US") gtdb.spef_time_unit = 1e-6;
    if (spef.time_unit == "1 MS") gtdb.spef_time_unit = 1e-3;
    if (spef.time_unit == "1 S") gtdb.spef_time_unit = 1.0; ;

    if (spef.capacitance_unit == "1 FF") gtdb.spef_cap_unit = 1e-15;
    if (spef.capacitance_unit == "1 PF") gtdb.spef_cap_unit = 1e-12;
    if (spef.capacitance_unit == "1 NF") gtdb.spef_cap_unit = 1e-9;
    if (spef.capacitance_unit == "1 UF") gtdb.spef_cap_unit = 1e-6;
    if (spef.capacitance_unit == "1 F") gtdb.spef_cap_unit = 1.0;

    if (spef.resistance_unit == "1 OHM") gtdb.spef_res_unit = 1.0;
    if (spef.resistance_unit == "1 KOHM") gtdb.spef_res_unit = 1e3;
    if (spef.resistance_unit == "1 MOHM") gtdb.spef_res_unit = 1e6;

    logger.info("spef time_unit: %.5E s", *gtdb.spef_time_unit);
    logger.info("spef capacitance_unit: %.5E F", *gtdb.spef_cap_unit);
    logger.info("spef resistance_unit: %.5E Ohm", *gtdb.spef_res_unit);

    spef.expand_name();
}
using std::string;
using std::ofstream;
using std::endl;
using std::stringstream;

namespace {

struct SpefRcBuildCounts {
    int parsed_nets = 0;
    int missing_nets = 0;
    int missing_pin_nodes = 0;
    int unresolved_cap_nodes = 0;
    int unresolved_res_nodes = 0;
    int ground_caps = 0;
    int coupling_caps = 0;
    int folded_coupling_terms = 0;
    int resistors = 0;
    int skipped_self_resistors = 0;
    int skipped_loop_edges = 0;
    int repaired_edges = 0;
    int fallback_nets = 0;
};

struct LocalRcNetGraph {
    std::vector<int> edge_from;
    std::vector<int> edge_to;
    std::vector<float> edge_res;
    std::vector<float> node_cap;
    std::vector<int> node2pin;
    std::vector<std::string> node_names;
    std::unordered_map<std::string, int> node_name2id;
    std::unordered_map<std::string, int> pin_name2id;
};

static std::string normalized_spef_name(std::string name)
{
    validate_token(name);
    return name;
}

static bool spef_digits_only(const std::string& value)
{
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
    });
}

static void add_name_alias(std::unordered_map<std::string, int>& map,
                           const std::string& name,
                           int value)
{
    if (name.empty()) {
        return;
    }
    map.emplace(name, value);
    map.emplace(normalized_spef_name(name), value);
}

static void add_attr_cap(std::vector<float>& node_cap, int node, float cap)
{
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        node_cap[node * NUM_ATTR + attr] += cap;
    }
}

static std::string spef_upper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

static bool spef_includes_pin_caps_from_design_flow(const std::string& design_flow)
{
    std::string flow = spef_upper(design_flow);
    size_t pin_cap_pos = flow.find("PIN_CAP");
    if (pin_cap_pos == std::string::npos) {
        return false;
    }
    std::string pin_cap_clause = flow.substr(pin_cap_pos);
    return pin_cap_clause.find("NONE") == std::string::npos;
}

static int count_tree_edges_from_root(const LocalRcNetGraph& local)
{
    if (local.node2pin.empty()) {
        return 0;
    }
    std::vector<uint8_t> seen(local.node2pin.size(), 0);
    std::vector<int> stack;
    int tree_edges = 0;
    seen[0] = 1;
    stack.emplace_back(0);
    for (size_t cursor = 0; cursor < stack.size(); ++cursor) {
        int node = stack[cursor];
        for (size_t edge = 0; edge < local.edge_from.size(); ++edge) {
            int from = local.edge_from[edge];
            int to = local.edge_to[edge];
            int next = -1;
            if (from == node) next = to;
            if (to == node) next = from;
            if (next >= 0 && next < static_cast<int>(seen.size()) && !seen[next]) {
                seen[next] = 1;
                stack.emplace_back(next);
                tree_edges++;
            }
        }
    }
    return tree_edges;
}


}  // namespace

HostRcGraph GPUTimer::build_spef_rc() {
    if (!gtdb.spef_res_unit.has_value() || !gtdb.spef_cap_unit.has_value()) {
        throw std::runtime_error("build_spef_rc requires read_spef() before building the SPEF RC graph.");
    }

    torch::Tensor flat_net2pin_start_map_at = timing_raw_db.flat_net2pin_start_map.clone().cpu().contiguous();
    torch::Tensor flat_net2pin_map_at = timing_raw_db.flat_net2pin_map.clone().cpu().contiguous();
    const int* flat_net2pin_start_map = flat_net2pin_start_map_at.data_ptr<int>();
    const int* flat_net2pin_map = flat_net2pin_map_at.data_ptr<int>();

    float spef_res_ratio = *gtdb.spef_res_unit / gtdb.res_unit;
    float spef_cap_ratio = *gtdb.spef_cap_unit / gtdb.cap_unit;
    float spef_time_ratio = gtdb.spef_time_unit.has_value() ? *gtdb.spef_time_unit / gtdb.time_unit : 1.0f;
    logger.info("spef lib ratios: res %.5E cap %.5E time %.5E", spef_res_ratio, spef_cap_ratio, spef_time_ratio);

    std::unordered_map<std::string, int> net_name_to_index;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        add_name_alias(net_name_to_index, gtdb.net_names[i], i);
    }

    std::unordered_map<std::string, int> global_pin_name_to_id;
    for (int i = 0; i < static_cast<int>(gtdb.pin_names.size()); ++i) {
        add_name_alias(global_pin_name_to_id, gtdb.pin_names[i], i);
    }

    std::vector<LocalRcNetGraph> local_nets(num_nets);
    std::vector<uint8_t> parsed_net(num_nets, 0);
    SpefRcBuildCounts counts;
    const std::string delimiter = spef.delimiter.empty() ? ":" : spef.delimiter;
    const bool spef_includes_pin_caps = spef_includes_pin_caps_from_design_flow(spef.design_flow);

    auto init_net = [&](int net_idx) {
        auto& local = local_nets[net_idx];
        if (!local.node2pin.empty()) {
            return;
        }
        int start = flat_net2pin_start_map[net_idx];
        int end = flat_net2pin_start_map[net_idx + 1];
        local.node2pin.reserve(end - start);
        local.node_cap.reserve((end - start) * NUM_ATTR);
        for (int j = start; j < end; ++j) {
            int pin_id = flat_net2pin_map[j];
            int local_id = static_cast<int>(local.node2pin.size());
            local.node2pin.emplace_back(pin_id);
            local.node_names.emplace_back(gtdb.pin_names[pin_id]);
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                local.node_cap.emplace_back(0.0f);
            }
            add_name_alias(local.node_name2id, gtdb.pin_names[pin_id], local_id);
            add_name_alias(local.pin_name2id, gtdb.pin_names[pin_id], local_id);
        }
    };

    auto add_internal_node = [&](int net_idx, const std::string& node_name) {
        auto& local = local_nets[net_idx];
        int node_id = static_cast<int>(local.node2pin.size());
        local.node2pin.emplace_back(-1);
        local.node_names.emplace_back(node_name);
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            local.node_cap.emplace_back(0.0f);
        }
        add_name_alias(local.node_name2id, node_name, node_id);
        return node_id;
    };

    auto resolve_node = [&](const std::string& raw_name, int net_idx, bool create) {
        init_net(net_idx);
        auto& local = local_nets[net_idx];
        std::string node_name = normalized_spef_name(raw_name);
        if (auto it = local.node_name2id.find(node_name); it != local.node_name2id.end()) {
            return it->second;
        }

        size_t delim_pos = delimiter.empty() ? std::string::npos : node_name.rfind(delimiter);
        if (delim_pos != std::string::npos) {
            std::string name1 = normalized_spef_name(node_name.substr(0, delim_pos));
            std::string name2 = normalized_spef_name(node_name.substr(delim_pos + delimiter.size()));
            std::string pin_candidate = name1 + ":" + name2;
            if (auto it = local.pin_name2id.find(pin_candidate); it != local.pin_name2id.end()) {
                add_name_alias(local.node_name2id, node_name, it->second);
                return it->second;
            }
            if (auto it = local.pin_name2id.find(node_name); it != local.pin_name2id.end()) {
                return it->second;
            }
            if (auto pin_it = global_pin_name_to_id.find(pin_candidate); pin_it != global_pin_name_to_id.end()) {
                counts.missing_pin_nodes++;
                return -1;
            }

            auto net_it = net_name_to_index.find(name1);
            if (net_it != net_name_to_index.end()) {
                if (net_it->second == net_idx && spef_digits_only(name2)) {
                    return create ? add_internal_node(net_idx, node_name) : -1;
                }
                return -1;
            }

            counts.missing_pin_nodes++;
            return -1;
        }

        if (auto it = local.pin_name2id.find(node_name); it != local.pin_name2id.end()) {
            return it->second;
        }
        if (global_pin_name_to_id.find(node_name) != global_pin_name_to_id.end()) {
            counts.missing_pin_nodes++;
        }
        return -1;
    };

    for (const auto& n : spef.nets) {
        string net_name = normalized_spef_name(n.name);
        auto net_itr = net_name_to_index.find(net_name);
        if (net_itr == net_name_to_index.end()) continue;

        int net_idx = net_itr->second;
        if (!parsed_net[net_idx]) {
            parsed_net[net_idx] = 1;
            counts.parsed_nets++;
        }
        init_net(net_idx);
        auto& local = local_nets[net_idx];

        for (const auto& [node1, node2, cap] : n.caps) {
            const float cap_internal = cap * spef_cap_ratio;
            if (node2.empty()) {
                counts.ground_caps++;
                int node = resolve_node(node1, net_idx, true);
                if (node >= 0) {
                    add_attr_cap(local.node_cap, node, cap_internal);
                } else {
                    counts.unresolved_cap_nodes++;
                }
            } else {
                counts.coupling_caps++;
                int node_a = resolve_node(node1, net_idx, true);
                int node_b = resolve_node(node2, net_idx, true);
                if (node_a >= 0) {
                    add_attr_cap(local.node_cap, node_a, cap_internal);
                    counts.folded_coupling_terms++;
                }
                if (node_b >= 0) {
                    add_attr_cap(local.node_cap, node_b, cap_internal);
                    counts.folded_coupling_terms++;
                }
                if (node_a < 0 && node_b < 0) {
                    counts.unresolved_cap_nodes++;
                }
            }
        }

        for (const auto& [node1, node2, res] : n.ress) {
            counts.resistors++;
            int from = resolve_node(node1, net_idx, true);
            int to = resolve_node(node2, net_idx, true);
            if (from < 0 || to < 0) {
                counts.unresolved_res_nodes++;
                continue;
            }
            if (from == to) {
                counts.skipped_self_resistors++;
                continue;
            }
            local.edge_from.emplace_back(from);
            local.edge_to.emplace_back(to);
            local.edge_res.emplace_back(res * spef_res_ratio);
        }
    }

    HostRcGraph graph;
    graph.num_nets = num_nets;
    graph.includes_pin_caps.assign(num_nets, spef_includes_pin_caps ? 1 : 0);
    graph.net2node_start.emplace_back(0);
    graph.net2edge_start.emplace_back(0);

    for (int i = 0; i < num_nets; ++i) {
        if (!parsed_net[i]) {
            counts.missing_nets++;
            counts.fallback_nets++;
            init_net(i);
            auto& local = local_nets[i];
            for (int node = 1; node < static_cast<int>(local.node2pin.size()); ++node) {
                local.edge_from.emplace_back(0);
                local.edge_to.emplace_back(node);
                local.edge_res.emplace_back(0.0f);
            }
        }

        auto& local = local_nets[i];
        if (!local.node2pin.empty()) {
            std::vector<uint8_t> seen(local.node2pin.size(), 0);
            std::vector<int> stack;
            seen[0] = 1;
            stack.emplace_back(0);
            for (size_t cursor = 0; cursor < stack.size(); ++cursor) {
                int node = stack[cursor];
                for (size_t edge = 0; edge < local.edge_from.size(); ++edge) {
                    int from = local.edge_from[edge];
                    int to = local.edge_to[edge];
                    int next = -1;
                    if (from == node) next = to;
                    if (to == node) next = from;
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
                    counts.repaired_edges++;
                }
            }
        }
        int tree_edges = count_tree_edges_from_root(local);
        int skipped_loop_edges = static_cast<int>(local.edge_from.size()) - tree_edges;
        if (skipped_loop_edges > 0) {
            counts.skipped_loop_edges += skipped_loop_edges;
        }

        for (size_t edge = 0; edge < local.edge_from.size(); ++edge) {
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
    graph.skipped_loop_edges = counts.skipped_loop_edges;
    graph.repaired_edges = counts.repaired_edges;

    logger.info("SPEF RC graph: parsed_nets=%d missing_nets=%d nodes=%d edges=%d includes_pin_caps=%d",
                counts.parsed_nets, counts.missing_nets, graph.num_nodes, graph.num_edges,
                spef_includes_pin_caps ? 1 : 0);
    logger.info("SPEF RC details: ground_caps=%d coupling_caps=%d folded_coupling_terms=%d resistors=%d self_res=%d loop_edges=%d missing_pin_nodes=%d unresolved_cap_nodes=%d unresolved_res_nodes=%d repaired_edges=%d fallback_nets=%d",
                counts.ground_caps, counts.coupling_caps, counts.folded_coupling_terms,
                counts.resistors, counts.skipped_self_resistors, counts.skipped_loop_edges,
                counts.missing_pin_nodes, counts.unresolved_cap_nodes, counts.unresolved_res_nodes,
                counts.repaired_edges, counts.fallback_nets);
    return graph;
}

void GPUTimer::debug_dump_spef_rc_net(const std::string& net_name) {
    HostRcGraph graph = build_spef_rc();
    std::string normalized_net_name = normalized_spef_name(net_name);
    int net_idx = -1;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        if (gtdb.net_names[i] == net_name || normalized_spef_name(gtdb.net_names[i]) == normalized_net_name) {
            net_idx = i;
            break;
        }
    }
    if (net_idx < 0) {
        logger.warning("SPEF RC dump: net %s not found", net_name.c_str());
        return;
    }
    int nst = graph.net2node_start[net_idx];
    int nend = graph.net2node_start[net_idx + 1];
    int est = graph.net2edge_start[net_idx];
    int eend = graph.net2edge_start[net_idx + 1];
    printf("[SPEF RC DUMP] net=%s id=%d nodes=%d edges=%d includes_pin_caps=%d repaired_edges_total=%d skipped_loop_edges_total=%d\n",
           gtdb.net_names[net_idx].c_str(), net_idx, nend - nst, eend - est,
           graph.includes_pin_caps[net_idx] ? 1 : 0, graph.repaired_edges,
           graph.skipped_loop_edges);
    for (int node = nst; node < nend; ++node) {
        int local_node = node - nst;
        int pin = graph.node2pin[node];
        printf("[SPEF RC DUMP] node local=%d global=%d pin=%d name=%s cap=(%.9e,%.9e,%.9e,%.9e)\n",
               local_node, node, pin, graph.node_names[node].c_str(),
               graph.node_cap[node * NUM_ATTR + 0],
               graph.node_cap[node * NUM_ATTR + 1],
               graph.node_cap[node * NUM_ATTR + 2],
               graph.node_cap[node * NUM_ATTR + 3]);
    }
    for (int edge = est; edge < eend; ++edge) {
        printf("[SPEF RC DUMP] edge local=%d global=%d from=%d to=%d res=%.9e\n",
               edge - est, edge, graph.edge_from[edge] - nst,
               graph.edge_to[edge] - nst, graph.edge_res[edge]);
    }
    fflush(stdout);
}

void GPUTimer::update_rc_timing_spef() {
    HostRcGraph graph = build_spef_rc();
    auto device = timing_raw_db.node_size_x.device();
    torch::Tensor edge_res = torch::from_blob(graph.edge_res.data(), {graph.num_edges}, torch::dtype(torch::kFloat32)).contiguous().to(device);
    torch::Tensor node_cap = torch::from_blob(graph.node_cap.data(), {graph.num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32)).contiguous().to(device);
    torch::Tensor node_order = torch::zeros({graph.num_nodes}, torch::kInt32).contiguous().to(device);
    torch::Tensor edge_order = torch::zeros({graph.num_edges}, torch::kInt32).contiguous().to(device);
    torch::Tensor parent_node = -torch::ones({graph.num_nodes}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor res_parent = torch::zeros({graph.num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32).device(device));

    RcTreeHost rc_tree;
    rc_tree.edge_from = &graph.edge_from;
    rc_tree.edge_to = &graph.edge_to;
    rc_tree.flat_net2node_start_map = &graph.net2node_start;
    rc_tree.flat_net2edge_start_map = &graph.net2edge_start;
    rc_tree.node2pin_map = &graph.node2pin;
    rc_tree.includes_pin_caps = &graph.includes_pin_caps;
    rc_tree.node_order = node_order.data_ptr<int>();
    rc_tree.edge_order = edge_order.data_ptr<int>();
    rc_tree.parent_node = parent_node.data_ptr<int>();
    rc_tree.edge_res = edge_res.data_ptr<float>();
    rc_tree.node_cap = node_cap.data_ptr<float>();
    rc_tree.res_parent = res_parent.data_ptr<float>();
    rc_tree.pinLoad = pinLoad;
    rc_tree.pinImpulse = pinImpulse;
    rc_tree.pinCap = pinCap;
    rc_tree.pinWireCap = pinWireCap;
    rc_tree.pinRootDelay = pinRootDelay;
    rc_tree.pinRootRes = pinRootRes;
    rc_tree.num_nets = num_nets;
    rc_tree.num_pins = num_pins;
    rc_tree.num_nodes = graph.num_nodes;
    rc_tree.num_edges = graph.num_edges;

    flatten_rc_tree(rc_tree);
    propagate_rc_tree(rc_tree);
}
}  // namespace gt
