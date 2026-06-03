#include "gputimer/core/openroad/OpenroadRcInternal.h"

namespace gt {
namespace openroad_rc {

void add_attr_cap(std::vector<float>& node_cap, int node, float cap)
{
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        node_cap[node * NUM_ATTR + attr] += cap;
    }
}

void set_attr_cap(std::vector<float>& node_cap, int node, float cap)
{
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        node_cap[node * NUM_ATTR + attr] = cap;
    }
}

void set_attr_cap(std::vector<float>& node_cap, int node, int attr, float cap)
{
    if (attr >= 0 && attr < NUM_ATTR) {
        node_cap[node * NUM_ATTR + attr] = cap;
    }
}

float pin_cap_attr_host(const GTDatabase& gtdb, int pin, int attr)
{
    if (pin < 0 || attr < 0 || attr >= NUM_ATTR) {
        return 0.0f;
    }
    const int stride = NUM_ATTR + 2;
    const int base = pin * stride;
    if (base + stride > static_cast<int>(gtdb.pin_capacitance.size())) {
        return 0.0f;
    }
    float cap = gtdb.pin_capacitance[base + attr];
    if (std::isfinite(cap)) {
        return cap;
    }
    cap = gtdb.pin_capacitance[base + NUM_ATTR + (attr >> 1)];
    return std::isfinite(cap) ? cap : 0.0f;
}

std::string openroad_gr_edge_key(int from, int to, const std::string& res_id)
{
    return std::to_string(from) + '\t' + std::to_string(to) + '\t' + res_id;
}

void ensure_local_node(LocalSpefNetRc& local, int node_id)
{
    while (node_id >= static_cast<int>(local.node2pin.size())) {
        local.node2pin.emplace_back(-1);
        local.node_names.emplace_back("");
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            local.node_cap.emplace_back(0.0f);
        }
    }
}

std::string spef_upper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

bool spef_includes_pin_caps_from_design_flow(const std::string& design_flow)
{
    std::string flow = spef_upper(design_flow);
    size_t pin_cap_pos = flow.find("PIN_CAP");
    if (pin_cap_pos == std::string::npos) {
        return false;
    }
    std::string pin_cap_clause = flow.substr(pin_cap_pos);
    return pin_cap_clause.find("NONE") == std::string::npos;
}

int count_tree_edges_from_root(const LocalSpefNetRc& local)
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

FlatLocalAdjacency build_flat_local_adjacency(const LocalSpefNetRc& local)
{
    FlatLocalAdjacency adjacency;
    const int node_count = static_cast<int>(local.node2pin.size());
    adjacency.start.assign(static_cast<std::size_t>(node_count) + 1, 0);

    for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
        const int from = local.edge_from[edge];
        const int to = local.edge_to[edge];
        if (from >= 0 && to >= 0 && from < node_count && to < node_count) {
            ++adjacency.start[from + 1];
            ++adjacency.start[to + 1];
        }
    }
    for (int node = 0; node < node_count; ++node) {
        adjacency.start[node + 1] += adjacency.start[node];
    }

    const int adjacency_size = adjacency.start[node_count];
    adjacency.edge.resize(adjacency_size);
    adjacency.next.resize(adjacency_size);
    std::vector<int> cursor(adjacency.start.begin(), adjacency.start.end() - 1);
    for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
        const int from = local.edge_from[edge];
        const int to = local.edge_to[edge];
        if (from >= 0 && to >= 0 && from < node_count && to < node_count) {
            int pos = cursor[from]++;
            adjacency.edge[pos] = static_cast<int>(edge);
            adjacency.next[pos] = to;
            pos = cursor[to]++;
            adjacency.edge[pos] = static_cast<int>(edge);
            adjacency.next[pos] = from;
        }
    }

    return adjacency;
}

int prune_to_rooted_tree(LocalSpefNetRc& local)
{
    if (local.node2pin.empty() || local.edge_from.empty()) {
        return 0;
    }

    const int node_count = static_cast<int>(local.node2pin.size());
    std::vector<uint8_t> seen(node_count, 0);
    std::vector<uint8_t> keep(local.edge_from.size(), 0);
    std::vector<int> stack;
    seen[0] = 1;
    stack.emplace_back(0);

    const long long node_edge_product =
        static_cast<long long>(node_count) *
        static_cast<long long>(local.edge_from.size());
    if (node_edge_product > 4096) {
        const FlatLocalAdjacency adjacency = build_flat_local_adjacency(local);
        for (std::size_t cursor = 0; cursor < stack.size(); ++cursor) {
            const int node = stack[cursor];
            for (int pos = adjacency.start[node]; pos < adjacency.start[node + 1]; ++pos) {
                const int next = adjacency.next[pos];
                if (next >= 0 && next < node_count && !seen[next]) {
                    seen[next] = 1;
                    stack.emplace_back(next);
                    keep[adjacency.edge[pos]] = 1;
                }
            }
        }
    } else {
        for (std::size_t cursor = 0; cursor < stack.size(); ++cursor) {
            const int node = stack[cursor];
            for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
                const int from = local.edge_from[edge];
                const int to = local.edge_to[edge];
                int next = -1;
                if (from == node) next = to;
                if (to == node) next = from;
                if (next >= 0 && next < node_count && !seen[next]) {
                    seen[next] = 1;
                    stack.emplace_back(next);
                    keep[edge] = 1;
                }
            }
        }
    }

    int kept_edges = 0;
    for (uint8_t keep_edge : keep) {
        if (keep_edge) {
            kept_edges++;
        }
    }
    const int dropped_edges = static_cast<int>(local.edge_from.size()) - kept_edges;
    if (dropped_edges == 0) {
        return 0;
    }

    std::vector<int> edge_from;
    std::vector<int> edge_to;
    std::vector<float> edge_res;
    edge_from.reserve(kept_edges);
    edge_to.reserve(kept_edges);
    edge_res.reserve(kept_edges);
    for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
        if (!keep[edge]) {
            continue;
        }
        edge_from.emplace_back(local.edge_from[edge]);
        edge_to.emplace_back(local.edge_to[edge]);
        edge_res.emplace_back(local.edge_res[edge]);
    }
    local.edge_from.swap(edge_from);
    local.edge_to.swap(edge_to);
    local.edge_res.swap(edge_res);
    return dropped_edges;
}


int append_blank_node(LocalSpefNetRc& local,
                      int pin_id,
                      const std::string& name,
                      const OpenroadRoutePt& route_pt,
                      bool keep_route_node_names)
{
    const int node_id = static_cast<int>(local.node2pin.size());
    local.node2pin.emplace_back(pin_id);
    if (keep_route_node_names) {
        local.node_names.emplace_back(name);
    }
    local.route_points.emplace_back(route_pt);
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        local.node_cap.emplace_back(0.0f);
    }
    return node_id;
}

int append_pin_node(LocalSpefNetRc& local,
                    int pin_id,
                    const std::vector<std::string>& pin_names,
                    bool keep_route_node_names)
{
    OpenroadRoutePt route_pt;
    std::string name;
    if (keep_route_node_names && pin_id >= 0 && pin_id < static_cast<int>(pin_names.size())) {
        name = pin_names[pin_id];
    }
    return append_blank_node(local, pin_id, name, route_pt, keep_route_node_names);
}

int append_route_node(int net_idx,
                      const OpenroadRoutePtKey& key,
                      std::vector<std::unique_ptr<LocalSpefNetRc>>& local_nets,
                      std::vector<std::unique_ptr<OpenroadRouteNodeMap>>& route_node_maps,
                      bool keep_route_node_names)
{
    constexpr int route_node_linear_scan_limit = 16;
    auto& local_ptr = local_nets[net_idx];
    if (!local_ptr) {
        local_ptr = std::make_unique<LocalSpefNetRc>();
    }
    LocalSpefNetRc& local = *local_ptr;
    auto& map_ptr = route_node_maps[net_idx];
    if (map_ptr) {
        auto iter = map_ptr->find(key);
        if (iter != map_ptr->end()) {
            return iter->second;
        }
    } else {
        for (int node = 0; node < static_cast<int>(local.route_points.size()); ++node) {
            if (route_point_matches(local.route_points[node], key)) {
                return node;
            }
        }
        if (static_cast<int>(local.route_points.size()) >= route_node_linear_scan_limit) {
            map_ptr = std::make_unique<OpenroadRouteNodeMap>();
            map_ptr->reserve(local.route_points.size() * 2 + 1);
            for (int node = 0; node < static_cast<int>(local.route_points.size()); ++node) {
                const OpenroadRoutePt& pt = local.route_points[node];
                if (pt.valid) {
                    map_ptr->emplace(OpenroadRoutePtKey{pt.x, pt.y, pt.layer}, node);
                }
            }
        }
    }
    OpenroadRoutePt route_pt{key.x, key.y, key.layer, true};
    std::string name;
    if (keep_route_node_names) {
        name = std::to_string(key.x) + "," + std::to_string(key.y) + ",M" +
               std::to_string(key.layer);
    }
    const int node_id = append_blank_node(local, -1, name, route_pt, keep_route_node_names);
    if (map_ptr) {
        map_ptr->emplace(key, node_id);
    }
    return node_id;
}

void add_edge(LocalSpefNetRc& local, int from, int to, float res)
{
    local.edge_from.emplace_back(from);
    local.edge_to.emplace_back(to);
    local.edge_res.emplace_back(res);
}

void reorder_root(LocalSpefNetRc& local, int root_node)
{
    if (root_node <= 0 || root_node >= static_cast<int>(local.node2pin.size())) {
        return;
    }
    std::swap(local.node2pin[0], local.node2pin[root_node]);
    if (root_node < static_cast<int>(local.node_names.size())) {
        std::swap(local.node_names[0], local.node_names[root_node]);
    }
    if (root_node < static_cast<int>(local.route_points.size())) {
        std::swap(local.route_points[0], local.route_points[root_node]);
    }
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        std::swap(local.node_cap[attr], local.node_cap[root_node * NUM_ATTR + attr]);
    }
    for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
        if (local.edge_from[edge] == 0) {
            local.edge_from[edge] = root_node;
        } else if (local.edge_from[edge] == root_node) {
            local.edge_from[edge] = 0;
        }
        if (local.edge_to[edge] == 0) {
            local.edge_to[edge] = root_node;
        } else if (local.edge_to[edge] == root_node) {
            local.edge_to[edge] = 0;
        }
    }
}

}  // namespace openroad_rc
}  // namespace gt
