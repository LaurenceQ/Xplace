#include "gputimer/core/openroad/OpenroadRcInternal.h"

namespace gt {

using namespace openroad_rc;

void GPUTimer::debug_dump_openroad_gr_rc_net(const std::string& file,

                                             const std::string& net_name) {
    HostRcGraph graph = build_openroad_gr_rc(file);
    std::unordered_map<std::string, int> net_name_to_index;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        add_gr_name_alias(net_name_to_index, gtdb.net_names[i], i);
    }

    int net_idx = -1;
    auto iter = net_name_to_index.find(net_name);
    if (iter != net_name_to_index.end()) {
        net_idx = iter->second;
    } else {
        std::string normalized = net_name;
        validate_token(normalized);
        iter = net_name_to_index.find(normalized);
        if (iter != net_name_to_index.end()) {
            net_idx = iter->second;
        }
    }
    if (net_idx < 0) {
        logger.warning("OpenROAD GR RC dump: net %s not found", net_name.c_str());
        return;
    }

    int nst = graph.net2node_start[net_idx];
    int nend = graph.net2node_start[net_idx + 1];
    int est = graph.net2edge_start[net_idx];
    int eend = graph.net2edge_start[net_idx + 1];
    printf("[GR RC DUMP] file=%s net=%s id=%d nodes=%d edges=%d includes_pin_caps=%d repaired_edges_total=%d skipped_loop_edges_total=%d\n",
           file.c_str(), gtdb.net_names[net_idx].c_str(), net_idx, nend - nst,
           eend - est, graph.includes_pin_caps[net_idx] ? 1 : 0,
           graph.repaired_edges, graph.skipped_loop_edges);
    for (int node = nst; node < nend; ++node) {
        int local_node = node - nst;
        int pin = graph.node2pin[node];
        printf("[GR RC DUMP] node local=%d global=%d pin=%d name=%s cap=(%.9e,%.9e,%.9e,%.9e)\n",
               local_node, node, pin, graph.node_names[node].c_str(),
               graph.node_cap[node * NUM_ATTR + 0],
               graph.node_cap[node * NUM_ATTR + 1],
               graph.node_cap[node * NUM_ATTR + 2],
               graph.node_cap[node * NUM_ATTR + 3]);
    }
    for (int edge = est; edge < eend; ++edge) {
        printf("[GR RC DUMP] edge local=%d global=%d from=%d to=%d res=%.9e\n",
               edge - est, edge, graph.edge_from[edge] - nst,
               graph.edge_to[edge] - nst, graph.edge_res[edge]);
    }
    fflush(stdout);
}

void GPUTimer::debug_dump_openroad_route_segments_rc_net(
    const std::string& file,
    const std::string& net_name)
{
    HostRcGraph graph = build_openroad_route_segments_rc(file);
    std::unordered_map<std::string, int> net_name_to_index;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        add_gr_name_alias(net_name_to_index, gtdb.net_names[i], i);
    }

    int net_idx = -1;
    auto iter = net_name_to_index.find(net_name);
    if (iter != net_name_to_index.end()) {
        net_idx = iter->second;
    } else {
        std::string normalized = net_name;
        validate_token(normalized);
        iter = net_name_to_index.find(normalized);
        if (iter != net_name_to_index.end()) {
            net_idx = iter->second;
        }
    }
    if (net_idx < 0) {
        logger.warning("OpenROAD route segment RC: net %s not found", net_name.c_str());
        return;
    }

    int nst = graph.net2node_start[net_idx];
    int nend = graph.net2node_start[net_idx + 1];
    int est = graph.net2edge_start[net_idx];
    int eend = graph.net2edge_start[net_idx + 1];
    printf("[ROUTE SEG RC DUMP] file=%s net=%s id=%d nodes=%d edges=%d repaired_edges_total=%d skipped_loop_edges_total=%d\n",
           file.c_str(),
           gtdb.net_names[net_idx].c_str(),
           net_idx,
           nend - nst,
           eend - est,
           graph.repaired_edges,
           graph.skipped_loop_edges);
    auto node_name = [&](int global_node) -> const char* {
        if (global_node >= 0 &&
            global_node < static_cast<int>(graph.node_names.size())) {
            return graph.node_names[global_node].c_str();
        }
        return "";
    };
    for (int node = nst; node < nend; ++node) {
        int local_node = node - nst;
        int pin = graph.node2pin[node];
        printf("[ROUTE SEG RC DUMP] node local=%d global=%d pin=%d name=%s cap=(%.9e,%.9e,%.9e,%.9e)\n",
               local_node,
               node,
               pin,
               node_name(node),
               graph.node_cap[node * NUM_ATTR + 0],
               graph.node_cap[node * NUM_ATTR + 1],
               graph.node_cap[node * NUM_ATTR + 2],
               graph.node_cap[node * NUM_ATTR + 3]);
    }
    for (int edge = est; edge < eend; ++edge) {
        printf("[ROUTE SEG RC DUMP] edge local=%d global=%d from=%d to=%d res=%.9e from_name=%s to_name=%s\n",
               edge - est,
               edge,
               graph.edge_from[edge] - nst,
               graph.edge_to[edge] - nst,
               graph.edge_res[edge],
               node_name(graph.edge_from[edge]),
               node_name(graph.edge_to[edge]));
    }
    fflush(stdout);
}

void GPUTimer::debug_compare_openroad_route_segments_rc(
    const std::string& gr_rc_file,
    const std::string& route_segments_file,
    int top_n)
{
    HostRcGraph gr_graph = build_openroad_gr_rc(gr_rc_file);
    HostRcGraph route_graph = build_openroad_route_segments_rc(route_segments_file);

    struct NetDiff {
        int net_idx = -1;
        int gr_nodes = 0;
        int route_nodes = 0;
        int gr_edges = 0;
        int route_edges = 0;
        double edge_abs = 0.0;
        double cap_abs = 0.0;
        float max_edge_abs = 0.0f;
        float max_cap_abs = 0.0f;
        int max_edge = -1;
        int max_cap_node = -1;
        int max_cap_attr = -1;
        bool shape_mismatch = false;
    };

    struct SemanticEdgeValue {
        double res = 0.0;
        int count = 0;
    };

    struct SemanticNetDiff {
        int net_idx = -1;
        int gr_nodes = 0;
        int route_nodes = 0;
        int gr_edges = 0;
        int route_edges = 0;
        int missing_nodes = 0;
        int missing_edges = 0;
        int edge_count_mismatches = 0;
        int fallback_keys = 0;
        double edge_abs = 0.0;
        double cap_abs = 0.0;
        double max_edge_abs = 0.0;
        double max_cap_abs = 0.0;
        double max_edge_gr_res = 0.0;
        double max_edge_route_res = 0.0;
        std::string max_edge_key;
        std::string max_cap_key;
    };

    auto semantic_node_key = [&](const HostRcGraph& graph,
                                 int global_node,
                                 int net_node_start,
                                 int& fallback_keys) {
        const int pin_id = graph.node2pin[global_node];
        if (pin_id >= 0) {
            std::string key = std::string("P:") + std::to_string(pin_id);
            if (pin_id < static_cast<int>(gtdb.pin_names.size())) {
                key += ":" + gtdb.pin_names[pin_id];
            }
            return key;
        }
        if (global_node >= 0 && global_node < static_cast<int>(graph.node_names.size()) &&
            !graph.node_names[global_node].empty()) {
            return std::string("R:") + graph.node_names[global_node];
        }
        fallback_keys++;
        return std::string("L:") + std::to_string(global_node - net_node_start);
    };

    auto semantic_edge_key = [](const std::string& lhs, const std::string& rhs) {
        if (lhs < rhs) {
            return lhs + "\t" + rhs;
        }
        return rhs + "\t" + lhs;
    };

    auto build_semantic_node_caps =
        [&](const HostRcGraph& graph,
            int nst,
            int nend,
            int& fallback_keys) {
            std::unordered_map<std::string, std::array<double, NUM_ATTR>> caps;
            for (int node = nst; node < nend; ++node) {
                const std::string key = semantic_node_key(graph, node, nst, fallback_keys);
                auto& cap = caps[key];
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    cap[attr] += graph.node_cap[node * NUM_ATTR + attr];
                }
            }
            return caps;
        };

    auto build_semantic_edges =
        [&](const HostRcGraph& graph,
            int nst,
            int est,
            int eend,
            int& fallback_keys) {
            std::unordered_map<std::string, SemanticEdgeValue> edges;
            for (int edge = est; edge < eend; ++edge) {
                const std::string from =
                    semantic_node_key(graph, graph.edge_from[edge], nst, fallback_keys);
                const std::string to =
                    semantic_node_key(graph, graph.edge_to[edge], nst, fallback_keys);
                auto& value = edges[semantic_edge_key(from, to)];
                value.res += graph.edge_res[edge];
                value.count++;
            }
            return edges;
        };

    std::vector<NetDiff> diffs;
    std::vector<SemanticNetDiff> semantic_diffs;
    diffs.reserve(num_nets);
    semantic_diffs.reserve(num_nets);
    double total_edge_abs = 0.0;
    double total_cap_abs = 0.0;
    double semantic_total_edge_abs = 0.0;
    double semantic_total_cap_abs = 0.0;
    int shape_mismatch_nets = 0;
    int semantic_shape_mismatch_nets = 0;
    int semantic_missing_nodes = 0;
    int semantic_missing_edges = 0;
    int semantic_edge_count_mismatches = 0;
    int semantic_fallback_keys = 0;

    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        const int gr_nst = gr_graph.net2node_start[net_idx];
        const int gr_nend = gr_graph.net2node_start[net_idx + 1];
        const int gr_est = gr_graph.net2edge_start[net_idx];
        const int gr_eend = gr_graph.net2edge_start[net_idx + 1];
        const int route_nst = route_graph.net2node_start[net_idx];
        const int route_nend = route_graph.net2node_start[net_idx + 1];
        const int route_est = route_graph.net2edge_start[net_idx];
        const int route_eend = route_graph.net2edge_start[net_idx + 1];

        NetDiff diff;
        diff.net_idx = net_idx;
        diff.gr_nodes = gr_nend - gr_nst;
        diff.route_nodes = route_nend - route_nst;
        diff.gr_edges = gr_eend - gr_est;
        diff.route_edges = route_eend - route_est;
        diff.shape_mismatch = diff.gr_nodes != diff.route_nodes ||
                              diff.gr_edges != diff.route_edges;
        if (diff.shape_mismatch) {
            shape_mismatch_nets++;
        }

        const int edge_count = std::min(diff.gr_edges, diff.route_edges);
        for (int edge = 0; edge < edge_count; ++edge) {
            const float delta = std::fabs(gr_graph.edge_res[gr_est + edge] -
                                          route_graph.edge_res[route_est + edge]);
            diff.edge_abs += delta;
            if (delta > diff.max_edge_abs) {
                diff.max_edge_abs = delta;
                diff.max_edge = edge;
            }
        }

        const int node_count = std::min(diff.gr_nodes, diff.route_nodes);
        for (int node = 0; node < node_count; ++node) {
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                const float delta = std::fabs(
                    gr_graph.node_cap[(gr_nst + node) * NUM_ATTR + attr] -
                    route_graph.node_cap[(route_nst + node) * NUM_ATTR + attr]);
                diff.cap_abs += delta;
                if (delta > diff.max_cap_abs) {
                    diff.max_cap_abs = delta;
                    diff.max_cap_node = node;
                    diff.max_cap_attr = attr;
                }
            }
        }

        total_edge_abs += diff.edge_abs;
        total_cap_abs += diff.cap_abs;
        if (diff.shape_mismatch || diff.edge_abs > 0.0 || diff.cap_abs > 0.0) {
            diffs.emplace_back(diff);
        }

        SemanticNetDiff sem_diff;
        sem_diff.net_idx = net_idx;
        sem_diff.gr_nodes = diff.gr_nodes;
        sem_diff.route_nodes = diff.route_nodes;
        sem_diff.gr_edges = diff.gr_edges;
        sem_diff.route_edges = diff.route_edges;

        int net_fallback_keys = 0;
        auto gr_node_caps = build_semantic_node_caps(gr_graph, gr_nst, gr_nend, net_fallback_keys);
        auto route_node_caps =
            build_semantic_node_caps(route_graph, route_nst, route_nend, net_fallback_keys);
        auto gr_edges =
            build_semantic_edges(gr_graph, gr_nst, gr_est, gr_eend, net_fallback_keys);
        auto route_edges =
            build_semantic_edges(route_graph, route_nst, route_est, route_eend, net_fallback_keys);
        sem_diff.fallback_keys = net_fallback_keys;

        std::unordered_set<std::string> node_keys;
        node_keys.reserve(gr_node_caps.size() + route_node_caps.size());
        for (const auto& item : gr_node_caps) {
            node_keys.insert(item.first);
        }
        for (const auto& item : route_node_caps) {
            node_keys.insert(item.first);
        }
        for (const std::string& key : node_keys) {
            const auto gr_iter = gr_node_caps.find(key);
            const auto route_iter = route_node_caps.find(key);
            if (gr_iter == gr_node_caps.end() || route_iter == route_node_caps.end()) {
                sem_diff.missing_nodes++;
            }
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                const double gr_cap =
                    gr_iter == gr_node_caps.end() ? 0.0 : gr_iter->second[attr];
                const double route_cap =
                    route_iter == route_node_caps.end() ? 0.0 : route_iter->second[attr];
                const double delta = std::fabs(gr_cap - route_cap);
                sem_diff.cap_abs += delta;
                if (delta > sem_diff.max_cap_abs) {
                    sem_diff.max_cap_abs = delta;
                    sem_diff.max_cap_key = key + "/attr" + std::to_string(attr);
                }
            }
        }

        std::unordered_set<std::string> edge_keys;
        edge_keys.reserve(gr_edges.size() + route_edges.size());
        for (const auto& item : gr_edges) {
            edge_keys.insert(item.first);
        }
        for (const auto& item : route_edges) {
            edge_keys.insert(item.first);
        }
        for (const std::string& key : edge_keys) {
            const auto gr_iter = gr_edges.find(key);
            const auto route_iter = route_edges.find(key);
            if (gr_iter == gr_edges.end() || route_iter == route_edges.end()) {
                sem_diff.missing_edges++;
            }
            const double gr_res = gr_iter == gr_edges.end() ? 0.0 : gr_iter->second.res;
            const double route_res =
                route_iter == route_edges.end() ? 0.0 : route_iter->second.res;
            const int gr_count = gr_iter == gr_edges.end() ? 0 : gr_iter->second.count;
            const int route_count = route_iter == route_edges.end() ? 0 : route_iter->second.count;
            if (gr_count != route_count) {
                sem_diff.edge_count_mismatches++;
            }
            const double delta = std::fabs(gr_res - route_res);
            sem_diff.edge_abs += delta;
            if (delta > sem_diff.max_edge_abs) {
                sem_diff.max_edge_abs = delta;
                sem_diff.max_edge_gr_res = gr_res;
                sem_diff.max_edge_route_res = route_res;
                sem_diff.max_edge_key = key;
            }
        }

        semantic_total_edge_abs += sem_diff.edge_abs;
        semantic_total_cap_abs += sem_diff.cap_abs;
        semantic_missing_nodes += sem_diff.missing_nodes;
        semantic_missing_edges += sem_diff.missing_edges;
        semantic_edge_count_mismatches += sem_diff.edge_count_mismatches;
        semantic_fallback_keys += sem_diff.fallback_keys;
        const bool semantic_shape_mismatch =
            sem_diff.missing_nodes > 0 ||
            sem_diff.missing_edges > 0 ||
            sem_diff.edge_count_mismatches > 0 ||
            sem_diff.fallback_keys > 0;
        if (semantic_shape_mismatch) {
            semantic_shape_mismatch_nets++;
        }
        if (semantic_shape_mismatch ||
            sem_diff.edge_abs > 0.0 ||
            sem_diff.cap_abs > 0.0) {
            semantic_diffs.emplace_back(std::move(sem_diff));
        }
    }

    std::sort(diffs.begin(), diffs.end(), [](const NetDiff& lhs, const NetDiff& rhs) {
        if (lhs.shape_mismatch != rhs.shape_mismatch) {
            return lhs.shape_mismatch > rhs.shape_mismatch;
        }
        const double lhs_score = lhs.edge_abs + lhs.cap_abs;
        const double rhs_score = rhs.edge_abs + rhs.cap_abs;
        return lhs_score > rhs_score;
    });

    std::sort(semantic_diffs.begin(),
              semantic_diffs.end(),
              [](const SemanticNetDiff& lhs, const SemanticNetDiff& rhs) {
                  const bool lhs_shape = lhs.missing_nodes > 0 ||
                                         lhs.missing_edges > 0 ||
                                         lhs.edge_count_mismatches > 0 ||
                                         lhs.fallback_keys > 0;
                  const bool rhs_shape = rhs.missing_nodes > 0 ||
                                         rhs.missing_edges > 0 ||
                                         rhs.edge_count_mismatches > 0 ||
                                         rhs.fallback_keys > 0;
                  if (lhs_shape != rhs_shape) {
                      return lhs_shape > rhs_shape;
                  }
                  const double lhs_score = lhs.edge_abs + lhs.cap_abs;
                  const double rhs_score = rhs.edge_abs + rhs.cap_abs;
                  return lhs_score > rhs_score;
              });

    printf("[GR RC COMPARE] gr_rc=%s\n", gr_rc_file.c_str());
    printf("[GR RC COMPARE] route_segments=%s\n", route_segments_file.c_str());
    printf("[GR RC SEMCOMPARE] shape_mismatch_nets=%d missing_nodes=%d missing_edges=%d edge_count_mismatches=%d fallback_keys=%d total_edge_abs=%.9e total_cap_abs=%.9e changed_nets=%zu\n",
           semantic_shape_mismatch_nets,
           semantic_missing_nodes,
           semantic_missing_edges,
           semantic_edge_count_mismatches,
           semantic_fallback_keys,
           semantic_total_edge_abs,
           semantic_total_cap_abs,
           semantic_diffs.size());
    const int semantic_limit =
        std::max(0, std::min(top_n, static_cast<int>(semantic_diffs.size())));
    for (int i = 0; i < semantic_limit; ++i) {
        const SemanticNetDiff& diff = semantic_diffs[i];
        printf("[GR RC SEMCOMPARE] rank=%d net=%s id=%d nodes=%d/%d edges=%d/%d missing_nodes=%d missing_edges=%d edge_count_mismatches=%d fallback_keys=%d edge_abs=%.9e cap_abs=%.9e max_edge_abs=%.9e max_edge_gr=%.9e max_edge_route=%.9e edge_key=%s max_cap_abs=%.9e cap_key=%s\n",
               i + 1,
               gtdb.net_names[diff.net_idx].c_str(),
               diff.net_idx,
               diff.gr_nodes,
               diff.route_nodes,
               diff.gr_edges,
               diff.route_edges,
               diff.missing_nodes,
               diff.missing_edges,
               diff.edge_count_mismatches,
               diff.fallback_keys,
               diff.edge_abs,
               diff.cap_abs,
               diff.max_edge_abs,
               diff.max_edge_gr_res,
               diff.max_edge_route_res,
               diff.max_edge_key.c_str(),
               diff.max_cap_abs,
               diff.max_cap_key.c_str());
    }
    printf("[GR RC COMPARE] gr nodes=%d edges=%d route nodes=%d edges=%d shape_mismatch_nets=%d total_edge_abs=%.9e total_cap_abs=%.9e changed_nets=%zu\n",
           gr_graph.num_nodes,
           gr_graph.num_edges,
           route_graph.num_nodes,
           route_graph.num_edges,
           shape_mismatch_nets,
           total_edge_abs,
           total_cap_abs,
           diffs.size());

    const int limit = std::max(0, std::min(top_n, static_cast<int>(diffs.size())));
    for (int i = 0; i < limit; ++i) {
        const NetDiff& diff = diffs[i];
        printf("[GR RC COMPARE] rank=%d net=%s id=%d nodes=%d/%d edges=%d/%d shape_mismatch=%d edge_abs=%.9e cap_abs=%.9e max_edge_abs=%.9e edge=%d max_cap_abs=%.9e cap_node=%d cap_attr=%d\n",
               i + 1,
               gtdb.net_names[diff.net_idx].c_str(),
               diff.net_idx,
               diff.gr_nodes,
               diff.route_nodes,
               diff.gr_edges,
               diff.route_edges,
               diff.shape_mismatch ? 1 : 0,
               diff.edge_abs,
               diff.cap_abs,
               diff.max_edge_abs,
               diff.max_edge,
               diff.max_cap_abs,
               diff.max_cap_node,
               diff.max_cap_attr);
    }
    fflush(stdout);
}

}  // namespace gt
