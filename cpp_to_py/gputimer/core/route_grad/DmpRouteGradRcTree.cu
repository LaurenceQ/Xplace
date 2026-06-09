#include "gputimer/core/route_grad/DmpRouteGrad.h"
#include "gputimer/core/route_grad/DmpRouteGradInternal.h"

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gt {

// RC-tree reverse pass and fixed-topology finite-difference validation.
// This file maps timing adjoints on Elmore/root PI quantities back to route
// segment resistance and node capacitance.
double node_cap_with_optional_pin(const GPUTimer& timer,
                                  const HostRcGraph& graph,
                                  int net_id,
                                  int node,
                                  int attr)
{
    double cap = static_cast<double>(graph.node_cap[static_cast<size_t>(node) * NUM_ATTR + attr]);
    const int pin = graph.node2pin[node];
    if (pin >= 0 && !graph_includes_pin_caps(graph, net_id)) {
        cap += static_cast<double>(openroad_rc::pin_cap_attr_host(timer.gtdb, pin, attr));
    }
    return cap;
}

// Reverse one route-segment RC tree from sink Elmore and root PI adjoints back
// to edge resistance and node capacitance. This corresponds to the M/N/P moment,
// C1/C2/rpi, and path-Elmore formulas in the derivative memo.
void reverse_one_net_rc_tree(const GPUTimer& timer,
                             const HostRcGraph& graph,
                             int net_id,
                             double rc_time_factor,
                             const std::vector<double>& bar_elmore,
                             const std::vector<double>& bar_root_c1,
                             const std::vector<double>& bar_root_c2,
                             const std::vector<double>& bar_root_rpi,
                             std::vector<double>& edge_res_grad,
                             std::vector<double>& node_cap_grad)
{
    const int nst = graph.net2node_start[net_id];
    const int nend = graph.net2node_start[net_id + 1];
    const int est = graph.net2edge_start[net_id];
    const int eend = graph.net2edge_start[net_id + 1];
    const int node_count = nend - nst;
    if (node_count <= 0) {
        return;
    }

    std::vector<std::vector<std::pair<int, int>>> adjacency(static_cast<size_t>(node_count));
    for (int edge = est; edge < eend; ++edge) {
        const int from = graph.edge_from[edge] - nst;
        const int to = graph.edge_to[edge] - nst;
        if (from < 0 || from >= node_count || to < 0 || to >= node_count || from == to) {
            continue;
        }
        adjacency[from].push_back({to, edge});
        adjacency[to].push_back({from, edge});
    }

    std::vector<int> parent(node_count, -2);
    std::vector<int> parent_edge(node_count, -1);
    std::vector<int> order;
    order.reserve(node_count);
    std::queue<int> queue;
    parent[0] = -1;
    queue.push(0);
    while (!queue.empty()) {
        const int node = queue.front();
        queue.pop();
        order.push_back(node);
        for (const auto& next : adjacency[node]) {
            const int child = next.first;
            if (parent[child] != -2) {
                continue;
            }
            parent[child] = node;
            parent_edge[child] = next.second;
            queue.push(child);
        }
    }
    if (order.empty()) {
        return;
    }

    std::vector<std::vector<int>> children(static_cast<size_t>(node_count));
    for (int node = 1; node < node_count; ++node) {
        if (parent[node] >= 0) {
            children[parent[node]].push_back(node);
        }
    }

    std::vector<double> moment_m(node_count);
    std::vector<double> moment_n(node_count);
    std::vector<double> moment_p(node_count);
    std::vector<double> bar_delay(node_count);
    std::vector<double> bar_m(node_count);
    std::vector<double> bar_n(node_count);
    std::vector<double> bar_p(node_count);

    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        std::fill(moment_m.begin(), moment_m.end(), 0.0);
        std::fill(moment_n.begin(), moment_n.end(), 0.0);
        std::fill(moment_p.begin(), moment_p.end(), 0.0);
        std::fill(bar_delay.begin(), bar_delay.end(), 0.0);
        std::fill(bar_m.begin(), bar_m.end(), 0.0);
        std::fill(bar_n.begin(), bar_n.end(), 0.0);
        std::fill(bar_p.begin(), bar_p.end(), 0.0);

        for (int idx = static_cast<int>(order.size()) - 1; idx >= 0; --idx) {
            const int local = order[idx];
            const int global = nst + local;
            moment_m[local] += node_cap_with_optional_pin(timer, graph, net_id, global, attr);
            const int par = parent[local];
            if (par >= 0) {
                const int edge = parent_edge[local];
                if (edge < 0) {
                    continue;
                }
                const double r_scaled = static_cast<double>(graph.edge_res[edge]) * rc_time_factor;
                const double m_child = moment_m[local];
                const double n_child = moment_n[local];
                moment_m[par] += m_child;
                moment_n[par] += n_child - r_scaled * m_child * m_child;
                moment_p[par] += moment_p[local] - 2.0 * r_scaled * m_child * n_child +
                                 r_scaled * r_scaled * m_child * m_child * m_child;
            }
        }

        for (int local : order) {
            const int global = nst + local;
            const int pin = graph.node2pin[global];
            if (pin >= 0 && pin < timer.num_pins) {
                bar_delay[local] += bar_elmore[pin * NUM_ATTR + attr];
            }
        }

        for (int idx = static_cast<int>(order.size()) - 1; idx >= 1; --idx) {
            const int local = order[idx];
            const int edge = parent_edge[local];
            const int par = parent[local];
            if (edge < 0 || par < 0) {
                continue;
            }
            const double adj_delay = bar_delay[local];
            if (adj_delay != 0.0 && std::isfinite(adj_delay)) {
                const double r_scaled = static_cast<double>(graph.edge_res[edge]) * rc_time_factor;
                edge_res_grad[edge] += adj_delay * moment_m[local] * rc_time_factor;
                bar_m[local] += adj_delay * r_scaled;
                bar_delay[par] += adj_delay;
            }
        }

        const int root_global = nst;
        const int root_pin = graph.node2pin[root_global];
        if (root_pin >= 0 && root_pin < timer.num_pins) {
            const int root_slot = root_pin * NUM_ATTR + attr;
            if (root_slot >= 0 && root_slot < static_cast<int>(bar_root_c1.size())) {
                double bc1 = bar_root_c1[root_slot];
                const double bc2 = bar_root_c2[root_slot];
                const double brpi = bar_root_rpi[root_slot];
                const double m_root = moment_m[0];
                const double n_root = moment_n[0];
                const double p_root = moment_p[0];
                if ((bc1 != 0.0 || bc2 != 0.0 || brpi != 0.0) &&
                    std::isfinite(m_root) && std::isfinite(n_root) && std::isfinite(p_root) &&
                    p_root != 0.0 && n_root != 0.0) {
                    bar_m[0] += bc2;
                    bc1 -= bc2;
                    bar_n[0] += bc1 * (2.0 * n_root / p_root) +
                                brpi * (3.0 * p_root * p_root /
                                        (n_root * n_root * n_root * n_root));
                    bar_p[0] += bc1 * (-n_root * n_root / (p_root * p_root)) +
                                brpi * (-2.0 * p_root /
                                        (n_root * n_root * n_root));
                    (void)m_root;
                }
            }
        }

        for (size_t pos = 0; pos < order.size(); ++pos) {
            const int local = order[pos];
            const int global = nst + local;
            if (bar_m[local] != 0.0 && std::isfinite(bar_m[local])) {
                node_cap_grad[global] += bar_m[local];
            }
            for (int child : children[local]) {
                const int edge = parent_edge[child];
                if (edge < 0) {
                    continue;
                }
                const double r_scaled = static_cast<double>(graph.edge_res[edge]) * rc_time_factor;
                const double m_child = moment_m[child];
                const double n_child = moment_n[child];
                const double bp = bar_p[local];
                const double bn = bar_n[local];
                const double bm = bar_m[local];

                bar_m[child] += bm;
                bar_n[child] += bn;
                if (bn != 0.0 && std::isfinite(bn)) {
                    edge_res_grad[edge] += (-bn * m_child * m_child) * rc_time_factor;
                    bar_m[child] += -2.0 * bn * r_scaled * m_child;
                }
                if (bp != 0.0 && std::isfinite(bp)) {
                    edge_res_grad[edge] +=
                        bp * (-2.0 * m_child * n_child +
                              2.0 * r_scaled * m_child * m_child * m_child) * rc_time_factor;
                    bar_m[child] +=
                        bp * (-2.0 * r_scaled * n_child +
                              3.0 * r_scaled * r_scaled * m_child * m_child);
                    bar_n[child] += -2.0 * bp * r_scaled * m_child;
                    bar_p[child] += bp;
                }
            }
        }
    }
}


constexpr int kRouteGradRcTreeCheckColumns = 13;

std::vector<RouteGradRcTreeCheckSample>
sample_rc_tree_check_nets(const HostRcGraph& graph, int sample_net_count, int seed)
{
    if (sample_net_count <= 0) {
        throw std::runtime_error("sample_net_count must be positive for RC-tree gradcheck.");
    }
    const int num_nets = static_cast<int>(graph.net2edge_start.size()) - 1;
    std::vector<int> candidate_nets;
    candidate_nets.reserve(std::max(0, num_nets));
    for (int net = 0; net < num_nets; ++net) {
        const int est = graph.net2edge_start[net];
        const int eend = graph.net2edge_start[net + 1];
        bool has_edge = false;
        for (int edge = est; edge < eend; ++edge) {
            const int to = graph.edge_to[edge];
            if (edge >= 0 && edge < graph.num_edges &&
                to >= 0 && to < graph.num_nodes &&
                std::isfinite(graph.edge_res[edge]) && graph.edge_res[edge] > 0.0f) {
                has_edge = true;
                break;
            }
        }
        if (has_edge) {
            candidate_nets.push_back(net);
        }
    }

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::shuffle(candidate_nets.begin(), candidate_nets.end(), rng);
    if (sample_net_count < static_cast<int>(candidate_nets.size())) {
        candidate_nets.resize(sample_net_count);
    }

    std::vector<RouteGradRcTreeCheckSample> samples;
    samples.reserve(candidate_nets.size());
    for (int net : candidate_nets) {
        const int est = graph.net2edge_start[net];
        const int eend = graph.net2edge_start[net + 1];
        std::vector<int> edges;
        edges.reserve(std::max(0, eend - est));
        for (int edge = est; edge < eend; ++edge) {
            const int to = graph.edge_to[edge];
            if (to >= 0 && to < graph.num_nodes &&
                std::isfinite(graph.edge_res[edge]) && graph.edge_res[edge] > 0.0f) {
                edges.push_back(edge);
            }
        }
        if (edges.empty()) {
            continue;
        }
        std::shuffle(edges.begin(), edges.end(), rng);
        const int edge = edges.front();
        samples.push_back({net, edge, graph.edge_to[edge]});
    }
    return samples;
}

double route_grad_check_weight(int net_id, int object_id, int attr, int channel)
{
    uint64_t x = static_cast<uint64_t>(net_id + 1) * 0x9e3779b185ebca87ULL;
    x ^= static_cast<uint64_t>(object_id + 17) * 0xc2b2ae3d27d4eb4fULL;
    x ^= static_cast<uint64_t>(attr + 31) * 0x165667b19e3779f9ULL;
    x ^= static_cast<uint64_t>(channel + 7) * 0xd6e8feb86659fd93ULL;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    const double mag = 0.05 + static_cast<double>(x & 0x3ffULL) / 4096.0;
    return ((x >> 10) & 1ULL) ? mag : -mag;
}

void seed_rc_tree_check_adjoint(const GPUTimer& timer,
                                const HostRcGraph& graph,
                                const std::vector<RouteGradRcTreeCheckSample>& samples,
                                std::vector<double>& bar_elmore,
                                std::vector<double>& bar_root_c1,
                                std::vector<double>& bar_root_c2,
                                std::vector<double>& bar_root_rpi)
{
    for (const RouteGradRcTreeCheckSample& sample : samples) {
        const int nst = graph.net2node_start[sample.net_id];
        const int nend = graph.net2node_start[sample.net_id + 1];
        for (int node = nst; node < nend; ++node) {
            const int pin = graph.node2pin[node];
            if (pin < 0 || pin >= timer.num_pins) {
                continue;
            }
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                const int slot = pin * NUM_ATTR + attr;
                bar_elmore[slot] += route_grad_check_weight(sample.net_id, pin, attr, 0);
            }
        }
        const int root_pin = graph.node2pin[nst];
        if (root_pin >= 0 && root_pin < timer.num_pins) {
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                const int slot = root_pin * NUM_ATTR + attr;
                bar_root_c1[slot] += route_grad_check_weight(sample.net_id, root_pin, attr, 1);
                bar_root_c2[slot] += route_grad_check_weight(sample.net_id, root_pin, attr, 2);
                bar_root_rpi[slot] += route_grad_check_weight(sample.net_id, root_pin, attr, 3);
            }
        }
    }
}

double route_grad_rc_tree_node_cap_base(const HostRcGraph& graph, int node)
{
    double base = 0.0;
    if (node < 0 || node >= graph.num_nodes) {
        return base;
    }
    const size_t off = static_cast<size_t>(node) * NUM_ATTR;
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        base = std::max(base, std::fabs(static_cast<double>(graph.node_cap[off + attr])));
    }
    return base;
}

double route_grad_rc_tree_safe_rel_err(double fd, double adj)
{
    const double denom = std::max(std::fabs(fd), 1.0e-30);
    if (!std::isfinite(fd) || !std::isfinite(adj) || denom <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::fabs(adj - fd) / denom;
}

// Fixed-topology local scalar used only by rc_tree_gradcheck(). It forms
// bar^T outputs for Elmore and root PI outputs so finite differences can verify
// the RC-tree reverse independently from timing winner changes.
double forward_one_net_rc_tree_value(const GPUTimer& timer,
                                     const HostRcGraph& graph,
                                     int net_id,
                                     double rc_time_factor,
                                     const std::vector<double>& bar_elmore,
                                     const std::vector<double>& bar_root_c1,
                                     const std::vector<double>& bar_root_c2,
                                     const std::vector<double>& bar_root_rpi)
{
    const int nst = graph.net2node_start[net_id];
    const int nend = graph.net2node_start[net_id + 1];
    const int est = graph.net2edge_start[net_id];
    const int eend = graph.net2edge_start[net_id + 1];
    const int node_count = nend - nst;
    if (node_count <= 0) {
        return 0.0;
    }

    std::vector<std::vector<std::pair<int, int>>> adjacency(static_cast<size_t>(node_count));
    for (int edge = est; edge < eend; ++edge) {
        const int from = graph.edge_from[edge] - nst;
        const int to = graph.edge_to[edge] - nst;
        if (from < 0 || from >= node_count || to < 0 || to >= node_count || from == to) {
            continue;
        }
        adjacency[from].push_back({to, edge});
        adjacency[to].push_back({from, edge});
    }

    std::vector<int> parent(node_count, -2);
    std::vector<int> parent_edge(node_count, -1);
    std::vector<int> order;
    order.reserve(node_count);
    std::queue<int> queue;
    parent[0] = -1;
    queue.push(0);
    while (!queue.empty()) {
        const int node = queue.front();
        queue.pop();
        order.push_back(node);
        for (const auto& next : adjacency[node]) {
            const int child = next.first;
            if (parent[child] != -2) {
                continue;
            }
            parent[child] = node;
            parent_edge[child] = next.second;
            queue.push(child);
        }
    }
    if (order.empty()) {
        return 0.0;
    }

    std::vector<std::vector<int>> children(static_cast<size_t>(node_count));
    for (int node = 1; node < node_count; ++node) {
        if (parent[node] >= 0) {
            children[parent[node]].push_back(node);
        }
    }

    std::vector<double> moment_m(node_count);
    std::vector<double> moment_n(node_count);
    std::vector<double> moment_p(node_count);
    std::vector<double> delay(node_count);
    double value = 0.0;

    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        std::fill(moment_m.begin(), moment_m.end(), 0.0);
        std::fill(moment_n.begin(), moment_n.end(), 0.0);
        std::fill(moment_p.begin(), moment_p.end(), 0.0);
        std::fill(delay.begin(), delay.end(), 0.0);

        for (int idx = static_cast<int>(order.size()) - 1; idx >= 0; --idx) {
            const int local = order[idx];
            const int global = nst + local;
            moment_m[local] += node_cap_with_optional_pin(timer, graph, net_id, global, attr);
            const int par = parent[local];
            if (par >= 0) {
                const int edge = parent_edge[local];
                if (edge < 0) {
                    continue;
                }
                const double r_scaled = static_cast<double>(graph.edge_res[edge]) * rc_time_factor;
                const double m_child = moment_m[local];
                const double n_child = moment_n[local];
                moment_m[par] += m_child;
                moment_n[par] += n_child - r_scaled * m_child * m_child;
                moment_p[par] += moment_p[local] - 2.0 * r_scaled * m_child * n_child +
                                 r_scaled * r_scaled * m_child * m_child * m_child;
            }
        }

        for (int local : order) {
            const int global = nst + local;
            const int pin = graph.node2pin[global];
            if (pin >= 0 && pin < timer.num_pins) {
                value += bar_elmore[pin * NUM_ATTR + attr] * delay[local];
            }
            for (int child : children[local]) {
                const int edge = parent_edge[child];
                if (edge < 0) {
                    continue;
                }
                delay[child] = delay[local] +
                               static_cast<double>(graph.edge_res[edge]) *
                                   rc_time_factor * moment_m[child];
            }
        }

        const int root_pin = graph.node2pin[nst];
        if (root_pin >= 0 && root_pin < timer.num_pins) {
            const int root_slot = root_pin * NUM_ATTR + attr;
            const double m_root = moment_m[0];
            const double n_root = moment_n[0];
            const double p_root = moment_p[0];
            if (std::isfinite(m_root) && std::isfinite(n_root) && std::isfinite(p_root) &&
                p_root != 0.0 && n_root != 0.0) {
                const double c1 = n_root * n_root / p_root;
                const double c2 = m_root - c1;
                const double rpi = -p_root * p_root / (n_root * n_root * n_root);
                value += bar_root_c1[root_slot] * c1 +
                         bar_root_c2[root_slot] * c2 +
                         bar_root_rpi[root_slot] * rpi;
            }
        }
    }
    return value;
}

std::vector<std::string> dmp_route_segment_rc_tree_gradcheck_columns()
{
    return {"net_id",
            "edge_id",
            "node_id",
            "edge_fd",
            "edge_adj",
            "edge_abs_err",
            "edge_rel_err",
            "node_fd",
            "node_adj",
            "node_abs_err",
            "node_rel_err",
            "valid_edge",
            "valid_node"};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
make_route_grad_tensors(const HostRcGraph& graph,
                        const std::vector<double>& edge_res_grad,
                        const std::vector<double>& node_cap_grad)
{
    std::vector<double> edge_cap_grad(static_cast<size_t>(graph.num_edges), 0.0);
    for (int edge = 0; edge < graph.num_edges; ++edge) {
        const int from = graph.edge_from[edge];
        const int to = graph.edge_to[edge];
        const double from_grad = (from >= 0 && from < graph.num_nodes) ? node_cap_grad[from] : 0.0;
        const double to_grad = (to >= 0 && to < graph.num_nodes) ? node_cap_grad[to] : 0.0;
        edge_cap_grad[edge] = 0.5 * (from_grad + to_grad);
    }

    auto edge_res_tensor = torch::from_blob(const_cast<double*>(edge_res_grad.data()),
                                            {graph.num_edges},
                                            torch::TensorOptions().dtype(torch::kFloat64))
                               .clone();
    auto node_cap_tensor = torch::from_blob(const_cast<double*>(node_cap_grad.data()),
                                            {graph.num_nodes},
                                            torch::TensorOptions().dtype(torch::kFloat64))
                               .clone();
    auto edge_cap_tensor = torch::from_blob(edge_cap_grad.data(),
                                            {graph.num_edges},
                                            torch::TensorOptions().dtype(torch::kFloat64))
                               .clone();
    return {edge_res_tensor, node_cap_tensor, edge_cap_tensor};
}

}  // namespace


namespace {

constexpr int kValidateColumns = 13;

std::vector<RouteGradFdSample> sample_one_edge_node_per_net(const HostRcGraph& graph,
                                                            int sample_net_count,
                                                            int seed)
{
    if (sample_net_count <= 0) {
        throw std::runtime_error("sample_net_count must be positive.");
    }
    std::vector<int> candidate_nets;
    candidate_nets.reserve(std::max(0, static_cast<int>(graph.net2edge_start.size()) - 1));
    const int num_nets = static_cast<int>(graph.net2edge_start.size()) - 1;
    for (int net = 0; net < num_nets; ++net) {
        const int est = graph.net2edge_start[net];
        const int eend = graph.net2edge_start[net + 1];
        bool has_candidate = false;
        for (int edge = est; edge < eend; ++edge) {
            const int to = graph.edge_to[edge];
            if (edge >= 0 && edge < graph.num_edges &&
                to >= 0 && to < graph.num_nodes &&
                std::isfinite(graph.edge_res[edge]) &&
                graph.edge_res[edge] > 0.0f) {
                has_candidate = true;
                break;
            }
        }
        if (has_candidate) {
            candidate_nets.push_back(net);
        }
    }

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::shuffle(candidate_nets.begin(), candidate_nets.end(), rng);
    if (sample_net_count < static_cast<int>(candidate_nets.size())) {
        candidate_nets.resize(sample_net_count);
    }

    std::vector<RouteGradFdSample> samples;
    samples.reserve(candidate_nets.size());
    for (int net : candidate_nets) {
        const int est = graph.net2edge_start[net];
        const int eend = graph.net2edge_start[net + 1];
        std::vector<int> edges;
        edges.reserve(std::max(0, eend - est));
        for (int edge = est; edge < eend; ++edge) {
            const int to = graph.edge_to[edge];
            if (to >= 0 && to < graph.num_nodes &&
                std::isfinite(graph.edge_res[edge]) &&
                graph.edge_res[edge] > 0.0f) {
                edges.push_back(edge);
            }
        }
        if (edges.empty()) {
            continue;
        }
        std::shuffle(edges.begin(), edges.end(), rng);
        const int edge = edges.front();
        samples.push_back({net, edge, graph.edge_to[edge]});
    }
    return samples;
}

double safe_rel_err(double fd, double adj)
{
    const double denom = std::max(std::fabs(fd), 1.0e-30);
    if (!std::isfinite(fd) || !std::isfinite(adj) || denom <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::fabs(adj - fd) / denom;
}

double tensor_value(const torch::Tensor& tensor, int64_t row, int64_t col)
{
    return tensor.index({row, col}).item<double>();
}

}  // namespace

std::vector<std::string> dmp_route_segment_grad_fd_validate_columns()
{
    return {"net_id",
            "edge_id",
            "node_id",
            "edge_fd",
            "edge_adj",
            "edge_abs_err",
            "edge_rel_err",
            "node_fd",
            "node_adj",
            "node_abs_err",
            "node_rel_err",
            "valid_edge",
            "valid_node"};
}

std::tuple<torch::Tensor, std::vector<std::string>>
GPUTimer::debug_dmp_route_segment_grad_fd_validate(const std::string& route_segments_file,
                                                   int sample_net_count,
                                                   int seed,
                                                   double eps_rel,
                                                   double eps_abs_edge,
                                                   double eps_abs_node,
                                                   double tau_ns)
{
    if (route_segments_file.empty()) {
        throw std::runtime_error("route_segments_file is required for route segment gradient FD validation.");
    }
    if (eps_rel < 0.0 || eps_abs_edge < 0.0 || eps_abs_node < 0.0 ||
        !std::isfinite(eps_rel) || !std::isfinite(eps_abs_edge) ||
        !std::isfinite(eps_abs_node)) {
        throw std::runtime_error("FD validation eps values must be non-negative and finite.");
    }

    HostRcGraph graph = build_openroad_route_segments_rc(route_segments_file);
    std::vector<RouteGradFdSample> samples =
        sample_one_edge_node_per_net(graph, sample_net_count, seed);
    std::vector<int64_t> edge_ids;
    std::vector<int64_t> node_ids;
    edge_ids.reserve(samples.size());
    node_ids.reserve(samples.size());
    for (const RouteGradFdSample& sample : samples) {
        edge_ids.push_back(sample.edge_id);
        node_ids.push_back(sample.node_id);
    }

    auto [edge_grad, node_grad, edge_cap_grad] =
        compute_dmp_route_segment_soft_timing_grad(route_segments_file, tau_ns);
    (void)edge_cap_grad;
    edge_grad = edge_grad.to(torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))
                    .contiguous();
    node_grad = node_grad.to(torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))
                    .contiguous();

    auto [edge_fd, edge_fd_cols] =
        debug_dmp_route_segment_fd_grad(route_segments_file,
                                        "edge_res",
                                        edge_ids,
                                        0,
                                        seed,
                                        eps_rel,
                                        eps_abs_edge,
                                        tau_ns);
    auto [node_fd, node_fd_cols] =
        debug_dmp_route_segment_fd_grad(route_segments_file,
                                        "node_cap",
                                        node_ids,
                                        0,
                                        seed,
                                        eps_rel,
                                        eps_abs_node,
                                        tau_ns);
    (void)edge_fd_cols;
    (void)node_fd_cols;
    edge_fd = edge_fd.to(torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))
                 .contiguous();
    node_fd = node_fd.to(torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))
                 .contiguous();

    std::vector<double> rows(samples.size() * kValidateColumns,
                             std::numeric_limits<double>::quiet_NaN());
    for (size_t row = 0; row < samples.size(); ++row) {
        const RouteGradFdSample& sample = samples[row];
        const double edge_fd_grad = tensor_value(edge_fd, static_cast<int64_t>(row), 6);
        const double node_fd_grad = tensor_value(node_fd, static_cast<int64_t>(row), 6);
        const double edge_adj = edge_grad.index({sample.edge_id}).item<double>();
        const double node_adj = node_grad.index({sample.node_id}).item<double>();
        const double valid_edge = tensor_value(edge_fd, static_cast<int64_t>(row), 8);
        const double valid_node = tensor_value(node_fd, static_cast<int64_t>(row), 8);

        const size_t off = row * kValidateColumns;
        rows[off + 0] = static_cast<double>(sample.net_id);
        rows[off + 1] = static_cast<double>(sample.edge_id);
        rows[off + 2] = static_cast<double>(sample.node_id);
        rows[off + 3] = edge_fd_grad;
        rows[off + 4] = edge_adj;
        rows[off + 5] = std::fabs(edge_adj - edge_fd_grad);
        rows[off + 6] = safe_rel_err(edge_fd_grad, edge_adj);
        rows[off + 7] = node_fd_grad;
        rows[off + 8] = node_adj;
        rows[off + 9] = std::fabs(node_adj - node_fd_grad);
        rows[off + 10] = safe_rel_err(node_fd_grad, node_adj);
        rows[off + 11] = valid_edge;
        rows[off + 12] = valid_node;
    }

    auto table = torch::from_blob(rows.data(),
                                  {static_cast<long>(samples.size()), kValidateColumns},
                                  torch::TensorOptions().dtype(torch::kFloat64))
                     .clone();
    return {table, dmp_route_segment_grad_fd_validate_columns()};
}

}  // namespace gt
