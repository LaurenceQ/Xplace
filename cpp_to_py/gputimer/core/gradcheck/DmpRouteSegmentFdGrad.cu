#include "gputimer/core/DmpModel.h"
#include "gputimer/core/GPUTimer.h"
#include "gputimer/db/GTDatabase.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gt {

void calc_res_cap_dmp(DmpModel* dmp_db, int num_nets);
void propagate_rc_tree_dmp(DmpModel* dmp_db, int num_nets);
void dmp_prepare_timing_after_rc(DmpModel* h_dmp_db, DmpModel* dmp_db);

namespace {

constexpr int kFdColumns = 9;

void fd_check_cuda(cudaError_t err, const char* label)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in route-segment FD gradcheck at ") +
                                 label + ": " + cudaGetErrorString(err));
    }
}

void release_fd_dmp_state(GPUTimer& timer)
{
    if (timer.h_dmp_db != nullptr) {
        timer.h_dmp_db->release_rc_transient();
        timer.h_dmp_db->release_after_timing();
        delete timer.h_dmp_db;
        timer.h_dmp_db = nullptr;
    }
    if (timer.dmp_db != nullptr) {
        fd_check_cuda(cudaFree(timer.dmp_db), "free dmp_db descriptor");
        timer.dmp_db = nullptr;
    }
}

double compute_late_endpoint_logsumexp_ns(GPUTimer& timer, double tau_ns)
{
    if (!(tau_ns > 0.0) || !std::isfinite(tau_ns)) {
        throw std::runtime_error("tau_ns must be positive and finite.");
    }

    torch::Tensor endpoint_ids;
    if (timer.timing_raw_db.endpoint_unique_pin_ids.defined() &&
        timer.timing_raw_db.endpoint_unique_pin_ids.numel() > 0) {
        endpoint_ids = timer.timing_raw_db.endpoint_unique_pin_ids;
    } else if (timer.timing_raw_db.endpoints_id.defined() &&
               timer.timing_raw_db.endpoints_id.numel() > 0) {
        endpoint_ids = std::get<0>(torch::_unique(timer.timing_raw_db.endpoints_id));
    } else {
        throw std::runtime_error("No endpoint pin ids are available for FD objective.");
    }

    const torch::Tensor pin_at = timer.timing_raw_db.pinAT;
    const torch::Tensor pin_rat = timer.timing_raw_db.pinRAT;
    endpoint_ids = endpoint_ids.to(torch::TensorOptions().dtype(torch::kLong).device(pin_at.device()));

    const double time_to_ns = static_cast<double>(timer.time_unit()) * 1.0e9;
    torch::Tensor violation =
        (pin_at.index_select(0, endpoint_ids) - pin_rat.index_select(0, endpoint_ids))
            .narrow(1, 2, 2)
            .to(torch::kFloat64) *
        time_to_ns;
    torch::Tensor flat = violation.reshape({-1});
    torch::Tensor valid = torch::isfinite(flat) & (flat > -1.0e20) & (flat < 1.0e20);
    torch::Tensor values = flat.masked_select(valid);
    if (values.numel() == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double max_value = values.max().item<double>();
    const double sum_exp = torch::exp((values - max_value) / tau_ns).sum().item<double>();
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return max_value + tau_ns * std::log(sum_exp);
}

void run_route_segment_dmp_from_graph(GPUTimer& timer, HostRcGraph& graph)
{
    release_fd_dmp_state(timer);
    timer.update_states();
    timer.initialize_dmp_rc_explicit(graph.edge_from,
                                     graph.edge_to,
                                     graph.net2node_start,
                                     graph.net2edge_start,
                                     graph.node2pin,
                                     graph.edge_res,
                                     graph.node_cap,
                                     graph.includes_pin_caps,
                                     timer.num_nets,
                                     graph.num_nodes,
                                     graph.num_edges);
    calc_res_cap_dmp(timer.dmp_db, timer.num_nets);
    propagate_rc_tree_dmp(timer.dmp_db, timer.num_nets);
    dmp_prepare_timing_after_rc(timer.h_dmp_db, timer.dmp_db);
    timer.update_timing_dmp();
    fd_check_cuda(cudaDeviceSynchronize(), "run_route_segment_dmp_from_graph");
}

std::vector<int64_t> normalize_ids(const std::vector<int64_t>& ids,
                                   int limit,
                                   const char* kind)
{
    std::vector<int64_t> out;
    out.reserve(ids.size());
    for (const int64_t id : ids) {
        if (id < 0 || id >= limit) {
            throw std::runtime_error(std::string("FD gradcheck ") + kind +
                                     " id out of range: " + std::to_string(id) +
                                     " limit=" + std::to_string(limit));
        }
        out.push_back(id);
    }
    return out;
}

std::vector<int64_t> sample_positive_ids(const std::vector<float>& values,
                                         int logical_count,
                                         bool node_cap_mode,
                                         int sample_count,
                                         int seed)
{
    std::vector<int64_t> candidates;
    candidates.reserve(logical_count);
    for (int id = 0; id < logical_count; ++id) {
        double base = 0.0;
        if (node_cap_mode) {
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                base += static_cast<double>(values[static_cast<size_t>(id) * NUM_ATTR + attr]);
            }
            base /= static_cast<double>(NUM_ATTR);
        } else {
            base = values[id];
        }
        if (std::isfinite(base) && base > 0.0) {
            candidates.push_back(id);
        }
    }

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::shuffle(candidates.begin(), candidates.end(), rng);
    if (sample_count < static_cast<int>(candidates.size())) {
        candidates.resize(std::max(sample_count, 0));
    }
    return candidates;
}

double node_cap_base_value(const HostRcGraph& graph, int64_t node_id)
{
    double base = 0.0;
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        base += static_cast<double>(graph.node_cap[static_cast<size_t>(node_id) * NUM_ATTR + attr]);
    }
    return base / static_cast<double>(NUM_ATTR);
}

double perturb_node_cap(HostRcGraph& graph, int64_t node_id, double eps)
{
    const size_t base = static_cast<size_t>(node_id) * NUM_ATTR;
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        graph.node_cap[base + attr] += static_cast<float>(eps);
    }
    return node_cap_base_value(graph, node_id);
}

void restore_node_cap(HostRcGraph& graph, int64_t node_id, const float original[NUM_ATTR])
{
    const size_t base = static_cast<size_t>(node_id) * NUM_ATTR;
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        graph.node_cap[base + attr] = original[attr];
    }
}

std::vector<std::string> fd_column_names()
{
    return {"id",
            "base_value",
            "eps",
            "y0_ns",
            "y1_ns",
            "delta_y_ns",
            "fd_grad_ns_per_input_unit",
            "rel_step",
            "valid"};
}

void try_restore_baseline_state(GPUTimer& timer, HostRcGraph& graph) noexcept
{
    try {
        run_route_segment_dmp_from_graph(timer, graph);
    } catch (...) {
    }
}

}  // namespace

std::tuple<torch::Tensor, std::vector<std::string>>
GPUTimer::debug_dmp_route_segment_fd_grad(const std::string& route_segments_file,
                                          const std::string& kind,
                                          const std::vector<int64_t>& ids,
                                          int sample_count,
                                          int seed,
                                          double eps_rel,
                                          double eps_abs,
                                          double tau_ns)
{
    if (route_segments_file.empty()) {
        throw std::runtime_error("route_segments_file is required for FD gradcheck.");
    }
    if (eps_rel < 0.0 || eps_abs < 0.0 || !std::isfinite(eps_rel) || !std::isfinite(eps_abs)) {
        throw std::runtime_error("eps_rel and eps_abs must be non-negative and finite.");
    }

    const bool edge_res_mode = kind == "edge_res";
    const bool node_cap_mode = kind == "node_cap";
    if (!edge_res_mode && !node_cap_mode) {
        throw std::runtime_error("FD gradcheck kind must be 'edge_res' or 'node_cap'.");
    }

    HostRcGraph graph = build_openroad_route_segments_rc(route_segments_file);
    if (graph.node_cap.size() != static_cast<size_t>(graph.num_nodes) * NUM_ATTR) {
        throw std::runtime_error("Route-segment RC graph has invalid node_cap shape.");
    }

    std::vector<int64_t> work_ids;
    if (!ids.empty()) {
        work_ids = normalize_ids(ids,
                                 edge_res_mode ? graph.num_edges : graph.num_nodes,
                                 edge_res_mode ? "edge_res" : "node_cap");
    } else {
        if (sample_count <= 0) {
            throw std::runtime_error("FD gradcheck requires explicit ids or a positive sample_count.");
        }
        work_ids = sample_positive_ids(edge_res_mode ? graph.edge_res : graph.node_cap,
                                       edge_res_mode ? graph.num_edges : graph.num_nodes,
                                       node_cap_mode,
                                       sample_count,
                                       seed);
    }

    std::vector<double> rows(static_cast<size_t>(work_ids.size()) * kFdColumns,
                             std::numeric_limits<double>::quiet_NaN());

    try {
        run_route_segment_dmp_from_graph(*this, graph);
        const double y0 = compute_late_endpoint_logsumexp_ns(*this, tau_ns);

        for (size_t row = 0; row < work_ids.size(); ++row) {
            const int64_t id = work_ids[row];
            double base_value = 0.0;
            if (edge_res_mode) {
                base_value = static_cast<double>(graph.edge_res[id]);
            } else {
                base_value = node_cap_base_value(graph, id);
            }

            const double eps = std::max(std::fabs(base_value) * eps_rel, eps_abs);
            const double rel_step = std::fabs(base_value) > 0.0
                                        ? eps / std::fabs(base_value)
                                        : std::numeric_limits<double>::infinity();

            double y1 = std::numeric_limits<double>::quiet_NaN();
            double delta_y = std::numeric_limits<double>::quiet_NaN();
            double fd_grad = std::numeric_limits<double>::quiet_NaN();
            double valid = 0.0;

            if (eps > 0.0 && std::isfinite(eps) && std::isfinite(y0)) {
                double y_plus = std::numeric_limits<double>::quiet_NaN();
                double y_minus = std::numeric_limits<double>::quiet_NaN();
                bool can_minus = false;
                if (edge_res_mode) {
                    const float original = graph.edge_res[id];
                    graph.edge_res[id] = static_cast<float>(static_cast<double>(original) + eps);
                    try {
                        run_route_segment_dmp_from_graph(*this, graph);
                        y_plus = compute_late_endpoint_logsumexp_ns(*this, tau_ns);
                    } catch (...) {
                        graph.edge_res[id] = original;
                        try_restore_baseline_state(*this, graph);
                        throw;
                    }
                    graph.edge_res[id] = original;

                    can_minus = static_cast<double>(original) - eps > 0.0;
                    if (can_minus) {
                        graph.edge_res[id] = static_cast<float>(static_cast<double>(original) - eps);
                        try {
                            run_route_segment_dmp_from_graph(*this, graph);
                            y_minus = compute_late_endpoint_logsumexp_ns(*this, tau_ns);
                        } catch (...) {
                            graph.edge_res[id] = original;
                            try_restore_baseline_state(*this, graph);
                            throw;
                        }
                        graph.edge_res[id] = original;
                    }
                } else {
                    float original[NUM_ATTR];
                    const size_t base = static_cast<size_t>(id) * NUM_ATTR;
                    can_minus = true;
                    for (int attr = 0; attr < NUM_ATTR; ++attr) {
                        original[attr] = graph.node_cap[base + attr];
                        can_minus = can_minus &&
                                    static_cast<double>(original[attr]) - eps >= 0.0;
                    }
                    perturb_node_cap(graph, id, eps);
                    try {
                        run_route_segment_dmp_from_graph(*this, graph);
                        y_plus = compute_late_endpoint_logsumexp_ns(*this, tau_ns);
                    } catch (...) {
                        restore_node_cap(graph, id, original);
                        try_restore_baseline_state(*this, graph);
                        throw;
                    }
                    restore_node_cap(graph, id, original);

                    if (can_minus) {
                        perturb_node_cap(graph, id, -eps);
                        try {
                            run_route_segment_dmp_from_graph(*this, graph);
                            y_minus = compute_late_endpoint_logsumexp_ns(*this, tau_ns);
                        } catch (...) {
                            restore_node_cap(graph, id, original);
                            try_restore_baseline_state(*this, graph);
                            throw;
                        }
                        restore_node_cap(graph, id, original);
                    }
                }

                y1 = y_plus;
                if (std::isfinite(y_plus) && std::isfinite(y_minus)) {
                    delta_y = y_plus - y_minus;
                    fd_grad = delta_y / (2.0 * eps);
                } else if (std::isfinite(y_plus)) {
                    delta_y = y_plus - y0;
                    fd_grad = delta_y / eps;
                }
                valid = std::isfinite(fd_grad) ? 1.0 : 0.0;
            }

            const size_t off = row * kFdColumns;
            rows[off + 0] = static_cast<double>(id);
            rows[off + 1] = base_value;
            rows[off + 2] = eps;
            rows[off + 3] = y0;
            rows[off + 4] = y1;
            rows[off + 5] = delta_y;
            rows[off + 6] = fd_grad;
            rows[off + 7] = rel_step;
            rows[off + 8] = valid;
        }

        run_route_segment_dmp_from_graph(*this, graph);
    } catch (...) {
        try_restore_baseline_state(*this, graph);
        throw;
    }

    auto tensor = torch::from_blob(rows.data(),
                                   {static_cast<long>(work_ids.size()), kFdColumns},
                                   torch::TensorOptions().dtype(torch::kFloat64))
                      .clone();
    return {tensor, fd_column_names()};
}

}  // namespace gt
