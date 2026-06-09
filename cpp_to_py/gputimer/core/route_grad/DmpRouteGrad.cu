#include "gputimer/core/route_grad/DmpRouteGrad.h"
#include "gputimer/core/route_grad/DmpRouteGradInternal.h"
#include "gputimer/core/route_grad/DmpRouteGradDevice.cuh"

#include "gputimer/core/GPUTimer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gt {

// Public route-gradient API, host orchestration, and timing reverse pass.
// This sidecar runs route-segment DMP timing, rebuilds compact local primitive
// slopes, seeds the soft endpoint objective, and walks timing adjoints backward
// before handing Elmore/root-PI adjoints to the RC-tree reverse file.
// Public route-gradient API and host orchestration.
// This sidecar runs route-segment DMP timing, seeds the soft endpoint objective,
// and calls the timing/RC reverse helpers without touching the normal timer path.
// Host-side orchestration and reverse pass helpers. Device kernels above produce
// local primitive Jacobian pieces; this block copies compact slope arrays back,
// seeds the soft endpoint objective, and performs the timing/RC reverse pass on
// the host for debug transparency.
void route_grad_cuda_check(cudaError_t err, const char* label)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in route segment gradient at ") +
                                 label + ": " + cudaGetErrorString(err));
    }
}

RouteGradDeviceFloatBuffer::RouteGradDeviceFloatBuffer() = default;

RouteGradDeviceFloatBuffer::~RouteGradDeviceFloatBuffer()
{
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void RouteGradDeviceFloatBuffer::allocate(size_t count, const char* label)
{
    if (count == 0) {
        return;
    }
    route_grad_cuda_check(cudaMalloc(&ptr, count * sizeof(float)), label);
}

RouteGradDeviceIntBuffer::RouteGradDeviceIntBuffer() = default;

RouteGradDeviceIntBuffer::~RouteGradDeviceIntBuffer()
{
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void RouteGradDeviceIntBuffer::allocate(size_t count, const char* label)
{
    if (count == 0) {
        return;
    }
    route_grad_cuda_check(cudaMalloc(&ptr, count * sizeof(int)), label);
}

RouteGradDeviceU64Buffer::RouteGradDeviceU64Buffer() = default;

RouteGradDeviceU64Buffer::~RouteGradDeviceU64Buffer()
{
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void RouteGradDeviceU64Buffer::allocate(size_t count, const char* label)
{
    if (count == 0) {
        return;
    }
    route_grad_cuda_check(cudaMalloc(&ptr, count * sizeof(unsigned long long)), label);
}


void release_route_grad_dmp_state(GPUTimer& timer)
{
    if (timer.h_dmp_db != nullptr) {
        timer.h_dmp_db->release_rc_transient();
        timer.h_dmp_db->release_after_timing();
        delete timer.h_dmp_db;
        timer.h_dmp_db = nullptr;
    }
    if (timer.dmp_db != nullptr) {
        route_grad_cuda_check(cudaFree(timer.dmp_db), "free dmp_db descriptor");
        timer.dmp_db = nullptr;
    }
}

void run_route_segment_dmp_for_route_grad(GPUTimer& timer, HostRcGraph& graph)
{
    release_route_grad_dmp_state(timer);
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
    route_grad_cuda_check(cudaDeviceSynchronize(), "route segment gradient baseline timing");
}


// Route-gradient sidecar overview. This file is intentionally kept outside the
// normal DMP timing/power forward path. The public debug entry points at the
// bottom load OpenROAD route segments, run DMP timing for that RC graph, rebuild
// local primitive slopes, reverse a soft endpoint objective through timing, and
// finally reverse the RC tree to per-segment R/C adjoints.
//
// High-level call flow:
//   compute_dmp_route_segment_soft_timing_grad()
//     -> run_route_segment_dmp_for_route_grad()
//     -> compute_net_primitive_slopes() / compute_active_gate_primitive_slopes()
//        / compute_gate_slew_winner_slopes()
//     -> init_late_logsumexp_at_adjoint()
//     -> reverse_endpoint_test_rat_to_related_at()
//     -> reverse_active_at_to_elmore()
//     -> reverse_one_net_rc_tree()
//     -> make_route_grad_tensors()
//

// Compute analytic gradients of the soft endpoint timing objective with respect
// to route-segment RC. The returned tensors are edge-resistance, node-cap, and
// derived edge-cap gradients.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
GPUTimer::compute_dmp_route_segment_soft_timing_grad(const std::string& route_segments_file,
                                         double tau_ns)
{
    if (route_segments_file.empty()) {
        throw std::runtime_error("route_segments_file is required for route segment gradient.");
    }

    HostRcGraph graph = build_openroad_route_segments_rc(route_segments_file);
    if (graph.node_cap.size() != static_cast<size_t>(graph.num_nodes) * NUM_ATTR) {
        throw std::runtime_error("Route-segment RC graph has invalid node_cap shape.");
    }
    if (graph.net2node_start.size() < static_cast<size_t>(num_nets + 1) ||
        graph.net2edge_start.size() < static_cast<size_t>(num_nets + 1)) {
        throw std::runtime_error("Route-segment RC graph has invalid net start arrays.");
    }

    run_route_segment_dmp_for_route_grad(*this, graph);
    const RouteGradNetSlopesHost net_slopes = compute_net_primitive_slopes(*this);
    const RouteGradActiveGateSlopesHost active_gate_slopes =
        compute_active_gate_primitive_slopes(*this);
    const RouteGradGateSlewWinnerSlopesHost gate_slew_slopes =
        compute_gate_slew_winner_slopes(*this);

    const size_t pin_slots = static_cast<size_t>(num_pins) * NUM_ATTR;
    const std::vector<float> pin_at = copy_device_array(pinAT, pin_slots, "pinAT");
    const std::vector<float> pin_rat = copy_device_array(pinRAT, pin_slots, "pinRAT");
    const std::vector<index_type> level_list_host = copy_level_list(*this);
    const std::vector<index_type> at_prefix_pin_host =
        copy_device_array(at_prefix_pin, pin_slots, "at_prefix_pin");
    const std::vector<index_type> at_prefix_arc_host =
        copy_device_array(at_prefix_arc, pin_slots, "at_prefix_arc");
    const std::vector<index_type> at_prefix_attr_host =
        copy_device_array(at_prefix_attr, pin_slots, "at_prefix_attr");
    const std::vector<uint8_t> arc_types_host =
        copy_device_array(arc_types, static_cast<size_t>(num_arcs), "arc_types");
    const std::vector<float> test_related_at_host =
        copy_device_array(testRelatedAT, static_cast<size_t>(std::max(num_tests, 0)) * NUM_ATTR, "testRelatedAT");
    const std::vector<float> test_rat_host =
        copy_device_array(testRAT, static_cast<size_t>(std::max(num_tests, 0)) * NUM_ATTR, "testRAT");
    const std::vector<index_type> pin_backward_arc_list_end_host =
        copy_device_array(pin_backward_arc_list_end, static_cast<size_t>(num_pins + 1), "pin_backward_arc_list_end");
    const std::vector<index_type> pin_backward_arc_list_host =
        copy_device_array(pin_backward_arc_list, static_cast<size_t>(timing_raw_db.pin_backward_arc_list.numel()), "pin_backward_arc_list");
    const std::vector<index_type> timing_arc_from_pin_id_host =
        copy_device_array(timing_arc_from_pin_id, static_cast<size_t>(num_arcs), "timing_arc_from_pin_id");
    const std::vector<index_type> timing_arc_to_pin_id_host =
        copy_device_array(timing_arc_to_pin_id, static_cast<size_t>(num_arcs), "timing_arc_to_pin_id");
    const std::vector<int> arc_id2test_id_host =
        copy_device_array(arc_id2test_id, static_cast<size_t>(num_arcs), "arc_id2test_id");

    std::vector<double> bar_pin_at(pin_slots, 0.0);
    std::vector<double> bar_pin_rat(pin_slots, 0.0);
    std::vector<double> bar_pin_slew(pin_slots, 0.0);
    std::vector<double> bar_elmore(pin_slots, 0.0);
    std::vector<double> bar_root_c1(pin_slots, 0.0);
    std::vector<double> bar_root_c2(pin_slots, 0.0);
    std::vector<double> bar_root_rpi(pin_slots, 0.0);
    init_late_logsumexp_at_adjoint(*this, pin_at, pin_rat, tau_ns, bar_pin_at, bar_pin_rat);
    reverse_endpoint_test_rat_to_related_at(*this,
                                            pin_at,
                                            pin_rat,
                                            test_related_at_host,
                                            test_rat_host,
                                            pin_backward_arc_list_end_host,
                                            pin_backward_arc_list_host,
                                            timing_arc_from_pin_id_host,
                                            timing_arc_to_pin_id_host,
                                            arc_id2test_id_host,
                                            bar_pin_at,
                                            bar_pin_rat);
    reverse_active_at_to_elmore(*this,
                                level_list_host,
                                at_prefix_pin_host,
                                at_prefix_arc_host,
                                at_prefix_attr_host,
                                arc_types_host,
                                net_slopes,
                                active_gate_slopes,
                                gate_slew_slopes,
                                bar_pin_at,
                                bar_pin_slew,
                                bar_elmore,
                                bar_root_c1,
                                bar_root_c2,
                                bar_root_rpi);

    std::vector<double> edge_res_grad(static_cast<size_t>(graph.num_edges), 0.0);
    std::vector<double> node_cap_grad(static_cast<size_t>(graph.num_nodes), 0.0);
    const double rc_time_factor =
        (static_cast<double>(res_unit) * static_cast<double>(cap_unit)) /
        static_cast<double>(time_unit());

    for (int net = 0; net < num_nets; ++net) {
        reverse_one_net_rc_tree(*this,
                                graph,
                                net,
                                rc_time_factor,
                                bar_elmore,
                                bar_root_c1,
                                bar_root_c2,
                                bar_root_rpi,
                                edge_res_grad,
                                node_cap_grad);
    }

    return make_route_grad_tensors(graph, edge_res_grad, node_cap_grad);
}

std::vector<std::string> dmp_route_segment_primitive_slope_stat_columns()
{
    return {"net_delay_analytic",
            "net_delay_fd",
            "net_delay_fail",
            "net_slew_analytic",
            "net_slew_fd",
            "net_slew_fail",
            "active_gate_analytic",
            "active_gate_fd",
            "active_gate_fail",
            "gate_slew_analytic",
            "gate_slew_fd",
            "gate_slew_fail",
            "net_delay_fd_cap",
            "net_delay_fd_zero_c2",
            "net_delay_fd_pi",
            "net_delay_fd_other",
            "net_slew_fd_cap",
            "net_slew_fd_zero_c2",
            "net_slew_fd_pi",
            "net_slew_fd_other",
            "active_gate_fd_cap",
            "active_gate_fd_zero_c2",
            "active_gate_fd_pi",
            "active_gate_fd_other",
            "gate_slew_fd_cap",
            "gate_slew_fd_zero_c2",
            "gate_slew_fd_pi",
            "gate_slew_fd_other",
            "pi_fail_coeff",
            "pi_fail_implicit_setup",
            "pi_fail_implicit_lut",
            "pi_fail_implicit_equation",
            "pi_fail_implicit_solve",
            "pi_fail_implicit_output",
            "pi_fail_rd",
            "pi_fail_init",
            "pi_fail_forward_solve",
            "pi_fail_wave_slope",
            "pi_fail_extra_lut",
            "pi_recovered_ceff_from_delay"};
}

// Public debug API: count analytic/fail coverage for the local primitive slope
// preparation kernels without returning full gradient tensors.
std::tuple<torch::Tensor, std::vector<std::string>>
GPUTimer::debug_dmp_route_segment_primitive_slope_stats(const std::string& route_segments_file)
{
    if (route_segments_file.empty()) {
        throw std::runtime_error("route_segments_file is required for route segment primitive slope stats.");
    }

    HostRcGraph graph = build_openroad_route_segments_rc(route_segments_file);
    if (graph.node_cap.size() != static_cast<size_t>(graph.num_nodes) * NUM_ATTR) {
        throw std::runtime_error("Route-segment RC graph has invalid node_cap shape.");
    }
    if (graph.net2node_start.size() < static_cast<size_t>(num_nets + 1) ||
        graph.net2edge_start.size() < static_cast<size_t>(num_nets + 1)) {
        throw std::runtime_error("Route-segment RC graph has invalid net start arrays.");
    }

    run_route_segment_dmp_for_route_grad(*this, graph);
    RouteGradDeviceU64Buffer d_stats;
    d_stats.allocate(kRouteGradPrimitiveStatCount, "allocate route_grad primitive stats");
    route_grad_cuda_check(cudaMemset(d_stats.ptr,
                                     0,
                                     kRouteGradPrimitiveStatCount * sizeof(unsigned long long)),
                          "clear route_grad primitive stats");
    (void)compute_net_primitive_slopes(*this, d_stats.ptr);
    (void)compute_active_gate_primitive_slopes(*this, d_stats.ptr);
    (void)compute_gate_slew_winner_slopes(*this, d_stats.ptr);

    std::vector<unsigned long long> stats_u64(kRouteGradPrimitiveStatCount, 0ULL);
    route_grad_cuda_check(cudaMemcpy(stats_u64.data(),
                                     d_stats.ptr,
                                     kRouteGradPrimitiveStatCount * sizeof(unsigned long long),
                                     cudaMemcpyDeviceToHost),
                          "copy route_grad primitive stats");
    std::vector<int64_t> stats(stats_u64.size(), 0);
    for (size_t i = 0; i < stats_u64.size(); ++i) {
        stats[i] = static_cast<int64_t>(stats_u64[i]);
    }
    auto tensor = torch::from_blob(stats.data(),
                                   {static_cast<long>(stats.size())},
                                   torch::TensorOptions().dtype(torch::kInt64))
                      .clone();
    return {tensor, dmp_route_segment_primitive_slope_stat_columns()};
}

// Public debug API: fixed-topology finite-difference check for the RC-tree
// reverse block only. This intentionally does not test global timing winner
// stability.
std::tuple<torch::Tensor, std::vector<std::string>>
GPUTimer::debug_dmp_route_segment_rc_tree_gradcheck(const std::string& route_segments_file,
                                                    int sample_net_count,
                                                    int seed,
                                                    double eps_rel,
                                                    double eps_abs_edge,
                                                    double eps_abs_node)
{
    if (route_segments_file.empty()) {
        throw std::runtime_error("route_segments_file is required for RC-tree gradcheck.");
    }
    if (sample_net_count <= 0) {
        throw std::runtime_error("sample_net_count must be positive for RC-tree gradcheck.");
    }
    if (eps_rel < 0.0 || eps_abs_edge < 0.0 || eps_abs_node < 0.0 ||
        !std::isfinite(eps_rel) || !std::isfinite(eps_abs_edge) ||
        !std::isfinite(eps_abs_node)) {
        throw std::runtime_error("RC-tree gradcheck eps values must be non-negative and finite.");
    }

    HostRcGraph graph = build_openroad_route_segments_rc(route_segments_file);
    if (graph.node_cap.size() != static_cast<size_t>(graph.num_nodes) * NUM_ATTR) {
        throw std::runtime_error("Route-segment RC graph has invalid node_cap shape.");
    }
    if (graph.net2node_start.size() < static_cast<size_t>(num_nets + 1) ||
        graph.net2edge_start.size() < static_cast<size_t>(num_nets + 1)) {
        throw std::runtime_error("Route-segment RC graph has invalid net start arrays.");
    }

    const std::vector<RouteGradRcTreeCheckSample> samples =
        sample_rc_tree_check_nets(graph, sample_net_count, seed);
    const size_t pin_slots = static_cast<size_t>(num_pins) * NUM_ATTR;
    std::vector<double> bar_elmore(pin_slots, 0.0);
    std::vector<double> bar_root_c1(pin_slots, 0.0);
    std::vector<double> bar_root_c2(pin_slots, 0.0);
    std::vector<double> bar_root_rpi(pin_slots, 0.0);
    seed_rc_tree_check_adjoint(*this,
                               graph,
                               samples,
                               bar_elmore,
                               bar_root_c1,
                               bar_root_c2,
                               bar_root_rpi);

    std::vector<double> edge_res_grad(static_cast<size_t>(graph.num_edges), 0.0);
    std::vector<double> node_cap_grad(static_cast<size_t>(graph.num_nodes), 0.0);
    const double rc_time_factor =
        (static_cast<double>(res_unit) * static_cast<double>(cap_unit)) /
        static_cast<double>(time_unit());
    for (const RouteGradRcTreeCheckSample& sample : samples) {
        reverse_one_net_rc_tree(*this,
                                graph,
                                sample.net_id,
                                rc_time_factor,
                                bar_elmore,
                                bar_root_c1,
                                bar_root_c2,
                                bar_root_rpi,
                                edge_res_grad,
                                node_cap_grad);
    }

    std::vector<double> rows(samples.size() * kRouteGradRcTreeCheckColumns,
                             std::numeric_limits<double>::quiet_NaN());
    for (size_t row = 0; row < samples.size(); ++row) {
        const RouteGradRcTreeCheckSample& sample = samples[row];
        const double base = forward_one_net_rc_tree_value(*this,
                                                          graph,
                                                          sample.net_id,
                                                          rc_time_factor,
                                                          bar_elmore,
                                                          bar_root_c1,
                                                          bar_root_c2,
                                                          bar_root_rpi);

        double edge_fd = std::numeric_limits<double>::quiet_NaN();
        double valid_edge = 0.0;
        if (sample.edge_id >= 0 && sample.edge_id < graph.num_edges) {
            const float original = graph.edge_res[sample.edge_id];
            const double eps = std::max(std::fabs(static_cast<double>(original)) * eps_rel,
                                        eps_abs_edge);
            if (eps > 0.0 && std::isfinite(eps)) {
                graph.edge_res[sample.edge_id] = static_cast<float>(static_cast<double>(original) + eps);
                const double plus = forward_one_net_rc_tree_value(*this,
                                                                  graph,
                                                                  sample.net_id,
                                                                  rc_time_factor,
                                                                  bar_elmore,
                                                                  bar_root_c1,
                                                                  bar_root_c2,
                                                                  bar_root_rpi);
                bool has_minus = static_cast<double>(original) - eps > 0.0;
                double minus = std::numeric_limits<double>::quiet_NaN();
                if (has_minus) {
                    graph.edge_res[sample.edge_id] = static_cast<float>(static_cast<double>(original) - eps);
                    minus = forward_one_net_rc_tree_value(*this,
                                                          graph,
                                                          sample.net_id,
                                                          rc_time_factor,
                                                          bar_elmore,
                                                          bar_root_c1,
                                                          bar_root_c2,
                                                          bar_root_rpi);
                }
                graph.edge_res[sample.edge_id] = original;
                if (std::isfinite(plus) && std::isfinite(minus)) {
                    edge_fd = (plus - minus) / (2.0 * eps);
                } else if (std::isfinite(plus) && std::isfinite(base)) {
                    edge_fd = (plus - base) / eps;
                }
                valid_edge = std::isfinite(edge_fd) ? 1.0 : 0.0;
            }
        }

        double node_fd = std::numeric_limits<double>::quiet_NaN();
        double valid_node = 0.0;
        if (sample.node_id >= 0 && sample.node_id < graph.num_nodes) {
            float original[NUM_ATTR];
            const size_t off = static_cast<size_t>(sample.node_id) * NUM_ATTR;
            bool can_minus = true;
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                original[attr] = graph.node_cap[off + attr];
            }
            const double cap_base = route_grad_rc_tree_node_cap_base(graph, sample.node_id);
            const double eps = std::max(cap_base * eps_rel, eps_abs_node);
            if (eps > 0.0 && std::isfinite(eps)) {
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    graph.node_cap[off + attr] = static_cast<float>(static_cast<double>(original[attr]) + eps);
                    can_minus = can_minus && static_cast<double>(original[attr]) - eps >= 0.0;
                }
                const double plus = forward_one_net_rc_tree_value(*this,
                                                                  graph,
                                                                  sample.net_id,
                                                                  rc_time_factor,
                                                                  bar_elmore,
                                                                  bar_root_c1,
                                                                  bar_root_c2,
                                                                  bar_root_rpi);
                double minus = std::numeric_limits<double>::quiet_NaN();
                if (can_minus) {
                    for (int attr = 0; attr < NUM_ATTR; ++attr) {
                        graph.node_cap[off + attr] = static_cast<float>(static_cast<double>(original[attr]) - eps);
                    }
                    minus = forward_one_net_rc_tree_value(*this,
                                                          graph,
                                                          sample.net_id,
                                                          rc_time_factor,
                                                          bar_elmore,
                                                          bar_root_c1,
                                                          bar_root_c2,
                                                          bar_root_rpi);
                }
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    graph.node_cap[off + attr] = original[attr];
                }
                if (std::isfinite(plus) && std::isfinite(minus)) {
                    node_fd = (plus - minus) / (2.0 * eps);
                } else if (std::isfinite(plus) && std::isfinite(base)) {
                    node_fd = (plus - base) / eps;
                }
                valid_node = std::isfinite(node_fd) ? 1.0 : 0.0;
            }
        }

        const double edge_adj = (sample.edge_id >= 0 && sample.edge_id < graph.num_edges)
                                    ? edge_res_grad[sample.edge_id]
                                    : std::numeric_limits<double>::quiet_NaN();
        const double node_adj = (sample.node_id >= 0 && sample.node_id < graph.num_nodes)
                                    ? node_cap_grad[sample.node_id]
                                    : std::numeric_limits<double>::quiet_NaN();
        const size_t off = row * kRouteGradRcTreeCheckColumns;
        rows[off + 0] = static_cast<double>(sample.net_id);
        rows[off + 1] = static_cast<double>(sample.edge_id);
        rows[off + 2] = static_cast<double>(sample.node_id);
        rows[off + 3] = edge_fd;
        rows[off + 4] = edge_adj;
        rows[off + 5] = std::fabs(edge_adj - edge_fd);
        rows[off + 6] = route_grad_rc_tree_safe_rel_err(edge_fd, edge_adj);
        rows[off + 7] = node_fd;
        rows[off + 8] = node_adj;
        rows[off + 9] = std::fabs(node_adj - node_fd);
        rows[off + 10] = route_grad_rc_tree_safe_rel_err(node_fd, node_adj);
        rows[off + 11] = valid_edge;
        rows[off + 12] = valid_node;
    }

    auto tensor = torch::from_blob(rows.data(),
                                   {static_cast<long>(samples.size()), kRouteGradRcTreeCheckColumns},
                                   torch::TensorOptions().dtype(torch::kFloat64))
                      .clone();
    return {tensor, dmp_route_segment_rc_tree_gradcheck_columns()};
}


// Timing-side derivative preparation and reverse pass.
// Device kernels first emit compact local primitive slopes; the host then walks
// the timing graph backward from the soft endpoint objective.
RouteGradNetSlopesHost compute_net_primitive_slopes(GPUTimer& timer,
                                               unsigned long long* primitive_stats = nullptr)
{
    const size_t pin_slots = static_cast<size_t>(timer.num_pins) * NUM_ATTR;
    RouteGradNetSlopesHost host;
    host.delay_elmore_slope.assign(pin_slots, 0.0f);
    host.slew_elmore_slope.assign(pin_slots, 0.0f);
    host.delay_driver_root_slot.assign(pin_slots, -1);
    host.delay_driver_input_slew_slot.assign(pin_slots, -1);
    host.slew_driver_root_slot.assign(pin_slots, -1);
    host.slew_driver_input_slew_slot.assign(pin_slots, -1);
    host.delay_c1_slope.assign(pin_slots, 0.0f);
    host.delay_c2_slope.assign(pin_slots, 0.0f);
    host.delay_rpi_slope.assign(pin_slots, 0.0f);
    host.delay_input_slew_slope.assign(pin_slots, 0.0f);
    host.slew_c1_slope.assign(pin_slots, 0.0f);
    host.slew_c2_slope.assign(pin_slots, 0.0f);
    host.slew_rpi_slope.assign(pin_slots, 0.0f);
    host.slew_input_slew_slope.assign(pin_slots, 0.0f);
    if (pin_slots == 0 || timer.num_arcs <= 0) {
        return host;
    }
    if (timer.dmp_db == nullptr) {
        throw std::runtime_error("DMP model is not initialized for route segment net slopes.");
    }

    RouteGradDeviceFloatBuffer d_delay_elmore_slope;
    RouteGradDeviceFloatBuffer d_slew_elmore_slope;
    RouteGradDeviceIntBuffer d_delay_driver_root_slot;
    RouteGradDeviceIntBuffer d_delay_driver_input_slew_slot;
    RouteGradDeviceIntBuffer d_slew_driver_root_slot;
    RouteGradDeviceIntBuffer d_slew_driver_input_slew_slot;
    RouteGradDeviceFloatBuffer d_delay_c1_slope;
    RouteGradDeviceFloatBuffer d_delay_c2_slope;
    RouteGradDeviceFloatBuffer d_delay_rpi_slope;
    RouteGradDeviceFloatBuffer d_delay_input_slew_slope;
    RouteGradDeviceFloatBuffer d_slew_c1_slope;
    RouteGradDeviceFloatBuffer d_slew_c2_slope;
    RouteGradDeviceFloatBuffer d_slew_rpi_slope;
    RouteGradDeviceFloatBuffer d_slew_input_slew_slope;
    d_delay_elmore_slope.allocate(pin_slots, "allocate net delay Elmore slopes");
    d_slew_elmore_slope.allocate(pin_slots, "allocate net slew Elmore slopes");
    d_delay_driver_root_slot.allocate(pin_slots, "allocate net delay driver root slots");
    d_delay_driver_input_slew_slot.allocate(pin_slots, "allocate net delay driver input-slew slots");
    d_slew_driver_root_slot.allocate(pin_slots, "allocate net slew driver root slots");
    d_slew_driver_input_slew_slot.allocate(pin_slots, "allocate net slew driver input-slew slots");
    d_delay_c1_slope.allocate(pin_slots, "allocate net delay C1 slopes");
    d_delay_c2_slope.allocate(pin_slots, "allocate net delay C2 slopes");
    d_delay_rpi_slope.allocate(pin_slots, "allocate net delay rpi slopes");
    d_delay_input_slew_slope.allocate(pin_slots, "allocate net delay input-slew slopes");
    d_slew_c1_slope.allocate(pin_slots, "allocate net slew C1 slopes");
    d_slew_c2_slope.allocate(pin_slots, "allocate net slew C2 slopes");
    d_slew_rpi_slope.allocate(pin_slots, "allocate net slew rpi slopes");
    d_slew_input_slew_slope.allocate(pin_slots, "allocate net slew input-slew slopes");
    route_grad_cuda_check(cudaMemset(d_delay_elmore_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net delay Elmore slopes");
    route_grad_cuda_check(cudaMemset(d_slew_elmore_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net slew Elmore slopes");
    route_grad_cuda_check(cudaMemset(d_delay_driver_root_slot.ptr, 0xff, pin_slots * sizeof(int)),
                          "clear net delay driver root slots");
    route_grad_cuda_check(cudaMemset(d_delay_driver_input_slew_slot.ptr, 0xff, pin_slots * sizeof(int)),
                          "clear net delay driver input-slew slots");
    route_grad_cuda_check(cudaMemset(d_slew_driver_root_slot.ptr, 0xff, pin_slots * sizeof(int)),
                          "clear net slew driver root slots");
    route_grad_cuda_check(cudaMemset(d_slew_driver_input_slew_slot.ptr, 0xff, pin_slots * sizeof(int)),
                          "clear net slew driver input-slew slots");
    route_grad_cuda_check(cudaMemset(d_delay_c1_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net delay C1 slopes");
    route_grad_cuda_check(cudaMemset(d_delay_c2_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net delay C2 slopes");
    route_grad_cuda_check(cudaMemset(d_delay_rpi_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net delay rpi slopes");
    route_grad_cuda_check(cudaMemset(d_delay_input_slew_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net delay input-slew slopes");
    route_grad_cuda_check(cudaMemset(d_slew_c1_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net slew C1 slopes");
    route_grad_cuda_check(cudaMemset(d_slew_c2_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net slew C2 slopes");
    route_grad_cuda_check(cudaMemset(d_slew_rpi_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net slew rpi slopes");
    route_grad_cuda_check(cudaMemset(d_slew_input_slew_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear net slew input-slew slopes");

    RouteGradNetPrimitiveReverse op;
    op.model = timer.dmp_db;
    op.delay_elmore_slope = d_delay_elmore_slope.ptr;
    op.slew_elmore_slope = d_slew_elmore_slope.ptr;
    op.delay_driver_root_slot = d_delay_driver_root_slot.ptr;
    op.delay_driver_input_slew_slot = d_delay_driver_input_slew_slot.ptr;
    op.slew_driver_root_slot = d_slew_driver_root_slot.ptr;
    op.slew_driver_input_slew_slot = d_slew_driver_input_slew_slot.ptr;
    op.delay_c1_slope = d_delay_c1_slope.ptr;
    op.delay_c2_slope = d_delay_c2_slope.ptr;
    op.delay_rpi_slope = d_delay_rpi_slope.ptr;
    op.delay_input_slew_slope = d_delay_input_slew_slope.ptr;
    op.slew_c1_slope = d_slew_c1_slope.ptr;
    op.slew_c2_slope = d_slew_c2_slope.ptr;
    op.slew_rpi_slope = d_slew_rpi_slope.ptr;
    op.slew_input_slew_slope = d_slew_input_slew_slope.ptr;
    op.primitive_stats = primitive_stats;
    const int work_items = timer.num_arcs * NUM_ATTR;
    routeGradNetElmoreSlopeKernel<<<DMP_TIMING_BLOCK_NUMBER(work_items),
                                    DMP_TIMING_BLOCK_SIZE>>>(op);
    route_grad_cuda_check(cudaGetLastError(), "launch net primitive slope kernel");

    route_grad_cuda_check(cudaMemcpy(host.delay_elmore_slope.data(),
                                     d_delay_elmore_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net delay Elmore slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_elmore_slope.data(),
                                     d_slew_elmore_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net slew Elmore slopes");
    route_grad_cuda_check(cudaMemcpy(host.delay_driver_root_slot.data(),
                                     d_delay_driver_root_slot.ptr,
                                     pin_slots * sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "copy net delay driver root slots");
    route_grad_cuda_check(cudaMemcpy(host.delay_driver_input_slew_slot.data(),
                                     d_delay_driver_input_slew_slot.ptr,
                                     pin_slots * sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "copy net delay driver input-slew slots");
    route_grad_cuda_check(cudaMemcpy(host.slew_driver_root_slot.data(),
                                     d_slew_driver_root_slot.ptr,
                                     pin_slots * sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "copy net slew driver root slots");
    route_grad_cuda_check(cudaMemcpy(host.slew_driver_input_slew_slot.data(),
                                     d_slew_driver_input_slew_slot.ptr,
                                     pin_slots * sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "copy net slew driver input-slew slots");
    route_grad_cuda_check(cudaMemcpy(host.delay_c1_slope.data(),
                                     d_delay_c1_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net delay C1 slopes");
    route_grad_cuda_check(cudaMemcpy(host.delay_c2_slope.data(),
                                     d_delay_c2_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net delay C2 slopes");
    route_grad_cuda_check(cudaMemcpy(host.delay_rpi_slope.data(),
                                     d_delay_rpi_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net delay rpi slopes");
    route_grad_cuda_check(cudaMemcpy(host.delay_input_slew_slope.data(),
                                     d_delay_input_slew_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net delay input-slew slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_c1_slope.data(),
                                     d_slew_c1_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net slew C1 slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_c2_slope.data(),
                                     d_slew_c2_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net slew C2 slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_rpi_slope.data(),
                                     d_slew_rpi_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net slew rpi slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_input_slew_slope.data(),
                                     d_slew_input_slew_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy net slew input-slew slopes");
    return host;
}

RouteGradActiveGateSlopesHost compute_active_gate_primitive_slopes(GPUTimer& timer,
                                                             unsigned long long* primitive_stats = nullptr)
{
    const size_t pin_slots = static_cast<size_t>(timer.num_pins) * NUM_ATTR;
    RouteGradActiveGateSlopesHost host;
    host.root_slot.assign(pin_slots, -1);
    host.delay_c1_slope.assign(pin_slots, 0.0f);
    host.delay_c2_slope.assign(pin_slots, 0.0f);
    host.delay_rpi_slope.assign(pin_slots, 0.0f);
    host.delay_input_slew_slope.assign(pin_slots, 0.0f);
    host.slew_c1_slope.assign(pin_slots, 0.0f);
    host.slew_c2_slope.assign(pin_slots, 0.0f);
    host.slew_rpi_slope.assign(pin_slots, 0.0f);
    host.slew_input_slew_slope.assign(pin_slots, 0.0f);
    if (pin_slots == 0 || timer.dmp_db == nullptr) {
        return host;
    }

    RouteGradDeviceIntBuffer d_root_slot;
    RouteGradDeviceFloatBuffer d_delay_c1_slope;
    RouteGradDeviceFloatBuffer d_delay_c2_slope;
    RouteGradDeviceFloatBuffer d_delay_rpi_slope;
    RouteGradDeviceFloatBuffer d_delay_input_slew_slope;
    RouteGradDeviceFloatBuffer d_slew_c1_slope;
    RouteGradDeviceFloatBuffer d_slew_c2_slope;
    RouteGradDeviceFloatBuffer d_slew_rpi_slope;
    RouteGradDeviceFloatBuffer d_slew_input_slew_slope;
    d_root_slot.allocate(pin_slots, "allocate active gate slope slots");
    d_delay_c1_slope.allocate(pin_slots, "allocate active gate delay C1 slopes");
    d_delay_c2_slope.allocate(pin_slots, "allocate active gate delay C2 slopes");
    d_delay_rpi_slope.allocate(pin_slots, "allocate active gate delay rpi slopes");
    d_delay_input_slew_slope.allocate(pin_slots, "allocate active gate delay input-slew slopes");
    d_slew_c1_slope.allocate(pin_slots, "allocate active gate slew C1 slopes");
    d_slew_c2_slope.allocate(pin_slots, "allocate active gate slew C2 slopes");
    d_slew_rpi_slope.allocate(pin_slots, "allocate active gate slew rpi slopes");
    d_slew_input_slew_slope.allocate(pin_slots, "allocate active gate slew input-slew slopes");
    route_grad_cuda_check(cudaMemset(d_root_slot.ptr, 0xff, pin_slots * sizeof(int)),
                          "clear active gate slope slots");
    route_grad_cuda_check(cudaMemset(d_delay_c1_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear active gate delay C1 slopes");
    route_grad_cuda_check(cudaMemset(d_delay_c2_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear active gate delay C2 slopes");
    route_grad_cuda_check(cudaMemset(d_delay_rpi_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear active gate delay rpi slopes");
    route_grad_cuda_check(cudaMemset(d_delay_input_slew_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear active gate delay input-slew slopes");
    route_grad_cuda_check(cudaMemset(d_slew_c1_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear active gate slew C1 slopes");
    route_grad_cuda_check(cudaMemset(d_slew_c2_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear active gate slew C2 slopes");
    route_grad_cuda_check(cudaMemset(d_slew_rpi_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear active gate slew rpi slopes");
    route_grad_cuda_check(cudaMemset(d_slew_input_slew_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear active gate slew input-slew slopes");

    RouteGradActiveGatePrimitiveSlope op;
    op.model = timer.dmp_db;
    op.root_slot = d_root_slot.ptr;
    op.delay_c1_slope = d_delay_c1_slope.ptr;
    op.delay_c2_slope = d_delay_c2_slope.ptr;
    op.delay_rpi_slope = d_delay_rpi_slope.ptr;
    op.delay_input_slew_slope = d_delay_input_slew_slope.ptr;
    op.slew_c1_slope = d_slew_c1_slope.ptr;
    op.slew_c2_slope = d_slew_c2_slope.ptr;
    op.slew_rpi_slope = d_slew_rpi_slope.ptr;
    op.slew_input_slew_slope = d_slew_input_slew_slope.ptr;
    op.primitive_stats = primitive_stats;
    const int work_items = static_cast<int>(pin_slots);
    routeGradActiveGatePrimitiveSlopeKernel<<<DMP_TIMING_BLOCK_NUMBER(work_items),
                                              DMP_TIMING_BLOCK_SIZE>>>(op);
    route_grad_cuda_check(cudaGetLastError(), "launch active gate primitive slope kernel");
    route_grad_cuda_check(cudaMemcpy(host.root_slot.data(),
                                     d_root_slot.ptr,
                                     pin_slots * sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "copy active gate slope slots");
    route_grad_cuda_check(cudaMemcpy(host.delay_c1_slope.data(),
                                     d_delay_c1_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy active gate delay C1 slopes");
    route_grad_cuda_check(cudaMemcpy(host.delay_c2_slope.data(),
                                     d_delay_c2_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy active gate delay C2 slopes");
    route_grad_cuda_check(cudaMemcpy(host.delay_rpi_slope.data(),
                                     d_delay_rpi_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy active gate delay rpi slopes");
    route_grad_cuda_check(cudaMemcpy(host.delay_input_slew_slope.data(),
                                     d_delay_input_slew_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy active gate delay input-slew slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_c1_slope.data(),
                                     d_slew_c1_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy active gate slew C1 slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_c2_slope.data(),
                                     d_slew_c2_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy active gate slew C2 slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_rpi_slope.data(),
                                     d_slew_rpi_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy active gate slew rpi slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_input_slew_slope.data(),
                                     d_slew_input_slew_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy active gate slew input-slew slopes");
    return host;
}

RouteGradGateSlewWinnerSlopesHost compute_gate_slew_winner_slopes(GPUTimer& timer,
                                                           unsigned long long* primitive_stats = nullptr)
{
    const size_t pin_slots = static_cast<size_t>(timer.num_pins) * NUM_ATTR;
    RouteGradGateSlewWinnerSlopesHost host;
    host.root_slot.assign(pin_slots, -1);
    host.input_slew_slot.assign(pin_slots, -1);
    host.slew_c1_slope.assign(pin_slots, 0.0f);
    host.slew_c2_slope.assign(pin_slots, 0.0f);
    host.slew_rpi_slope.assign(pin_slots, 0.0f);
    host.slew_input_slew_slope.assign(pin_slots, 0.0f);
    if (pin_slots == 0 || timer.dmp_db == nullptr) {
        return host;
    }

    RouteGradDeviceIntBuffer d_root_slot;
    RouteGradDeviceIntBuffer d_input_slew_slot;
    RouteGradDeviceFloatBuffer d_slew_c1_slope;
    RouteGradDeviceFloatBuffer d_slew_c2_slope;
    RouteGradDeviceFloatBuffer d_slew_rpi_slope;
    RouteGradDeviceFloatBuffer d_slew_input_slew_slope;
    d_root_slot.allocate(pin_slots, "allocate gate slew-winner root slots");
    d_input_slew_slot.allocate(pin_slots, "allocate gate slew-winner input-slew slots");
    d_slew_c1_slope.allocate(pin_slots, "allocate gate slew-winner C1 slopes");
    d_slew_c2_slope.allocate(pin_slots, "allocate gate slew-winner C2 slopes");
    d_slew_rpi_slope.allocate(pin_slots, "allocate gate slew-winner rpi slopes");
    d_slew_input_slew_slope.allocate(pin_slots, "allocate gate slew-winner input-slew slopes");
    route_grad_cuda_check(cudaMemset(d_root_slot.ptr, 0xff, pin_slots * sizeof(int)),
                          "clear gate slew-winner root slots");
    route_grad_cuda_check(cudaMemset(d_input_slew_slot.ptr, 0xff, pin_slots * sizeof(int)),
                          "clear gate slew-winner input-slew slots");
    route_grad_cuda_check(cudaMemset(d_slew_c1_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear gate slew-winner C1 slopes");
    route_grad_cuda_check(cudaMemset(d_slew_c2_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear gate slew-winner C2 slopes");
    route_grad_cuda_check(cudaMemset(d_slew_rpi_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear gate slew-winner rpi slopes");
    route_grad_cuda_check(cudaMemset(d_slew_input_slew_slope.ptr, 0, pin_slots * sizeof(float)),
                          "clear gate slew-winner input-slew slopes");

    RouteGradActiveGateSlewWinnerSlope op;
    op.model = timer.dmp_db;
    op.root_slot = d_root_slot.ptr;
    op.input_slew_slot = d_input_slew_slot.ptr;
    op.slew_c1_slope = d_slew_c1_slope.ptr;
    op.slew_c2_slope = d_slew_c2_slope.ptr;
    op.slew_rpi_slope = d_slew_rpi_slope.ptr;
    op.slew_input_slew_slope = d_slew_input_slew_slope.ptr;
    op.primitive_stats = primitive_stats;
    const int work_items = static_cast<int>(pin_slots);
    routeGradActiveGateSlewWinnerSlopeKernel<<<DMP_TIMING_BLOCK_NUMBER(work_items),
                                               DMP_TIMING_BLOCK_SIZE>>>(op);
    route_grad_cuda_check(cudaGetLastError(), "launch gate slew-winner slope kernel");
    route_grad_cuda_check(cudaMemcpy(host.root_slot.data(),
                                     d_root_slot.ptr,
                                     pin_slots * sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "copy gate slew-winner root slots");
    route_grad_cuda_check(cudaMemcpy(host.input_slew_slot.data(),
                                     d_input_slew_slot.ptr,
                                     pin_slots * sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "copy gate slew-winner input-slew slots");
    route_grad_cuda_check(cudaMemcpy(host.slew_c1_slope.data(),
                                     d_slew_c1_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy gate slew-winner C1 slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_c2_slope.data(),
                                     d_slew_c2_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy gate slew-winner C2 slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_rpi_slope.data(),
                                     d_slew_rpi_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy gate slew-winner rpi slopes");
    route_grad_cuda_check(cudaMemcpy(host.slew_input_slew_slope.data(),
                                     d_slew_input_slew_slope.ptr,
                                     pin_slots * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy gate slew-winner input-slew slopes");
    return host;
}


std::vector<index_type> copy_level_list(const GPUTimer& timer)
{
    if (timer.level_list_end_cpu.empty()) {
        return {};
    }
    const int count = timer.level_list_end_cpu.back();
    return copy_device_array(timer.level_list, static_cast<size_t>(std::max(count, 0)), "level_list");
}

std::vector<int64_t> endpoint_ids_cpu(const GPUTimer& timer)
{
    torch::Tensor endpoint_ids;
    if (timer.timing_raw_db.endpoint_unique_pin_ids.defined() &&
        timer.timing_raw_db.endpoint_unique_pin_ids.numel() > 0) {
        endpoint_ids = timer.timing_raw_db.endpoint_unique_pin_ids;
    } else if (timer.timing_raw_db.endpoints_id.defined() &&
               timer.timing_raw_db.endpoints_id.numel() > 0) {
        endpoint_ids = std::get<0>(torch::_unique(timer.timing_raw_db.endpoints_id));
    } else {
        throw std::runtime_error("No endpoint pin ids are available for route segment gradient.");
    }
    endpoint_ids = endpoint_ids.to(torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                       .contiguous();
    const auto* data = endpoint_ids.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + endpoint_ids.numel());
}

void init_late_logsumexp_at_adjoint(const GPUTimer& timer,
                                    const std::vector<float>& pin_at,
                                    const std::vector<float>& pin_rat,
                                    double tau_ns,
                                    std::vector<double>& bar_pin_at,
                                    std::vector<double>& bar_pin_rat)
{
    if (!(tau_ns > 0.0) || !std::isfinite(tau_ns)) {
        throw std::runtime_error("tau_ns must be positive and finite.");
    }
    const std::vector<int64_t> endpoints = endpoint_ids_cpu(timer);
    const double time_to_ns = static_cast<double>(timer.time_unit()) * 1.0e9;

    std::vector<RouteGradEndpointValue> values;
    values.reserve(endpoints.size() * 2);
    double max_value = -std::numeric_limits<double>::infinity();
    for (int64_t pin64 : endpoints) {
        if (pin64 < 0 || pin64 >= timer.num_pins) {
            continue;
        }
        const int pin = static_cast<int>(pin64);
        for (int attr = 2; attr < NUM_ATTR; ++attr) {
            const int slot = pin * NUM_ATTR + attr;
            const double at = static_cast<double>(pin_at[slot]);
            const double rat = static_cast<double>(pin_rat[slot]);
            const double value = (at - rat) * time_to_ns;
            if (!std::isfinite(value) || value <= -1.0e20 || value >= 1.0e20) {
                continue;
            }
            values.push_back({slot, value});
            max_value = std::max(max_value, value);
        }
    }
    if (values.empty() || !std::isfinite(max_value)) {
        throw std::runtime_error("No finite late endpoint values are available for route segment gradient.");
    }

    double sum_exp = 0.0;
    for (const RouteGradEndpointValue& entry : values) {
        sum_exp += std::exp((entry.value_ns - max_value) / tau_ns);
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
        throw std::runtime_error("Invalid logsumexp denominator in route segment gradient.");
    }
    for (const RouteGradEndpointValue& entry : values) {
        const double weight = std::exp((entry.value_ns - max_value) / tau_ns) / sum_exp;
        bar_pin_at[entry.slot] += weight * time_to_ns;
        bar_pin_rat[entry.slot] -= weight * time_to_ns;
    }
}

bool nearly_equal_time(float a, float b)
{
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }
    const float scale = std::max({1.0f, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) <= scale * 1.0e-4f;
}

void reverse_endpoint_test_rat_to_related_at(const GPUTimer& timer,
                                             const std::vector<float>& pin_at,
                                             const std::vector<float>& pin_rat,
                                             const std::vector<float>& test_related_at,
                                             const std::vector<float>& test_rat,
                                             const std::vector<index_type>& pin_backward_arc_list_end,
                                             const std::vector<index_type>& pin_backward_arc_list,
                                             const std::vector<index_type>& timing_arc_from_pin_id,
                                             const std::vector<index_type>& timing_arc_to_pin_id,
                                             const std::vector<int>& arc_id2test_id,
                                             std::vector<double>& bar_pin_at,
                                             std::vector<double>& bar_pin_rat)
{
    if (timer.num_tests <= 0 || pin_backward_arc_list_end.empty()) {
        return;
    }
    const float period = timer.clock_period;
    for (int to_pin = 0; to_pin < timer.num_pins; ++to_pin) {
        if (to_pin + 1 >= static_cast<int>(pin_backward_arc_list_end.size())) {
            break;
        }
        const index_type arc_begin = pin_backward_arc_list_end[to_pin];
        const index_type arc_end = pin_backward_arc_list_end[to_pin + 1];
        if (arc_begin == arc_end) {
            continue;
        }
        for (int attr = 2; attr < NUM_ATTR; ++attr) {
            const int to_slot = to_pin * NUM_ATTR + attr;
            const double adj_rat = bar_pin_rat[to_slot];
            if (adj_rat == 0.0 || !std::isfinite(adj_rat)) {
                continue;
            }
            for (index_type pos = arc_begin; pos < arc_end; ++pos) {
                if (pos < 0 || pos >= static_cast<index_type>(pin_backward_arc_list.size())) {
                    continue;
                }
                const int arc_id = static_cast<int>(pin_backward_arc_list[pos]);
                if (arc_id < 0 || arc_id >= timer.num_arcs ||
                    arc_id >= static_cast<int>(arc_id2test_id.size())) {
                    continue;
                }
                const int test_id = arc_id2test_id[arc_id];
                if (test_id < 0 || test_id >= timer.num_tests) {
                    continue;
                }
                if (timing_arc_to_pin_id[arc_id] != to_pin) {
                    continue;
                }
                const int test_slot = test_id * NUM_ATTR + attr;
                if (!nearly_equal_time(test_rat[test_slot], pin_rat[to_slot])) {
                    continue;
                }
                const int from_pin = static_cast<int>(timing_arc_from_pin_id[arc_id]);
                if (from_pin < 0 || from_pin >= timer.num_pins) {
                    continue;
                }
                const float related = test_related_at[test_slot];
                const int el = attr >> 1;
                const int related_el = el ^ 1;
                for (int rf = 0; rf < 2; ++rf) {
                    const int related_attr = (related_el << 1) | rf;
                    const int related_slot = from_pin * NUM_ATTR + related_attr;
                    const float at = pin_at[related_slot];
                    if (nearly_equal_time(at, related) ||
                        (std::isfinite(period) && nearly_equal_time(at + period, related)) ||
                        (std::isfinite(period) && nearly_equal_time(at + 0.5f * period, related))) {
                        bar_pin_at[related_slot] += adj_rat;
                        break;
                    }
                }
            }
        }
    }
}

// Reverse the timing graph from endpoint objective adjoints to local RC outputs.
// This currently follows the existing DMP active AT predecessor data while using
// reconstructed primitive slopes for net/gate delay and slew sensitivities.
void reverse_active_at_to_elmore(const GPUTimer& timer,
                                 const std::vector<index_type>& level_list,
                                 const std::vector<index_type>& at_prefix_pin,
                                 const std::vector<index_type>& at_prefix_arc,
                                 const std::vector<index_type>& at_prefix_attr,
                                 const std::vector<uint8_t>& arc_types,
                                 const RouteGradNetSlopesHost& net_slopes,
                                 const RouteGradActiveGateSlopesHost& active_gate_slopes,
                                 const RouteGradGateSlewWinnerSlopesHost& gate_slew_slopes,
                                 std::vector<double>& bar_pin_at,
                                 std::vector<double>& bar_pin_slew,
                                 std::vector<double>& bar_elmore,
                                 std::vector<double>& bar_root_c1,
                                 std::vector<double>& bar_root_c2,
                                 std::vector<double>& bar_root_rpi)
{
    if (timer.level_list_end_cpu.size() < 2 || level_list.empty()) {
        return;
    }
    for (int level = static_cast<int>(timer.level_list_end_cpu.size()) - 2; level >= 0; --level) {
        const int start = timer.level_list_end_cpu[level];
        const int end = timer.level_list_end_cpu[level + 1];
        for (int pos = end - 1; pos >= start; --pos) {
            if (pos < 0 || pos >= static_cast<int>(level_list.size())) {
                continue;
            }
            const int to_pin = static_cast<int>(level_list[pos]);
            if (to_pin < 0 || to_pin >= timer.num_pins) {
                continue;
            }
            for (int to_attr = 0; to_attr < NUM_ATTR; ++to_attr) {
                const int to_slot = to_pin * NUM_ATTR + to_attr;
                const double adj_at = bar_pin_at[to_slot];
                const double adj_slew = bar_pin_slew[to_slot];
                const bool has_at = adj_at != 0.0 && std::isfinite(adj_at);
                const bool has_slew = adj_slew != 0.0 && std::isfinite(adj_slew);
                if (!has_at && !has_slew) {
                    continue;
                }
                const int from_pin = static_cast<int>(at_prefix_pin[to_slot]);
                const int arc_id = static_cast<int>(at_prefix_arc[to_slot]);
                const int from_attr = static_cast<int>(at_prefix_attr[to_slot]);
                if (from_pin < 0 || from_pin >= timer.num_pins ||
                    arc_id < 0 || arc_id >= timer.num_arcs ||
                    from_attr < 0 || from_attr >= NUM_ATTR) {
                    continue;
                }
                const int from_slot = from_pin * NUM_ATTR + from_attr;
                if (arc_types[arc_id] == 0) {
                    if (has_at) {
                        double delay_slope = 1.0;
                        if (to_slot >= 0 &&
                            to_slot < static_cast<int>(net_slopes.delay_elmore_slope.size())) {
                            const double candidate =
                                static_cast<double>(net_slopes.delay_elmore_slope[to_slot]);
                            if (std::isfinite(candidate) && candidate != 0.0) {
                                delay_slope = candidate;
                            }
                        }
                        bar_elmore[to_slot] += adj_at * delay_slope;
                        bar_pin_at[from_slot] += adj_at;
                    }
                    if (has_slew && to_slot >= 0 &&
                        to_slot < static_cast<int>(net_slopes.slew_elmore_slope.size())) {
                        const double slew_slope =
                            static_cast<double>(net_slopes.slew_elmore_slope[to_slot]);
                        if (std::isfinite(slew_slope) && slew_slope != 0.0) {
                            bar_elmore[to_slot] += adj_slew * slew_slope;
                        }
                    }
                    if (has_at && to_slot >= 0 &&
                        to_slot < static_cast<int>(net_slopes.delay_driver_root_slot.size())) {
                        const int root_slot = net_slopes.delay_driver_root_slot[to_slot];
                        if (root_slot >= 0 && root_slot < static_cast<int>(bar_root_c1.size())) {
                            bar_root_c1[root_slot] +=
                                adj_at * static_cast<double>(net_slopes.delay_c1_slope[to_slot]);
                            bar_root_c2[root_slot] +=
                                adj_at * static_cast<double>(net_slopes.delay_c2_slope[to_slot]);
                            bar_root_rpi[root_slot] +=
                                adj_at * static_cast<double>(net_slopes.delay_rpi_slope[to_slot]);
                        }
                        const int input_slot = net_slopes.delay_driver_input_slew_slot[to_slot];
                        if (input_slot >= 0 && input_slot < static_cast<int>(bar_pin_slew.size())) {
                            bar_pin_slew[input_slot] +=
                                adj_at * static_cast<double>(net_slopes.delay_input_slew_slope[to_slot]);
                        }
                    }
                    if (has_slew && to_slot >= 0 &&
                        to_slot < static_cast<int>(net_slopes.slew_driver_root_slot.size())) {
                        const int root_slot = net_slopes.slew_driver_root_slot[to_slot];
                        if (root_slot >= 0 && root_slot < static_cast<int>(bar_root_c1.size())) {
                            bar_root_c1[root_slot] +=
                                adj_slew * static_cast<double>(net_slopes.slew_c1_slope[to_slot]);
                            bar_root_c2[root_slot] +=
                                adj_slew * static_cast<double>(net_slopes.slew_c2_slope[to_slot]);
                            bar_root_rpi[root_slot] +=
                                adj_slew * static_cast<double>(net_slopes.slew_rpi_slope[to_slot]);
                        }
                        const int input_slot = net_slopes.slew_driver_input_slew_slot[to_slot];
                        if (input_slot >= 0 && input_slot < static_cast<int>(bar_pin_slew.size())) {
                            bar_pin_slew[input_slot] +=
                                adj_slew * static_cast<double>(net_slopes.slew_input_slew_slope[to_slot]);
                        }
                    }
                    continue;
                }

                if (arc_types[arc_id] != 1) {
                    continue;
                }
                if (has_at && to_slot >= 0 &&
                    to_slot < static_cast<int>(active_gate_slopes.root_slot.size())) {
                    const int root_slot = active_gate_slopes.root_slot[to_slot];
                    if (root_slot >= 0 && root_slot < static_cast<int>(bar_root_c1.size())) {
                        bar_root_c1[root_slot] +=
                            adj_at * static_cast<double>(active_gate_slopes.delay_c1_slope[to_slot]);
                        bar_root_c2[root_slot] +=
                            adj_at * static_cast<double>(active_gate_slopes.delay_c2_slope[to_slot]);
                        bar_root_rpi[root_slot] +=
                            adj_at * static_cast<double>(active_gate_slopes.delay_rpi_slope[to_slot]);
                        bar_pin_slew[from_slot] +=
                            adj_at * static_cast<double>(active_gate_slopes.delay_input_slew_slope[to_slot]);
                    }
                }
                if (has_slew && to_slot >= 0 &&
                    to_slot < static_cast<int>(gate_slew_slopes.root_slot.size())) {
                    const int slew_root_slot = gate_slew_slopes.root_slot[to_slot];
                    if (slew_root_slot >= 0 && slew_root_slot < static_cast<int>(bar_root_c1.size())) {
                        bar_root_c1[slew_root_slot] +=
                            adj_slew * static_cast<double>(gate_slew_slopes.slew_c1_slope[to_slot]);
                        bar_root_c2[slew_root_slot] +=
                            adj_slew * static_cast<double>(gate_slew_slopes.slew_c2_slope[to_slot]);
                        bar_root_rpi[slew_root_slot] +=
                            adj_slew * static_cast<double>(gate_slew_slopes.slew_rpi_slope[to_slot]);
                    }
                    const int slew_input_slot = gate_slew_slopes.input_slew_slot[to_slot];
                    if (slew_input_slot >= 0 && slew_input_slot < static_cast<int>(bar_pin_slew.size())) {
                        bar_pin_slew[slew_input_slot] +=
                            adj_slew * static_cast<double>(gate_slew_slopes.slew_input_slew_slope[to_slot]);
                    }
                }
                if (has_at) {
                    bar_pin_at[from_slot] += adj_at;
                }
            }
        }
    }
}

bool graph_includes_pin_caps(const HostRcGraph& graph, int net_id)
{
    return net_id >= 0 &&
           net_id < static_cast<int>(graph.includes_pin_caps.size()) &&
           graph.includes_pin_caps[net_id] != 0;
}

}  // namespace gt
