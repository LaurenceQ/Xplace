#pragma once

#include "gputimer/core/route_grad/DmpRouteGradHost.h"
#include "gputimer/core/openroad/OpenroadRcInternal.h"
#include "gputimer/core/GPUTimer.h"

#include <cuda_runtime.h>

#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gt {

void calc_res_cap_dmp(DmpModel* dmp_db, int num_nets);
void propagate_rc_tree_dmp(DmpModel* dmp_db, int num_nets);
void dmp_prepare_timing_after_rc(DmpModel* h_dmp_db, DmpModel* dmp_db);

void route_grad_cuda_check(cudaError_t err, const char* label);
void release_route_grad_dmp_state(GPUTimer& timer);
void run_route_segment_dmp_for_route_grad(GPUTimer& timer, HostRcGraph& graph);

template <typename T>
std::vector<T> copy_device_array(const T* ptr, size_t count, const char* label)
{
    std::vector<T> out(count);
    if (count == 0) {
        return out;
    }
    if (ptr == nullptr) {
        throw std::runtime_error(std::string("Missing device array for route segment gradient: ") +
                                 label);
    }
    route_grad_cuda_check(cudaMemcpy(out.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost),
                          label);
    return out;
}

RouteGradNetSlopesHost compute_net_primitive_slopes(
    GPUTimer& timer,
    unsigned long long* primitive_stats = nullptr);
RouteGradActiveGateSlopesHost compute_active_gate_primitive_slopes(
    GPUTimer& timer,
    unsigned long long* primitive_stats = nullptr);
RouteGradGateSlewWinnerSlopesHost compute_gate_slew_winner_slopes(
    GPUTimer& timer,
    unsigned long long* primitive_stats = nullptr);

std::vector<index_type> copy_level_list(const GPUTimer& timer);
std::vector<int64_t> endpoint_ids_cpu(const GPUTimer& timer);
void init_late_logsumexp_at_adjoint(const GPUTimer& timer,
                                    const std::vector<float>& pin_at,
                                    const std::vector<float>& pin_rat,
                                    double tau_ns,
                                    std::vector<double>& bar_pin_at,
                                    std::vector<double>& bar_pin_rat);
void reverse_endpoint_test_rat_to_related_at(const GPUTimer& timer,
                                             const std::vector<float>& pin_at,
                                             const std::vector<float>& pin_rat,
                                             const std::vector<float>& test_related_at_host,
                                             const std::vector<float>& test_rat_host,
                                             const std::vector<index_type>& pin_backward_arc_list_end_host,
                                             const std::vector<index_type>& pin_backward_arc_list_host,
                                             const std::vector<index_type>& timing_arc_from_pin_id_host,
                                             const std::vector<index_type>& timing_arc_to_pin_id_host,
                                             const std::vector<int>& arc_id2test_id_host,
                                             std::vector<double>& bar_pin_at,
                                             std::vector<double>& bar_pin_rat);
void reverse_active_at_to_elmore(const GPUTimer& timer,
                                 const std::vector<index_type>& level_list_host,
                                 const std::vector<index_type>& at_prefix_pin_host,
                                 const std::vector<index_type>& at_prefix_arc_host,
                                 const std::vector<index_type>& at_prefix_attr_host,
                                 const std::vector<uint8_t>& arc_types_host,
                                 const RouteGradNetSlopesHost& net_slopes,
                                 const RouteGradActiveGateSlopesHost& active_gate_slopes,
                                 const RouteGradGateSlewWinnerSlopesHost& gate_slew_slopes,
                                 std::vector<double>& bar_pin_at,
                                 std::vector<double>& bar_pin_slew,
                                 std::vector<double>& bar_elmore,
                                 std::vector<double>& bar_root_c1,
                                 std::vector<double>& bar_root_c2,
                                 std::vector<double>& bar_root_rpi);

double node_cap_with_optional_pin(const GPUTimer& timer,
                                  const HostRcGraph& graph,
                                  int net_id,
                                  int node,
                                  int attr);
void reverse_one_net_rc_tree(const GPUTimer& timer,
                             const HostRcGraph& graph,
                             int net_id,
                             double rc_time_factor,
                             const std::vector<double>& bar_elmore,
                             const std::vector<double>& bar_root_c1,
                             const std::vector<double>& bar_root_c2,
                             const std::vector<double>& bar_root_rpi,
                             std::vector<double>& edge_res_grad,
                             std::vector<double>& node_cap_grad);
std::vector<RouteGradRcTreeCheckSample> sample_rc_tree_check_nets(
    const HostRcGraph& graph,
    int sample_net_count,
    int seed);
double route_grad_check_weight(int net_id, int object_id, int attr, int channel);
void seed_rc_tree_check_adjoint(const GPUTimer& timer,
                                const HostRcGraph& graph,
                                const std::vector<RouteGradRcTreeCheckSample>& samples,
                                std::vector<double>& bar_elmore,
                                std::vector<double>& bar_root_c1,
                                std::vector<double>& bar_root_c2,
                                std::vector<double>& bar_root_rpi);
double route_grad_rc_tree_node_cap_base(const HostRcGraph& graph, int node);
double route_grad_rc_tree_safe_rel_err(double fd, double adj);
double forward_one_net_rc_tree_value(const GPUTimer& timer,
                                     const HostRcGraph& graph,
                                     int net_id,
                                     double rc_time_factor,
                                     const std::vector<double>& bar_elmore,
                                     const std::vector<double>& bar_root_c1,
                                     const std::vector<double>& bar_root_c2,
                                     const std::vector<double>& bar_root_rpi);
std::vector<std::string> dmp_route_segment_rc_tree_gradcheck_columns();
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> make_route_grad_tensors(
    const HostRcGraph& graph,
    const std::vector<double>& edge_res_grad,
    const std::vector<double>& node_cap_grad);
std::vector<std::string> dmp_route_segment_primitive_slope_stat_columns();

}  // namespace gt
