#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace gt {

struct DmpRouteGradOptions {
    double tau_ns = 0.02;
};

struct DmpRouteGradFdValidateOptions {
    int sample_net_count = 10000;
    int seed = 1;
    double eps_rel = 1.0e-3;
    double eps_abs_edge = 0.0;
    double eps_abs_node = 1.0e-4;
    double tau_ns = 0.02;
};

std::vector<std::string> dmp_route_segment_grad_fd_validate_columns();

}  // namespace gt
