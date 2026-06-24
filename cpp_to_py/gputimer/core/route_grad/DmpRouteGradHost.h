#pragma once

#include "gputimer/core/openroad/OpenroadRcInternal.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gt {

class GPUTimer;

inline constexpr int kRouteGradRcTreeCheckColumns = 13;

void route_grad_cuda_check(cudaError_t err, const char* label);

struct RouteGradDeviceFloatBuffer {
    float* ptr = nullptr;

    RouteGradDeviceFloatBuffer();
    RouteGradDeviceFloatBuffer(const RouteGradDeviceFloatBuffer&) = delete;
    RouteGradDeviceFloatBuffer& operator=(const RouteGradDeviceFloatBuffer&) = delete;
    ~RouteGradDeviceFloatBuffer();

    void allocate(size_t count, const char* label);
};

struct RouteGradDeviceIntBuffer {
    int* ptr = nullptr;

    RouteGradDeviceIntBuffer();
    RouteGradDeviceIntBuffer(const RouteGradDeviceIntBuffer&) = delete;
    RouteGradDeviceIntBuffer& operator=(const RouteGradDeviceIntBuffer&) = delete;
    ~RouteGradDeviceIntBuffer();

    void allocate(size_t count, const char* label);
};

struct RouteGradDeviceU64Buffer {
    unsigned long long* ptr = nullptr;

    RouteGradDeviceU64Buffer();
    RouteGradDeviceU64Buffer(const RouteGradDeviceU64Buffer&) = delete;
    RouteGradDeviceU64Buffer& operator=(const RouteGradDeviceU64Buffer&) = delete;
    ~RouteGradDeviceU64Buffer();

    void allocate(size_t count, const char* label);
};

struct RouteGradNetSlopesHost {
    std::vector<float> delay_elmore_slope;
    std::vector<float> slew_elmore_slope;
    std::vector<int> delay_driver_root_slot;
    std::vector<int> delay_driver_input_slew_slot;
    std::vector<int> slew_driver_root_slot;
    std::vector<int> slew_driver_input_slew_slot;
    std::vector<float> delay_c1_slope;
    std::vector<float> delay_c2_slope;
    std::vector<float> delay_rpi_slope;
    std::vector<float> delay_input_slew_slope;
    std::vector<float> slew_c1_slope;
    std::vector<float> slew_c2_slope;
    std::vector<float> slew_rpi_slope;
    std::vector<float> slew_input_slew_slope;
};

struct RouteGradActiveGateSlopesHost {
    std::vector<int> root_slot;
    std::vector<float> delay_c1_slope;
    std::vector<float> delay_c2_slope;
    std::vector<float> delay_rpi_slope;
    std::vector<float> delay_input_slew_slope;
    std::vector<float> slew_c1_slope;
    std::vector<float> slew_c2_slope;
    std::vector<float> slew_rpi_slope;
    std::vector<float> slew_input_slew_slope;
};

struct RouteGradGateSlewWinnerSlopesHost {
    std::vector<int> root_slot;
    std::vector<int> input_slew_slot;
    std::vector<float> slew_c1_slope;
    std::vector<float> slew_c2_slope;
    std::vector<float> slew_rpi_slope;
    std::vector<float> slew_input_slew_slope;
};

struct RouteGradEndpointValue {
    int slot = -1;
    double value_ns = 0.0;
};

struct RouteGradRcTreeCheckSample {
    int net_id = -1;
    int edge_id = -1;
    int node_id = -1;
};

struct RouteGradFdSample {
    int net_id = -1;
    int edge_id = -1;
    int node_id = -1;
};


}  // namespace gt
