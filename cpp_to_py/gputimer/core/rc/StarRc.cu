#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "gputimer/core/rc/RcModels.h"
#include "gputimer/core/utils.cuh"

namespace gt {

__global__ void RCTreeNet(RcStarNet* star_net) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float* x = star_net->x;
    float* y = star_net->y;
    const float* pin_offset_x = star_net->pin_offset_x;
    const float* pin_offset_y = star_net->pin_offset_y;
    const int* pin2node_map = star_net->pin2node_map;
    const int* flat_net2pin_start_map = star_net->flat_net2pin_start_map;
    const int* flat_net2pin_map = star_net->flat_net2pin_map;
    float* pinLoad = star_net->pinLoad;
    float* pinImpulse = star_net->pinImpulse;
    float* pinCap = star_net->pinCap;
    float* pinRootDelay = star_net->pinRootDelay;
    float* pinRootRes = star_net->pinRootRes;
    const int num_nets = star_net->num_nets;
    const float unit_to_micron = star_net->unit_to_micron;
    const uint8_t* net_is_clock = star_net->net_is_clock;
    const float cf = star_net->cf;
    const float rf = star_net->rf;
    if (idx < num_nets) {
        int start_idx = flat_net2pin_start_map[idx];
        int end_idx = flat_net2pin_start_map[idx + 1];
        int root = flat_net2pin_map[start_idx];
        float x_root = x[pin2node_map[root]] + pin_offset_x[root];
        float y_root = y[pin2node_map[root]] + pin_offset_y[root];
        float root_cap = 0;

        // Load
        for (int i = start_idx + 1; i < end_idx; i++) {
            int pin_id = flat_net2pin_map[i];
            float x_pin = x[pin2node_map[pin_id]] + pin_offset_x[pin_id];
            float y_pin = y[pin2node_map[pin_id]] + pin_offset_y[pin_id];
            float dist = abs(x_pin - x_root) + abs(y_pin - y_root);
            float wl = dist / unit_to_micron;
            if (net_is_clock[idx]) wl = 0;
            float pin_cap = cf * wl * 0.5;
            float pin_res = rf * wl;
            root_cap += pin_cap;

            for (int j = 0; j < NUM_ATTR; j++) {
                float pin_cap_lib = pin_cap_attr(pinCap, pin_id, j);
                float load = pinLoad[pin_id * NUM_ATTR + j];

                pinLoad[pin_id * NUM_ATTR + j] = isnan(load) ? pin_cap + pin_cap_lib : load + pin_cap + pin_cap_lib;
                pinRootRes[pin_id * NUM_ATTR + j] = pin_res;
                pinLoad[root * NUM_ATTR + j] = isnan(pinLoad[root * NUM_ATTR + j]) ? pinLoad[pin_id * NUM_ATTR + j]
                                                                                   : pinLoad[root * NUM_ATTR + j] + pinLoad[pin_id * NUM_ATTR + j];
            }
        }
        // Root
        for (int j = 0; j < NUM_ATTR; j++) {
            float pin_cap_lib = pin_cap_attr(pinCap, root, j);
            float load = pinLoad[root * NUM_ATTR + j];
            pinLoad[root * NUM_ATTR + j] = isnan(load) ? root_cap + pin_cap_lib : load + root_cap + pin_cap_lib;
        }
        // Delay
        for (int i = start_idx + 1; i < end_idx; i++) {
            int pin_id = flat_net2pin_map[i];
            for (int j = 0; j < NUM_ATTR; j++) {
                pinRootDelay[pin_id * NUM_ATTR + j] = pinRootRes[pin_id * NUM_ATTR + j] * pinLoad[pin_id * NUM_ATTR + j];
                pinImpulse[pin_id * NUM_ATTR + j] = 0;
            }
        }
        // Impulse
        for (int i = start_idx + 1; i < end_idx; i++) {
            int pin_id = flat_net2pin_map[i];
            for (int j = 0; j < NUM_ATTR; j++) {
                float pin_cap_lib = pin_cap_attr(pinCap, pin_id, j);
                float res = pinRootRes[pin_id * NUM_ATTR + j];
                float cap = pinLoad[pin_id * NUM_ATTR + j];
                float delay = pinRootDelay[pin_id * NUM_ATTR + j];
                pinImpulse[pin_id * NUM_ATTR + j] = sqrt(2 * res * cap * delay - delay * delay);
            }
        }
        if (end_idx - start_idx == 1) {
            for (int j = 0; j < NUM_ATTR; j++) {
                pinRootDelay[root * NUM_ATTR + j] = 0;
                pinImpulse[root * NUM_ATTR + j] = 0;
            }
        }
    }
}

void update_rc_timing_cuda(const RcStarNet& star_net) {
    const int num_nets = star_net.num_nets;
    (void)star_net.num_pins;
    RcStarNet* d_star_net = nullptr;
    cudaMalloc(&d_star_net, sizeof(RcStarNet));
    cudaMemcpy(d_star_net, &star_net, sizeof(RcStarNet), cudaMemcpyHostToDevice);
    RCTreeNet<<<BLOCK_NUMBER(num_nets), BLOCK_SIZE>>>(d_star_net);
    cudaFree(d_star_net);
}


}  // namespace gt
